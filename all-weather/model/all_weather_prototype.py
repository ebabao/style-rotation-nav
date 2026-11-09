#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
all_weather_prototype.py
全天候策略模型开发 · 最小可运行原型（研究准备阶段骨架）

用途
----
把「参考文章 → 模型设计」中最核心的一条链路跑通：

    资产日收益 → 协方差估计 → 风险预算 / 风险平价求解 → 风险贡献诊断
    → 目标波动杠杆 → 再平衡回测 → 绩效与风险指标

重要声明
--------
demo() 使用【固定随机种子生成的合成数据】，仅用于验证代码链路与数值正确性，
不含任何真实行情，不构成任何回测结论、业绩展示或投资建议。
接入真实数据只需替换 load_returns()，返回 DataFrame（index=日期，columns=资产，值=日收益）。

设计原则（来自两篇参考文章）
    P1 不预测：不引入宏观方向判断、不择时，优化目标只用协方差矩阵。
    P2 风险口径统一：以「风险贡献」而非「资金权重」衡量均衡。
    P3 主体与增强层分离：本原型只实现基础风险平价层，CTA / 期权等增强层留开关。

依赖：numpy, pandas, scipy, matplotlib（仅画图用，可选）
运行：python3.11 all_weather_prototype.py [--no-plot]
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

TRADING_DAYS = 252
EPS = 1e-12


# ----------------------------------------------------------------------------
# 0. 数据接口（替换为真实数据源即可）
# ----------------------------------------------------------------------------
def load_returns() -> pd.DataFrame:
    """合成数据：4 类资产、8 年日频。仅用于跑通链路。"""
    rng = np.random.default_rng(20261108)
    n_days = TRADING_DAYS * 8

    names = ["Equity", "LongBond", "Gold", "Cash"]
    vol_annual = np.array([0.22, 0.05, 0.15, 0.01])       # 年化波动率假设
    mu_daily = np.array([0.00025, 0.00012, 0.00018, 0.00007])

    corr = np.array([
        [1.00, -0.15,  0.05, 0.00],                       # 股债弱负相关
        [-0.15, 1.00,  0.10, 0.00],
        [0.05,  0.10,  1.00, 0.00],
        [0.00,  0.00,  0.00, 1.00],
    ])
    d = np.diag(vol_annual / np.sqrt(TRADING_DAYS))
    cov_daily = d @ corr @ d
    chol = np.linalg.cholesky(cov_daily + np.eye(4) * EPS)

    z = rng.standard_normal((n_days, 4))
    r = mu_daily + z @ chol.T
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    return pd.DataFrame(r, index=dates, columns=names)


# ----------------------------------------------------------------------------
# 1. 协方差估计：文章 A 2.2 指出「波动率计算窗口」是各家独门秘籍
# ----------------------------------------------------------------------------
def estimate_cov(returns: pd.DataFrame, method: str = "shrink", window: int | None = None,
                 halflife: float | None = None, shrinkage: float = 0.20) -> np.ndarray:
    """
    method:
        sample  - 滚动样本协方差
        ewma    - 指数加权（半衰期 halflife，交易日）
        shrink  - 对角收缩（简化版 Ledoit-Wolf 风格，正式版建议 sklearn LedoitWolf/OAS）
    window: 只用最近 window 个交易日；None 表示全样本。
    """
    r = returns if window is None else returns.iloc[-min(window, len(returns)):]
    x = np.asarray(r, dtype=float)
    t, n = x.shape
    xc = x - x.mean(axis=0)

    if method == "sample":
        cov = (xc.T @ xc) / max(t - 1, 1)
    elif method == "ewma":
        lam = 0.94 if halflife is None else float(np.exp(-np.log(2.0) / halflife))
        w = lam ** np.arange(t - 1, -1, -1)               # 越近权重越大
        w = w / w.sum()
        cov = (xc * w[:, None]).T @ xc
    elif method == "shrink":
        s = (xc.T @ xc) / max(t - 1, 1)
        cov = (1.0 - shrinkage) * s + shrinkage * np.diag(np.diag(s))
    else:
        raise ValueError(f"unknown cov method: {method}")

    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)                              # 数值对称化
    return cov + np.eye(n) * EPS                            # 保证正定


def blended_cov(returns: pd.DataFrame, windows=(60, 120, 250),
                alphas=(0.2, 0.3, 0.5), **kwargs) -> np.ndarray:
    """长 / 中 / 短窗口加权，对应文章 A「长中短周期结合做综合考虑」的落地方式。"""
    a = np.asarray(alphas, dtype=float)
    a = a / a.sum()
    return sum(
        w * estimate_cov(returns, window=k, **kwargs) for k, w in zip(windows, a)
    )


# ----------------------------------------------------------------------------
# 2. 风险贡献（Euler 分解）与风险平价求解
# ----------------------------------------------------------------------------
def risk_contribution(w: np.ndarray, cov: np.ndarray):
    """返回 (风险贡献绝对值 RC, 风险占比 pct, 组合波动 sigma_p)。Σ RC_i = sigma_p。"""
    w = np.asarray(w, dtype=float)
    sigma_p = float(np.sqrt(w @ cov @ w))
    mrc = cov @ w / sigma_p                                # 边际风险贡献
    rc = w * mrc                                           # 总风险贡献
    return rc, rc / sigma_p, sigma_p


def solve_risk_budget(cov: np.ndarray, budget: np.ndarray | None = None):
    """
    Spinu (2013) 凸形式：
        min_{x>0}  0.5 * xᵀΣx - Σ_i b_i * ln(x_i)
        ∇f(x) = Σx - b/x = 0  →  x_i (Σx)_i = b_i
        w = x / Σx
    等价于 RC_i / sigma_p = b_i，且为全局最优（凸问题），不依赖初始点。
    """
    from scipy.optimize import minimize

    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    b = np.full(n, 1.0 / n) if budget is None else np.asarray(budget, dtype=float)
    b = b / b.sum()

    def obj(x):
        return 0.5 * x @ cov @ x - b @ np.log(x)

    def grad(x):
        return cov @ x - b / x

    res = minimize(obj, np.ones(n), jac=grad, method="L-BFGS-B",
                   bounds=[(EPS, None)] * n,
                   options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 5000})
    x = np.clip(res.x, EPS, None)
    w = x / x.sum()
    _, pct, sigma_p = risk_contribution(w, cov)
    return w, pct, sigma_p, {"success": bool(res.success), "message": str(res.message),
                             "max_deviation": float(np.max(np.abs(pct - b)))}


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    """逆波动率近似：仅在相关系数全为 0（或全相等）时才等于严格风险平价。"""
    iv = 1.0 / np.sqrt(np.diag(cov))
    return iv / iv.sum()


def four_quadrant_budget(assets_per_quadrant: dict[str, list[str]],
                         quadrant_weights=(0.25, 0.25, 0.25, 0.25)) -> dict:
    """
    文章 A 1.2：增长/通胀 × 超预期/不及预期，每象限 25% 风险。
    这里把象限风险预算平均分给该象限内的资产，得到资产层风险预算向量。
    层次化用法：先象限层再类内层 ERC；M0 阶段建议直接用等风险预算（见说明文档）。
    """
    out = {}
    for (q, members), qw in zip(assets_per_quadrant.items(), quadrant_weights):
        for m in members:
            out[m] = qw / len(members)
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


# ----------------------------------------------------------------------------
# 3. 目标波动与杠杆（含被动降仓，对应文章 B「风险平价配置可能触发降仓」）
# ----------------------------------------------------------------------------
def target_vol_leverage(w: np.ndarray, cov: np.ndarray, target_vol: float,
                        max_leverage: float = 2.0) -> tuple[float, float]:
    """返回 (杠杆倍数 L, 年化组合波动)。L = σ_target / σ_p，超限则截断。"""
    sigma_ann = float(np.sqrt(w @ cov @ w) * np.sqrt(TRADING_DAYS))
    leverage = target_vol / max(sigma_ann, EPS)
    return float(min(leverage, max_leverage)), sigma_ann


# ----------------------------------------------------------------------------
# 4. 再平衡回测骨架
# ----------------------------------------------------------------------------
def rebalance_backtest(returns: pd.DataFrame, cov_estimator, budget=None,
                       freq: str = "M", cost_bps: float = 5.0, band: float = 0.05,
                       target_vol: float | None = None, max_leverage: float = 1.0,
                       min_history: int = 120):
    """
    freq: 'M' 每月首个交易日 / 'Q' 每季首个交易日调仓
    band: 权重最大偏离小于该阈值则不调仓（抑制噪音与换手，文章 A 2.2）
    cov_estimator: callable(hist_returns) -> cov，只用调仓日之前的数据，避免前视偏差

    简化假设（正式版需补齐）：现金部分零收益、无冲击成本、无展期/折溢价、
    权重日内漂移忽略、杠杆无融资成本。
    """
    n = returns.shape[1]
    key_series = pd.Series(returns.index.to_period(freq), index=returns.index)
    first_of_period = key_series != key_series.shift(1)

    w = np.full(n, 1.0 / n)
    leverage = 1.0
    port_rets, log = [], []

    for i, (dt, row) in enumerate(returns.iterrows()):
        turnover = 0.0
        if i == 0 or bool(first_of_period.iloc[i]):
            hist = returns.iloc[:i]                        # 严格不含当日
            if len(hist) >= min_history:
                cov = cov_estimator(hist)
                w_new, _, _, _ = solve_risk_budget(cov, budget)
                if np.max(np.abs(w_new - w)) >= band:
                    turnover = np.abs(w_new - w).sum() / 2.0
                    w = w_new
                if target_vol is not None:
                    leverage, _ = target_vol_leverage(w, cov, target_vol, max_leverage)
                log.append({"date": dt, "turnover": turnover, "leverage": leverage,
                            **{f"w_{c}": w[j] for j, c in enumerate(returns.columns)}})

        r_p = leverage * float(np.asarray(row.values) @ w) - turnover * cost_bps / 1e4
        port_rets.append(r_p)

    nav = (1.0 + pd.Series(port_rets, index=returns.index)).cumprod()
    return nav, pd.DataFrame(log)


def metrics(nav: pd.Series, rf: float = 0.0) -> dict:
    """含文章 B 强调的『回撤修复时间』与长周期夏普基准对照。"""
    r = nav.pct_change().dropna()
    years = len(r) / TRADING_DAYS
    total = float(nav.iloc[-1] / nav.iloc[0])
    cagr = total ** (1.0 / years) - 1.0
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (cagr - rf) / max(vol, EPS)

    peak = nav.cummax()
    dd = nav / peak - 1.0
    mdd = float(dd.min())
    trough = dd.idxmin()
    peak_at_trough = float(peak.loc[trough])
    after = nav.loc[trough:]
    recovered = after[after >= peak_at_trough]
    recovery_days = None if recovered.empty else int(len(after.loc[trough:recovered.index[0]]) - 1)

    return {
        "年化收益": cagr,
        "年化波动": vol,
        "夏普比率": sharpe,
        "最大回撤": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "回撤修复交易日": "未修复" if recovery_days is None else recovery_days,
        "期末净值": total,
    }


# ----------------------------------------------------------------------------
# 5. Demo：合成数据验证链路 + 对照表 + 诊断图
# ----------------------------------------------------------------------------
def fmt(x, pct=True):
    return f"{x * 100:6.2f}%" if pct else f"{x:8.4f}"


def format_metric(key, value):
    """区分百分比 / 倍数 / 整数，避免把夏普比率显示成百分比。"""
    if isinstance(value, str) or key == "回撤修复交易日":
        return str(value)
    if key in {"年化收益", "年化波动", "最大回撤"}:
        return f"{value * 100:.2f}%"
    if key in {"夏普比率", "Calmar"}:
        return f"{value:.2f}"
    if key == "期末净值":
        return f"{value:.4f}x"
    return f"{value:.4f}"


def demo(plot: bool = True):
    print("=" * 78)
    print("声明：以下全部结果基于【固定种子合成数据】，仅验证代码链路与数值正确性，")
    print("      不含真实行情，不构成任何回测结论或投资建议。")
    print("=" * 78)

    r = load_returns()
    cov = blended_cov(r)

    print("\n[1] 协方差估计（长中短窗口加权 + 收缩）年化波动率：")
    for c, v in zip(r.columns, np.sqrt(np.diag(cov)) * np.sqrt(TRADING_DAYS)):
        print(f"    {c:<10s} {fmt(v)}")

    w_erc, pct_erc, sigma_erc, info = solve_risk_budget(cov)
    print(f"\n[2] 风险平价求解：{info}")
    print(f"    组合日波动 {sigma_erc:.6f} / 年化 {fmt(sigma_erc * np.sqrt(TRADING_DAYS))}")

    # 文章 A 给出的桥水经典资金比例（此处按 4 资产映射：股票/长债/黄金/现金）
    article_mix = np.array([0.30, 0.40, 0.15, 0.15])
    candidates = {
        "资金等权":       np.full(4, 0.25),
        "文章经典比例":   article_mix,
        "逆波动率近似":   inverse_vol_weights(cov),
        "风险平价 ERC":   w_erc,
    }

    print("\n[3] 资金权重 vs 风险贡献占比（核心对照，验证『配置风险而非资金』）：")
    header = f"{'组合':<14s}" + "".join(f"{c:>11s}" for c in r.columns) + f"{'年化波动':>12s}"
    print("    " + header)
    for name, w in candidates.items():
        _, pct, sig = risk_contribution(w, cov)
        ann = sig * np.sqrt(TRADING_DAYS)
        print(f"    {name:<14s}" + "".join(f"{fmt(x):>11s}" for x in w) + f"{fmt(ann):>12s}")
        print(f"    {'  └ 风险占比':<14s}" + "".join(f"{fmt(x):>11s}" for x in pct))

    print("\n    注：文章经典比例 30/40/15/15 按资金分配，股票仍贡献绝大部分风险；")
    print("        ERC 把风险占比拉平到各 25%，代价是债券资金占比进一步升高，")
    print("        因此需要目标波动杠杆才能把收益抬回可比水平（文章 A 2.1）。")

    if np.max(np.abs(pct_erc - 0.25)) > 1e-6:
        raise RuntimeError("风险平价求解未收敛到等风险贡献，请检查协方差矩阵")

    # 四象限风险预算示例
    quad = {
        "增长超预期": ["Equity"],
        "增长不及预期": ["LongBond"],
        "通胀超预期": ["Gold"],
        "通胀不及预期": ["Cash"],
    }
    b = four_quadrant_budget(quad)
    w_q, pct_q, _, _ = solve_risk_budget(cov, np.array([b[c] for c in r.columns]))
    print(f"\n[4] 四象限各 25% 风险预算（本例中恰好各 1 个资产，与 ERC 一致）：")
    print("    " + "  ".join(f"{k}:{fmt(v)}" for k, v in b.items()))

    # 回测骨架
    erc_nav, erc_log = rebalance_backtest(r, blended_cov, freq="M", band=0.05, cost_bps=5.0)
    eq_nav = (1.0 + r.mean(axis=1)).cumprod()              # 等权对照（隐含每日再平衡，近似基准）

    print("\n[5] 回测骨架指标（月度再平衡，5bp 单向换手成本，合成数据）：")
    print(f"    {'指标':<16s}{'风险平价 ERC':>16s}{'资金等权':>16s}")
    m1, m2 = metrics(erc_nav), metrics(eq_nav)
    for k in m1:
        print(f"    {k:<16s}{format_metric(k, m1[k]):>16s}{format_metric(k, m2[k]):>16s}")
    print("    注：合成数据的收益率/夏普不具任何参考意义，此处仅验证指标计算可运行。")
    print(f"    调仓次数：{len(erc_log)}，平均单次换手："
          f"{erc_log['turnover'].mean() * 100:.2f}%")

    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = list(r.columns)
            eng = {
                "资金等权": "Equal capital",
                "文章经典比例": "Article 30/40/15/15",
                "逆波动率近似": "Inverse vol",
                "风险平价 ERC": "Risk parity ERC",
            }
            x = np.arange(4)
            width = 0.2
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
            colors = ["#c0392b", "#2c5282", "#b8860b", "#7f8c8d"]

            for ax, mode in zip(axes, ["weight", "risk"]):
                for j, (name, w) in enumerate(candidates.items()):
                    vals = w if mode == "weight" else risk_contribution(w, cov)[1]
                    ax.bar(x + (j - 1.5) * width, vals, width, label=eng[name], color=colors[j],
                           alpha=0.9, edgecolor="white", linewidth=0.6)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=9)
                ax.set_ylim(0, 1.0)
                ax.set_ylabel("Capital weight" if mode == "weight" else "Risk contribution")
                ax.set_title("Capital allocation" if mode == "weight" else "Risk contribution (Euler)")
                ax.grid(axis="y", alpha=0.3)
                ax.legend(fontsize=8, ncol=2, frameon=False)
            fig.suptitle("Risk Parity vs Capital Weighting  |  SYNTHETIC DATA - NOT A BACKTEST",
                         fontsize=11, fontweight="bold", color="#555")
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            out = "/workspace/all_weather_demo_risk_contribution.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\n[6] 诊断图已输出：{out}")
        except Exception as e:                                  # noqa: BLE001
            print(f"\n[6] 画图跳过：{e}")

    print("\n" + "=" * 78)
    print("链路验证完成。下一步：替换 load_returns() 为真实行情，按说明文档 M0~M5 推进。")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="全天候策略模型最小原型（合成数据）")
    ap.add_argument("--no-plot", action="store_true", help="不生成诊断图")
    args = ap.parse_args()
    demo(plot=not args.no_plot)
