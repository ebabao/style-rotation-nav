# -*- coding: utf-8 -*-
"""第29章29.3 · 复现"高 AUC ≠ 好的避险触发器"（诚实对照回测）

核心实验（对齐文档）：
  把单因子 AUC 最高的 fip252(0.674) 直接当避险开关：信号触发 → 次日退守债券ETF。
  对照冠军（冻结 alpha 不做择时）。预期：收益崩塌、回撤未必改善——因为
  fip252 是"趋势平滑度/拥挤度"因子，平稳上涨后续涨（IC 为正），
  在主升浪中反复误报、切掉盈利仓位（文档原话）。

  同时验证 29.4 的正确做法：因子作为"观察指标"不直接切换 → 净值=冠军不变。

零前视：信号 f[t] 在 t 收盘可得，t+1 起生效（shift(1)）；成本单边千一。
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from timing_data import load_config, dump_json, OUT_DATA  # noqa: E402
from timing_engine import frozen_params, lock_params, run_full  # noqa: E402
from git_factors import build_factors, load_market_panel  # noqa: E402

CFG = load_config()
COST = CFG["data"]["cost_bps"] / 10000.0


def perf(nav: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-6)
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    sd = r.std()
    sh = r.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
    mdd = float((nav / nav.cummax() - 1).min())
    return dict(cagr=float(cagr), maxdd=float(mdd), sharpe=float(sh),
                calmar=float(cagr / max(abs(mdd), 1e-6)))


def overlay_backtest(signal: pd.Series, trigger_rule: str, q=0.10) -> dict:
    """把信号转成避险开关，叠加到冠军净值上。t 信号 t+1 生效，退守期间持债券日收益。

    trigger_rule:
      'low'      = 信号 ≤ 历史 q 分位 → 避险（fip252 低=不平滑，危险）
      'high'     = 信号 ≥ 历史 (1-q) 分位 → 避险（kurt/拥挤度高=危险）
    """
    champ_eq = np.asarray(run_full(frozen_params())["equity"], float)
    eq_dates = pd.DatetimeIndex(run_full(frozen_params())["dates"])
    champ_nav = pd.Series(champ_eq, index=eq_dates)
    champ_ret = champ_nav.pct_change()

    panel = load_market_panel()
    bond_ret = panel["asset_close"][CFG["universe"]["bond"]].pct_change() if CFG["universe"]["bond"] in panel["asset_close"] else pd.Series(0.0, index=panel["close"].index)
    bond_ret = bond_ret.reindex(champ_nav.index).fillna(0.0)

    sig = signal.reindex(champ_nav.index)
    if trigger_rule == "low":
        thr = sig.rolling(252, min_periods=60).quantile(q)
        danger = sig <= thr
    else:
        thr = sig.rolling(252, min_periods=60).quantile(1 - q)
        danger = sig >= thr
    danger = danger.shift(1).fillna(False).to_numpy()   # t 信号 t+1 生效

    # 切换成本：每次进出各扣单边成本
    switch = np.abs(np.diff(danger.astype(int), prepend=danger[0]))
    cost_drag = switch * COST
    r = champ_ret.to_numpy()
    eff = np.where(danger, bond_ret.to_numpy(), r) - cost_drag
    eff = np.nan_to_num(eff)
    nav = np.cumprod(1 + eff)
    out_nav = pd.Series(nav, index=champ_nav.index)
    return dict(nav=out_nav, danger_days=int(danger.sum()), n_switch=int(switch.sum()),
                danger_ratio=float(danger.mean()))


def run_backtest():
    panel = load_market_panel()
    F = build_factors(panel)
    champ_res = run_full(frozen_params())
    champ_nav = pd.Series(np.asarray(champ_res["equity"], float),
                          index=pd.DatetimeIndex(champ_res["dates"]))
    champ_m = perf(champ_nav)
    print(f"冠军(冻结alpha) 年化 {champ_m['cagr']*100:.2f}% 回撤 {champ_m['maxdd']*100:.2f}% "
          f"夏普 {champ_m['sharpe']:.4f}")

    print("\n=== 29.3 把高AUC因子直接当避险开关（对照回测）===")
    experiments = {}
    # fip252: AUC 0.674 最强，低=fip少=趋势不平滑 → 避险
    for fac, rule, q in [("fip252", "low", 0.10), ("fip252", "low", 0.20),
                         ("diffusion20", "low", 0.10), ("kurt20", "high", 0.10)]:
        r = overlay_backtest(F[fac], rule, q)
        m = perf(r["nav"])
        tag = f"{fac}|{rule}|q{q}"
        experiments[tag] = dict(metrics=m, danger_days=r["danger_days"],
                                n_switch=r["n_switch"], danger_ratio=round(r["danger_ratio"], 3),
                                nav=[round(float(v), 6) for v in r["nav"].to_numpy()],
                                dates=[str(d.date()) for d in r["nav"].index])
        print(f"  {tag:28s} 年化 {m['cagr']*100:6.2f}% 回撤 {m['maxdd']*100:7.2f}% "
              f"夏普 {m['sharpe']:.4f}  避险占比 {r['danger_ratio']:.1%} 切换 {r['n_switch']}次")

    # 29.4 正确做法：因子仅作观察指标，不切换 → 净值=冠军
    print("\n=== 29.4 正确做法：因子仅作监管观察指标（不直接切换）===")
    print(f"  fip252/kurt20 纳入观察 → 净值 unchanged = 冠军 "
          f"年化 {champ_m['cagr']*100:.2f}% 夏普 {champ_m['sharpe']:.4f}")

    # 崩塌幅度（最关键证据）
    key = "fip252|low|q0.1"
    collapse = experiments[key]["metrics"]
    cagr_drop = (champ_m["cagr"] - collapse["cagr"]) * 100
    mdd_change = (collapse["maxdd"] - champ_m["maxdd"]) * 100
    verdict = ("证伪：高AUC因子当避险开关导致收益崩塌" if cagr_drop > 5 else "未显著崩塌")
    print(f"\n[29.3结论] fip252(AUC最高)直接当开关：年化 {champ_m['cagr']*100:.2f}%→{collapse['cagr']*100:.2f}% "
          f"(跌 {cagr_drop:.1f}pp)，回撤 {champ_m['maxdd']*100:.2f}%→{collapse['maxdd']*100:.2f}% "
          f"({mdd_change:+.1f}pp) → {verdict}")

    out = dict(
        method=dict(label="fip252低分位避险(趋势不平滑)", execution="t信号t+1生效",
                    cost_bps=CFG["data"]["cost_bps"], detrend_note="零前视"),
        champion=champ_m,
        experiments=experiments,
        comparison=dict(key=key, cagr_drop_pp=round(float(cagr_drop), 2),
                        mdd_change_pp=round(float(mdd_change), 2), verdict=verdict),
        adoption_29_4=dict(
            observe_only=["fip252", "kurt20", "diffusion20"],
            note="因子纳入监管观察指标(拥挤/尾部风险)+ML预警模型输入, 不作为直接避险开关; 只提名不切换",
            nav_unchanged=True),
        note="本组合含黄金/纳指, 波动与事件率大于文档A股宽基面板; 结论方向一致但数值不同, 诚实披露。",
    )
    dump_json(out, OUT_DATA / "git_factor_backtest.json")
    # 看板接口
    dump_json(dict(backtest=out), OUT_DATA / "git_factor_trigger_report.json")
    print("SAVED git_factor_backtest.json / git_factor_trigger_report.json")
    return out


if __name__ == "__main__":
    run_backtest()
