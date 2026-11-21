# -*- coding: utf-8 -*-
"""择时因子机构版 · 进化学习监管层 + 影子台账（移植 sr-nav/evolution_guard.py 范式）

方法论（核桃量化《强化学习行业配置》2023-09-22）的可解释工程落地：
  本实现不是黑箱 DQN，而是表格化 Q-learning 监管层，用于对抗择时因子的
  IC 反转 / 冠军退化 / 数据滞后，且只提名不切换线上策略。

移植映射（sr-nav 行业轮动 → 本工程择时）：
  冠军 Champion   = 冻结 alpha（v2.0 零杠杆冠军，timing=none），逐点净值硬校验
  影子 Challengers= 围绕冠军的择时参数变体（tsmom/vol_managed/ma_smooth/multi_vote 等），
                   用同一 Engine20 独立回测，只记录虚拟净值，不影响冠军
  State 状态      = 冠军滚动夏普/回撤 + 择时 regime 因子已实现 RankIC + 市场 ER
  Reward 奖励     = 各挑战者相对冠军的滚动风险调整超额（夏普差）
  Q 信誉更新      = EWMA 在线更新（alpha 学习率），表格化 Q-learning
  Action 动作     = argmax Q 提名当期最优挑战者；连续达标才标 PROMOTE_CANDIDATE（仅建议）
  监管预警        = 择时因子 IC 反转 / 冠军退化 / 回撤异常 —— 只告警/提名，绝不擅自切换

复用不重写：Engine20（vendored v2.0 单引擎）、frozen_params、window_metrics。
零前视：所有 IC 配对仅用 d+h<=t 的已实现数据；滚动窗口只用截至 t 的数据。
"""
from __future__ import annotations

import hashlib
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from timing_data import load_config, load_close, dump_json, OUT_DATA  # noqa: E402
from timing_engine import (frozen_params, lock_params, main_engine,  # noqa: E402
                           run_full, run_window)
from aw_timing.research20 import metrics as r20_metrics  # noqa: E402

CFG = load_config()
REG = CFG["regulator"]
OUT = OUT_DATA


# ---------------------------------------------------------------- 挑战者（择时变体）
def build_challengers(champ: dict) -> dict:
    """影子挑战者 = 围绕冠军的择时参数变体（真实可回测，非编造）。

    每个变体只改择时 overlay，alpha 选股参数与冠军完全一致。
    """
    def v(name, **over):
        d = dict(champ)
        d.update(over)
        d["_name"] = name
        return d

    return {
        "冠军(no-timing)": v("冠军(no-timing)", timing_mode="none"),
        "低波管理-温和": v("低波管理-温和", timing_mode="vol_managed", timing_k=60,
                       timing_floor=0.5, timing_aggr=0.8, timing_gamma=1.0),
        "低波管理-激进": v("低波管理-激进", timing_mode="vol_managed", timing_k=60,
                       timing_floor=0.2, timing_aggr=1.5, timing_gamma=1.5),
        "趋势跟随": v("趋势跟随", timing_mode="tsmom", timing_k=90, timing_floor=0.2),
        "均线平滑": v("均线平滑", timing_mode="ma_smooth", timing_k=200,
                   timing_floor=0.5, timing_band=0.01),
        "多信号投票": v("多信号投票", timing_mode="multi_vote", timing_k=200,
                    timing_floor=0.5, vote_soft=0.0, vote_enter=0.65, vote_exit=0.35),
        "多信号-软仓": v("多信号-软仓", timing_mode="multi_vote", timing_k=200,
                     timing_floor=0.3, vote_soft=0.6, vote_enter=0.60, vote_exit=0.40),
    }


def run_all_navs(configs: dict) -> pd.DataFrame:
    """对每个配置独立回测，返回对齐日期的日频净值 DataFrame。"""
    navs = {}
    for name, cfg in configs.items():
        p = lock_params({k: val for k, val in cfg.items() if not k.startswith("_")})
        r = run_full(p)  # 单引擎，全样本
        eq = np.asarray(r["equity"], float)
        dts = pd.DatetimeIndex(r["dates"])
        navs[name] = pd.Series(eq, index=dts)
    return pd.DataFrame(navs)


# ---------------------------------------------------------------- 工具函数
def win_metrics(nav_win: pd.Series) -> dict:
    """滚动窗口风险调整指标（夏普年化、最大回撤、区间收益）——只用窗口内数据。"""
    r = nav_win.pct_change().dropna()
    if len(r) < 10:
        return dict(sharpe=np.nan, mdd=np.nan, ret=np.nan)
    sd = r.std()
    sharpe = r.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
    mdd = float((nav_win / nav_win.cummax() - 1).min())
    ret = float(nav_win.iloc[-1] / nav_win.iloc[0] - 1)
    return dict(sharpe=float(sharpe), mdd=mdd, ret=ret)


def regime_signals(close: pd.DataFrame) -> dict:
    """择时 regime 因子矩阵（市场级时序信号，用于已实现 RankIC 监管）。

    全部由 past data 构造，与主引擎同源：等权风险资产组合的动量/波动/ breadth。
    """
    eng = main_engine()
    risk = np.asarray(eng.risk, bool)
    ret1 = np.where(np.isfinite(eng.ret1[:, risk]), eng.ret1[:, risk], 0.0)
    idxret = ret1.mean(axis=1)                       # 等权风险资产日收益
    idx = pd.Series(np.cumprod(1.0 + idxret), index=eng.dates)
    out = {}
    # 时序动量信号（tanh，作为连续择时因子）
    for lb in REG["regime_lbs"]:
        out[f"tsmom_{lb}"] = pd.Series(np.tanh(idx / idx.shift(lb) - 1.0), index=eng.dates)
    # 波动状态（低波动 = risk-on）：1 - 已实现波动分位
    rv = pd.Series(idxret).rolling(REG["vol_win"], min_periods=REG["vol_win"]).std() * np.sqrt(252)
    pctl = rv.rolling(252, min_periods=120).rank(pct=True)
    out["lowvol_state"] = pd.Series(1.0 - pctl.to_numpy(), index=eng.dates)
    # 市场效率比 ER：净位移/路径长度（Kaufman，趋势干净度）
    dir_ = (idx - idx.shift(20)).abs()
    path = idx.diff().abs().rolling(20).sum()
    out["mkt_er"] = pd.Series((dir_ / path.replace(0, np.nan)).to_numpy(), index=eng.dates)
    return out


def realized_rankic(sig: pd.Series, fwd_port_ret: pd.Series, dates, t,
                    h=None, freq=None, win=None, max_back=None) -> tuple:
    """截至评估日 t 的择时因子「已实现时序 RankIC」（零前视）。

    择时 regime 因子是市场级标量信号，故做时序 IC：
    在滚动窗口内取 (信号@d, 未来h日组合收益@d) 配对序列，d+h<=t 保证收益已实现。
    返回 (IC均值, IC>0占比, 样本数)。IC 显著为负 → 择时因子反转预警。
    """
    h = h or REG["ic_h"]
    freq = freq or REG["ic_freq"]
    win = win or REG["reward_win"]
    pos = dates.get_loc(t)
    lo = max(0, pos - win)
    xs, ys = [], []
    for d in dates[lo:pos + 1:freq]:
        if dates.get_loc(d) + h > pos:       # 未来收益必须在评估日前已实现
            continue
        x = sig.loc[d]
        y = fwd_port_ret.loc[d] if d in fwd_port_ret.index else np.nan
        if np.isfinite(x) and pd.notna(y):
            xs.append(float(x))
            ys.append(float(y))
    if len(xs) < 8:
        return (np.nan, np.nan, 0)
    ic = pd.Series(xs).corr(pd.Series(ys), method="spearman")
    ic = float(ic) if pd.notna(ic) else np.nan
    # 分段 IC>0 占比（每 5 个配对算一次局部相关）
    pos_ratio = np.nan
    if len(xs) >= 20:
        locals_ = []
        for i in range(0, len(xs) - 4, 5):
            c = pd.Series(xs[i:i + 5]).corr(pd.Series(ys[i:i + 5]), method="spearman")
            if pd.notna(c):
                locals_.append(float(c))
        if locals_:
            pos_ratio = float(np.mean(np.array(locals_) > 0))
    return (ic, pos_ratio, len(xs))


def adaptive_ic_threshold(ic_history: list, t=None) -> float:
    """IC 反转自适应阈值：基于截至当前的已实现 IC 历史分布分位数（零前视）。

    固定阈值 -0.02 的问题：IC 标准差达 0.36，导致 36% 的时间都在预警（假警报）。
    自适应做法：阈值 = 历史 IC 的 ic_adapt_q 分位数，夹在 [floor, cap] 之间。
    这样只有"相对该因子自身常态显著转负"才预警，预警更有信息量。
    """
    if REG.get("ic_reversal_mode", "fixed") != "adaptive":
        return float(REG["ic_reversal_th"])
    if len(ic_history) < int(REG.get("ic_adapt_min", 60)):
        return float(REG["ic_reversal_th"])
    arr = np.asarray(ic_history, float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < int(REG.get("ic_adapt_min", 60)):
        return float(REG["ic_reversal_th"])
    q = float(np.quantile(arr, REG.get("ic_adapt_q", 0.10)))
    lo, hi = float(REG.get("ic_adapt_floor", -0.35)), float(REG.get("ic_adapt_cap", -0.02))
    return float(np.clip(q, lo, hi))


def risk_adjusted_winners(m_all: dict, champ_name: str) -> dict:
    """风险调整口径的"跑赢冠军"判定（机构版正确口径）。

    单一 CAGR 判定会抹掉"降仓提夏普/降回撤"的择时价值；机构语境下
    同时看 Sharpe 与 Calmar（CAGR/|MDD|），任一显著占优即为风险调整胜出。
    """
    ch = m_all[champ_name]
    ch_sh, ch_cal = ch["sharpe"], ch["cagr"] / max(abs(ch["maxdd"]), 1e-6)
    out = {}
    for n, m in m_all.items():
        if n == champ_name:
            continue
        sh = m["sharpe"]
        cal = m["cagr"] / max(abs(m["maxdd"]), 1e-6)
        out[n] = dict(
            sharpe_edge=sh - ch_sh, calmar_edge=cal - ch_cal,
            sharpe_win=bool(sh > ch_sh), calmar_win=bool(cal > ch_cal),
            mdd_improve=bool(m["maxdd"] > ch["maxdd"]),  # 回撤改善（更接近0）
            risk_adjusted_win=bool((sh > ch_sh) or (cal > ch_cal)),
        )
    return out


def shadow_switch_backtest(navs: pd.DataFrame, ledger: list, champ_name: str) -> dict:
    """闭环验证：真实模拟"按 Q 提名切换"的累计净值（机构版神经中枢的可验证证据）。

    每个评估日 t 的 leader 决定 [t, 下一评估日) 区间持有的品种，
    滚动拼接成影子组合净值，与冠军对比。零前视：t 日决策只用截至 t 的信息。
    """
    if not ledger:
        return {}
    champ = navs[champ_name].dropna()
    idx = navs.index
    rets = navs.pct_change()
    sw = pd.Series(0.0, index=idx)
    eval_dates = pd.to_datetime([r["date"] for r in ledger])
    leaders = [r["leader"] for r in ledger]
    for i, t in enumerate(eval_dates):
        end = eval_dates[i + 1] if i + 1 < len(eval_dates) else idx[-1] + pd.Timedelta(days=1)
        mask = (idx >= t) & (idx < end)
        sw.loc[mask] = rets.loc[mask, leaders[i]].fillna(0.0).values
    sw_nav = (1.0 + sw).cumprod()
    valid = sw_nav > 0
    sw_nav = sw_nav[valid]
    ch_nav = champ.reindex(sw_nav.index).dropna()
    sw_nav = sw_nav.reindex(ch_nav.index)
    sw_r = sw.reindex(ch_nav.index).fillna(0.0)
    ch_r = rets[champ_name].reindex(ch_nav.index).fillna(0.0)

    def metrics_of(nav, r):
        yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-6)
        cagr = nav.iloc[-1] ** (1 / yrs) - 1
        sd = r.std()
        sharpe = (r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
        mdd = float((nav / nav.cummax() - 1).min())
        calmar = cagr / max(abs(mdd), 1e-6)
        return dict(cagr=float(cagr), maxdd=float(mdd),
                    sharpe=float(sharpe), calmar=float(calmar))

    sw_m, ch_m = metrics_of(sw_nav, sw_r), metrics_of(ch_nav, ch_r)
    switch_ratio = float(np.mean([l != champ_name for l in leaders]))
    # 目标达成（风险调整口径）
    tgt_sh = float(CFG["targets"]["sharpe_min"])
    tgt_mdd = float(CFG["targets"]["mdd_max"])
    return dict(
        metrics=dict(shadow=sw_m, champion=ch_m),
        switch_ratio=switch_ratio,
        n_switches=int(sum(1 for a, b in zip(leaders, leaders[1:]) if a != b)),
        target_sharpe_met=bool(sw_m["sharpe"] >= tgt_sh),
        target_mdd_met=bool(abs(sw_m["maxdd"]) <= tgt_mdd),
        sharpe_edge=float(sw_m["sharpe"] - ch_m["sharpe"]),
        nav={k: [round(float(v), 6) for v in vv.to_numpy()]
             for k, vv in (("shadow", sw_nav), ("champion", ch_nav))},
        dates=[str(d.date()) for d in sw_nav.index],
    )


# ---------------------------------------------------------------- 监管主流程
def run_regulator() -> dict:
    champ = frozen_params()
    configs = build_challengers(champ)
    names = list(configs.keys())
    champ_name = "冠军(no-timing)"
    challengers = [n for n in names if n != champ_name]

    print("=== 影子挑战者独立回测 ===")
    navs = run_all_navs(configs)
    for n in names:
        print(f"  {n:<16} 末值 {navs[n].dropna().iloc[-1]:.4f}")

    # 硬校验：冠军净值 == 冻结 alpha 全样本回测（逐点一致）
    base_eq = run_full(champ)["equity"]
    champ_diff = float(np.max(np.abs(np.asarray(base_eq, float)
                                     - navs[champ_name].to_numpy(float))))
    print(f"\n[兼容校验] 冠军 vs 冻结 alpha 逐点最大差 {champ_diff:.2e}")
    assert champ_diff < 1e-9, "冠军净值与冻结 alpha 不一致，影子台账不成立"

    # 未来 h 日「等权风险资产组合」收益（用于择时因子时序 IC，仅配对 d+h<=t）
    close = load_close()
    eng = main_engine()
    risk_cols = [c for c in CFG["universe"]["RISK"] if c in close.columns]
    px = close[risk_cols]
    port_ret = px.pct_change().mean(axis=1)          # 等权组合日收益
    port_nav = (1.0 + port_ret.fillna(0.0)).cumprod()
    fwd_port = port_nav.shift(-REG["ic_h"]) / port_nav - 1.0
    fwd_port.index = pd.DatetimeIndex(eng.dates[:len(fwd_port)])
    sigs = regime_signals(close)
    core_sig = sigs[REG["core_signal"]]

    dates = pd.DatetimeIndex(eng.dates)
    Q = {n: 0.0 for n in challengers}
    streak = {n: 0 for n in challengers}
    cool = 0
    eval_positions = list(range(REG["warmup"], len(dates), REG["eval_freq"]))
    ledger, events, q_hist = [], [], []
    ic_history = []   # 已实现 IC 历史（零前视累积，用于自适应阈值）

    for pos in eval_positions:
        t = dates[pos]
        w0 = max(0, pos - REG["reward_win"])
        win_sl = slice(w0, pos + 1)

        # State：冠军滚动绩效 + 已实现 RankIC + 市场 ER
        cm = win_metrics(navs[champ_name].iloc[win_sl])
        ic_mean, ic_pos, ic_n = realized_rankic(core_sig, fwd_port, dates, t)
        er = core_sig.loc[t] if np.isfinite(core_sig.loc[t]) else np.nan
        # 自适应 IC 阈值：基于截至当前的已实现 IC 历史分布（先算阈值再纳入本期 IC）
        ic_th = adaptive_ic_threshold(ic_history, t)
        if pd.notna(ic_mean):
            ic_history.append(ic_mean)

        # Reward：各挑战者相对冠军的滚动夏普超额
        rewards = {}
        for n in challengers:
            m = win_metrics(navs[n].iloc[win_sl])
            rewards[n] = (m["sharpe"] - cm["sharpe"]) if (
                pd.notna(m["sharpe"]) and pd.notna(cm["sharpe"])) else 0.0

        # Q 更新：EWMA 表格化 Q-learning
        for n in challengers:
            Q[n] = (1 - REG["alpha"]) * Q[n] + REG["alpha"] * rewards[n]

        # Action：argmax Q 提名；HOLD(保持冠军)为默认动作，Q=0 参与竞争
        # ★治理修正：只有当挑战者信誉 Q 超过门槛（显著优于冠军）才提名切换，
        #   否则保持冠军不动。避免"全为负 Q 也硬选一个挑战者"的伪切换。
        best_ch = max(challengers, key=lambda k: Q[k])
        if Q[best_ch] > REG["promote_edge"]:
            leader = best_ch
        else:
            leader = champ_name   # HOLD：保持冻结 alpha
        for n in challengers:
            streak[n] = streak[n] + 1 if (n == leader and Q[n] > REG["promote_edge"]) else 0

        # 监管预警（多标签可叠加）
        alerts = []
        if pd.notna(ic_mean) and ic_mean < ic_th:
            alerts.append("因子IC反转")
        if pd.notna(cm["sharpe"]) and cm["sharpe"] < REG["champ_sharpe_th"]:
            alerts.append("冠军退化")
        if pd.notna(cm["mdd"]) and cm["mdd"] < REG["champ_mdd_th"]:
            alerts.append("回撤预警")

        promoted = None
        if cool > 0:
            cool -= 1
        if (leader in challengers and streak[leader] >= REG["promote_streak"]
                and Q[leader] > REG["promote_edge"]
                and "冠军退化" not in alerts and cool == 0):
            promoted = leader
            cool = REG["promote_cool"]
        if promoted:
            alerts.append("晋升候选:" + promoted)
            events.append(dict(date=str(t.date()), type="PROMOTE_CANDIDATE",
                               name=promoted,
                               detail=f"连续{streak[leader]}期信誉领先 Q={Q[leader]:.3f}"))
        for a in [x for x in alerts if x in ("因子IC反转", "冠军退化", "回撤预警")]:
            events.append(dict(date=str(t.date()), type=a,
                               detail=f"IC均值{ic_mean:.3f}(阈{ic_th:.3f}) 冠军夏普{cm['sharpe']:.2f} 回撤{cm['mdd']:.1%}"))

        row = dict(date=str(t.date()), window=f"{dates[w0].date()}~{t.date()}",
                   champ_sharpe=round(cm["sharpe"], 4) if pd.notna(cm["sharpe"]) else None,
                   champ_mdd=round(cm["mdd"], 4) if pd.notna(cm["mdd"]) else None,
                   champ_ret=round(cm["ret"], 4) if pd.notna(cm["ret"]) else None,
                   ic_mean=round(ic_mean, 4) if pd.notna(ic_mean) else None,
                   ic_th=round(ic_th, 4),
                   ic_pos=round(ic_pos, 3) if pd.notna(ic_pos) else None,
                   ic_n=ic_n, er=round(float(er), 4) if pd.notna(er) else None,
                   leader=leader,
                   leader_Q=round(Q[leader], 4) if leader in challengers else 0.0,
                   promoted=promoted or "", alerts="|".join(alerts) if alerts else "正常")
        for n in challengers:
            row["rew_" + n] = round(rewards[n], 4)
            row["Q_" + n] = round(Q[n], 4)
        ledger.append(row)
        q_hist.append(dict(date=str(t.date()), **{("Q_" + n): round(Q[n], 4) for n in challengers}))

    ldf, edf, qdf = pd.DataFrame(ledger), pd.DataFrame(events), pd.DataFrame(q_hist)
    ldf.to_csv(OUT / "影子台账.csv", index=False, encoding="utf-8-sig")
    edf.to_csv(OUT / "监管事件.csv", index=False, encoding="utf-8-sig")
    qdf.to_csv(OUT / "信誉Q演化.csv", index=False, encoding="utf-8-sig")
    navs.to_csv(OUT / "影子净值.csv", encoding="utf-8-sig")

    # 摘要
    m_all = {}
    for n in names:
        p = lock_params({k: v for k, v in configs[n].items() if not k.startswith("_")})
        m_all[n] = run_full(p)["metrics"]
    n_promote = int(sum(1 for e in events if e["type"] == "PROMOTE_CANDIDATE"))
    n_icrev = int(sum(1 for e in events if e["type"] == "因子IC反转"))
    n_decay = int(sum(1 for e in events if e["type"] == "冠军退化"))
    final_leader = ldf.iloc[-1]["leader"]
    champ_ann = m_all[champ_name]["cagr"]
    better_cagr = [n for n in challengers if m_all[n]["cagr"] > champ_ann]
    # ★优化：风险调整口径的胜者（Sharpe / Calmar 任一占优）
    ra = risk_adjusted_winners(m_all, champ_name)
    better_ra = [n for n, v in ra.items() if v["risk_adjusted_win"]]
    # ★优化：影子切换闭环验证
    sw = shadow_switch_backtest(navs, ledger, champ_name) if REG.get("shadow_switch", True) else {}

    summary = dict(
        engine={k: REG[k] for k in ("eval_freq", "warmup", "reward_win", "ic_h",
                                    "alpha", "promote_streak", "promote_edge", "promote_cool",
                                    "ic_reversal_mode", "ic_adapt_q", "win_criterion")},
        paradigm="sr-nav/evolution_guard Q-learning 监管层（移植到择时因子）",
        champion_diff=champ_diff,
        n_eval=len(ledger), n_events=len(events),
        n_promote=n_promote, n_icrev=n_icrev, n_decay=n_decay,
        icrev_rate=round(n_icrev / max(len(ledger), 1), 3),
        final_leader=final_leader,
        final_Q={n: round(Q[n], 4) for n in challengers},
        range=[str(dates[0].date()), str(dates[-1].date())],
        eval_first=ledger[0]["date"], eval_last=ledger[-1]["date"],
        metrics={n: {k: m_all[n][k] for k in ("cagr", "maxdd", "sharpe", "avg_exp")} for n in names},
        better_than_champion=better_ra,          # 风险调整口径（默认）
        better_by_cagr_only=better_cagr,         # 保留原口径供对照
        risk_adjusted_detail=ra,
        shadow_switch=({k: sw[k] for k in ("metrics", "switch_ratio", "n_switches",
                                           "target_sharpe_met", "target_mdd_met", "sharpe_edge")}
                       if sw else {}),
        champion_ann=champ_ann,
    )
    dump_json(summary, OUT / "evolution_summary.json")
    # 看板接口
    report = dict(
        summary=summary,
        dates=[str(d.date()) for d in dates[REG["warmup"]:]],
        nav={n: [round(float(v), 5) for v in navs[n].dropna().to_numpy()]
             for n in names},
        shadow_nav=({k: sw[k] for k in ("nav", "dates")} if sw else {}),
        ledger=ledger, events=events, q_hist=q_hist,
        challengers=challengers, names=names,
    )
    dump_json(report, OUT / "evolution_report.json")

    print("\n" + "=" * 88)
    print(f"评估 {len(ledger)} 期（{ledger[0]['date']}~{ledger[-1]['date']}）| "
          f"晋升候选 {n_promote} 次 | IC反转 {n_icrev} 次({n_icrev/max(len(ledger),1):.1%}) | "
          f"冠军退化 {n_decay} 次 | 末期领先 {final_leader}")
    print(f"收益口径跑赢冠军 {len(better_cagr)} 个；风险调整口径跑赢冠军 {len(better_ra)} 个: "
          + ("/".join(better_ra) if better_ra else "无"))
    if sw:
        sm, cm = sw["metrics"]["shadow"], sw["metrics"]["champion"]
        print(f"[影子切换闭环] 按Q提名切换: CAGR {sm['cagr']*100:.2f}% MDD {sm['maxdd']*100:.2f}% "
              f"Sharpe {sm['sharpe']:.4f} | 冠军 Sharpe {cm['sharpe']:.4f} "
              f"Δ {sw['sharpe_edge']:+.4f} | 目标夏普{'达标' if sw['target_sharpe_met'] else '未达'}")
    print("[结论] 监管层仅记录/预警/提名，未切换冻结 alpha；冠军净值逐点一致。")
    return summary


if __name__ == "__main__":
    run_regulator()
