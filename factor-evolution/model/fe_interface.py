"""
factor-evolution · v1.4 看板数据接口生成器
==========================================
把 v1.4 (mom250 零杠杆单因子) 的回放结果，写成与 rotation_core/model_data.py
完全兼容的 nav.json / signal.json / meta.json / nav_history.csv / rebalances.csv，
供主看板 index.html 直接消费。

设计原则：
  - 唯一数据源：factor-evolution/data/etf_ohlc.csv（与私有快照 close 字节级一致）
  - 无前视：t-1 决策、t 生效；fe_engine 已保证
  - 对账锚定报告基准日：基准日固定，之后新增行情只延伸曲线不改写历史结论
  - 与冻结 v2.1 完全隔离

用法：
  python fe_interface.py                 # 用 factor-evolution/data 生成，写到 ../data_out
  python fe_interface.py --out <dir>     # 指定输出目录（如私有仓库 docs/data 或公开 data）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import fe_factors as FF          # noqa: E402
import fe_engine as FE           # noqa: E402

DATA = os.path.join(ROOT, "data")
CFG_PATH = os.path.join(ROOT, "out", "v1.4_config.json")

MODEL_NAME = "v1.4.2 · mom250 零杠杆单因子 + 走熊过滤"
MODEL_CODE = "v1.4.2_mom250_zero_lev_bear_filter"
BASIS_START = "2014-02-10"       # 与页面历史口径一致：起点净值归一化日
REPORT_END = "2026-11-18"        # 对账基准日（数据快照末日），固定不随行情漂移
TRADING_DAYS = 252
CASH_ANNUAL = 0.0                # v1.4 以国债 ETF 补位，不计额外现金利息

CODES = {"CSI300ETF": "510300", "NDX100ETF": "513100", "GoldETF": "518880",
         "ChiNextETF": "159915", "STAR50ETF": "588000", "BondETF": "511010"}
ASSET_CN = {"CSI300ETF": "沪深300ETF", "NDX100ETF": "纳指100ETF", "GoldETF": "黄金ETF",
            "ChiNextETF": "创业板ETF", "STAR50ETF": "科创50ETF", "BondETF": "国债ETF"}
ROLE = {"BondETF": "防御", "GoldETF": "避险", "NDX100ETF": "跨境风险",
        "CSI300ETF": "A 股宽基", "ChiNextETF": "成长", "STAR50ETF": "成长"}


def _r(x, n=6):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def _pct(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x) * 100.0, n)


def params_fingerprint(params: dict) -> str:
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:10]


def _json_default(o):
    # numpy 标量/NaN 兜底，避免 np.bool_/np.float64 中断序列化
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"), sort_keys=False, default=_json_default)
        f.write("\n")
    os.replace(tmp, path)


def _write_csv(path, header, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def compute_vol(c, r, vol_win):
    T, A = c.shape
    vol = np.full((T, A), np.nan)
    for j in range(A):
        vol[:, j] = FE.pd_roll_std(r[:, j], vol_win) * np.sqrt(TRADING_DAYS)
    return vol


def build():
    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    p = cfg["params"]
    panel, dates = FF.load(DATA)
    F = FF.build_factors(panel)
    c, r = panel.close, panel.ret
    T, A = c.shape
    warm = int(p["warmup"])
    rebal = int(p["rebal"])
    vol_win = int(p["vol_win"])
    vol_cap = p["vol_cap"]
    vol_target = p["vol_target"]
    top_n = int(p["top_n"])
    dd_limit = p["dd_limit"]
    dd_recover = p.get("dd_recover", 0.95)
    dd_cooldown = int(p.get("dd_cooldown", 60))
    cost = p.get("cost_bps", 10.0) / 1e4

    vol = compute_vol(c, r, vol_win)
    fw = p["factors"]
    Fz = {n: FE.zscore_cross(F[n]) for n in fw if n in F and abs(fw[n]) > 1e-12}

    # v1.4.2 double 走熊过滤（与 fe_engine.run 同口径）
    use_bear = bool(p.get('bear_filter', False)) and ('mom250' in F)
    bear = np.zeros((T, A), dtype=bool)
    if use_bear:
        ma_n = int(p.get('bear_ma', 200))
        ma = np.full((T, A), np.nan)
        for j in range(A):
            for t in range(T):
                seg = c[max(0, t - ma_n + 1):t + 1, j]
                seg = seg[np.isfinite(seg)]
                if len(seg) >= ma_n * 0.8:
                    ma[t, j] = seg.mean()
        bear = (c < ma) & (F['mom250'] < 0)

    # ---- 逐日回放，记录持仓/仓位（与 fe_engine.run 同逻辑，附加记录） ----
    equity = 1.0
    peak = 1.0
    eq_curve = np.full(T, np.nan)
    exp_curve = np.zeros(T)
    w_curve = np.zeros((T, A))
    cur_w = np.zeros(A)
    cur_exp = 0.0
    turn = 0.0
    trades = 0
    circuit = False; cd = 0
    rebal_log = []

    def select(t, score=None):
        if score is None:
            score = np.zeros(A)
            for name, wgt in fw.items():
                if name in Fz:
                    score += wgt * np.nan_to_num(Fz[name][t], nan=0.0)
        elig = np.zeros(A, dtype=bool)
        for j in range(5):
            elig[j] = (np.isfinite(vol[t, j]) and vol[t, j] <= vol_cap
                       and np.isfinite(c[t, j]) and not (use_bear and bear[t, j]))
        n_ok = int(elig.sum())
        if n_ok == 0:
            nw = np.zeros(A); nw[5] = 1.0
            return nw, 0.0, [], score
        cand = np.where(elig)[0]
        k = min(top_n, n_ok)
        order = cand[np.argsort(-score[cand])][:k]
        nw = np.zeros(A)
        iv = 1.0 / np.maximum(vol[t, order], 1e-6)
        iv = iv / iv.sum()
        for idx, j in enumerate(order):
            nw[j] = iv[idx]
        vsel = np.nan_to_num(vol[t, order], nan=0.0)
        port_vol = float(np.sqrt(np.sum((iv * vsel) ** 2) + 2e-6))
        raw_exp = np.clip(vol_target / (port_vol + 1e-9), 0.0, 1.0)
        new_exp = np.clip(raw_exp, 0.0, 1.0)
        nw[5] = 1.0 - new_exp
        return nw, new_exp, [int(j) for j in order], score

    for t in range(warm, T):
        day_ret = 0.0
        for j in range(A):
            if np.isfinite(r[t, j]) and np.isfinite(cur_w[j]) and cur_w[j] != 0:
                day_ret += cur_w[j] * r[t, j]
        equity *= (1.0 + day_ret)
        eq_curve[t] = equity
        exp_curve[t] = cur_exp
        w_curve[t] = cur_w
        peak = max(peak, equity)
        dd = equity / peak - 1.0

        new_w = None; new_exp = None; picked = None; score = None; acted = False
        # 每日熔断检查 (t 日收盘 → t+1 生效)
        if (not circuit) and dd <= -dd_limit:
            circuit = True; cd = dd_cooldown
            new_w = np.zeros(A); new_w[5] = 1.0; new_exp = 0.0; picked = []; acted = True
        if circuit and not acted:
            cd -= 1
            recovered = dd >= -dd_limit * dd_recover
            if cd <= 0 and recovered and (t - warm) % rebal == 0 and t < T - 1:
                circuit = False
                new_w, new_exp, picked, score = select(t); acted = True
        if not acted and (t - warm) % rebal == 0 and t < T - 1 and (not circuit):
            new_w, new_exp, picked, score = select(t); acted = True

        if acted and new_w is not None:
            delta = np.abs(new_w - cur_w).sum() + abs(new_exp - cur_exp)
            turn += delta
            equity *= (1.0 - cost * delta)
            trades += int(np.sum(np.abs(new_w[:5] - cur_w[:5]) > 1e-6))
            cur_w, cur_exp = new_w, new_exp
            hold_names = [FF.ALL[j] for j in (picked or [])]
            rebal_log.append(dict(
                date=str(dates[t])[:10], exp=float(new_exp), picked=picked or [],
                names=hold_names, score=float(max(score[i] for i in picked)) if picked else None,
                circuit=bool(circuit and not picked),
            ))

    # ---- 报告口径序列：从 BASIS_START 起，起点归一化为 1 ----
    date_strs = [str(d)[:10] for d in dates]
    all_dates = date_strs[warm:T]
    eq_raw = eq_curve[warm:T]
    i0 = all_dates.index(BASIS_START) if BASIS_START in all_dates else 0
    dser = all_dates[i0:]
    base = eq_raw[i0]
    nav = [float(x / base) for x in eq_raw[i0:]]
    pos = [float(exp_curve[warm + k]) for k in range(i0, T - warm)]
    peak, ddser = nav[0], []
    for v in nav:
        peak = max(peak, v)
        ddser.append((v / peak - 1.0) * 100.0)

    # ---- 绩效指标（fe_engine 同口径） ----
    eq = np.asarray(nav)
    yrs = len(eq) / TRADING_DAYS
    cagr = (eq[-1] / eq[0]) ** (1.0 / yrs) - 1.0 if yrs > 0 else 0.0
    pk = np.maximum.accumulate(eq)
    mdd = float(np.min(eq / pk - 1.0))
    dret = np.diff(eq) / eq[:-1]
    sharpe = float(np.mean(dret) / (np.std(dret) + 1e-9) * np.sqrt(TRADING_DAYS))
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    vol_ann = float(np.std(dret) * np.sqrt(TRADING_DAYS))

    def window_ret(since):
        idx = [i for i, d in enumerate(dser) if d >= since]
        return (nav[-1] / nav[idx[0]] - 1.0) * 100.0 if idx else None

    y1 = window_ret(f"{int(dser[-1][:4]) - 1}-{dser[-1][5:]}")
    ytd = window_ret(f"{dser[-1][:4]}-01-01")

    # 月胜率：每月最后净值 vs 上月最后净值
    from collections import OrderedDict
    month_end = OrderedDict()
    for d, v in zip(dser, nav):
        month_end[d[:7]] = v
    mvals = list(month_end.values())
    mrets = [mvals[i] / mvals[i - 1] - 1.0 for i in range(1, len(mvals))]
    month_win = (sum(1 for x in mrets if x > 0) / len(mrets) * 100.0) if mrets else 0.0

    perf = dict(
        total_return=_pct(eq[-1] / eq[0] - 1.0, 2),
        cagr=_pct(cagr, 4),
        max_drawdown=_pct(mdd, 4),
        calmar=round(calmar, 2),
        sharp=round(sharpe, 4),
        month_win=round(month_win, 4),
        vol_ann=_pct(vol_ann, 2),
        ret_1y=_pct(y1 / 100.0, 2) if y1 is not None else None,
        ret_ytd=_pct(ytd / 100.0, 2) if ytd is not None else None,
        final_nav=_r(eq[-1], 4),
        bars=len(dser),
        trades=trades,
        turnover=_r(turn, 2),
    )

    # ---- 分年统计 ----
    yearly = []
    years = sorted({d[:4] for d in dser})
    for y in years:
        idx = [i for i, d in enumerate(dser) if d[:4] == y]
        s0 = idx[0] - 1 if idx[0] > 0 else idx[0]
        seg = nav[s0:idx[-1] + 1]
        if len(seg) < 2:
            continue
        total = seg[-1] / seg[0] - 1.0
        rets = [seg[k + 1] / seg[k] - 1.0 for k in range(len(seg) - 1)]
        n = len(rets)
        ann = (1.0 + total) ** (TRADING_DAYS / n) - 1.0 if total > -1 else -1.0
        pk, md = seg[0], 0.0
        for v in seg:
            pk = max(pk, v)
            md = max(md, 1.0 - v / pk)
        mean = sum(rets) / n
        var = sum((x - mean) ** 2 for x in rets) / (n - 1) if n > 1 else 0.0
        std = math.sqrt(var) * math.sqrt(TRADING_DAYS)
        yearly.append(dict(
            year=int(y), total_return_pct=_pct(total, 2), annual_return_pct=_pct(ann, 2),
            max_drawdown_pct=round(md * 100.0, 2),
            return_dd_ratio=round(total / md, 2) if md > 1e-9 else None,
            std_pct=_pct(std, 2),
            sharpe=round((mean * TRADING_DAYS - CASH_ANNUAL) / std, 2) if std > 1e-9 else None,
        ))

    # ---- 最新信号（取最后一个决策日） ----
    last = rebal_log[-1]
    t_last = date_strs.index(last["date"])
    last_exp = last["exp"]
    in_circuit = bool(last.get("circuit")) and not last["picked"]
    # 当前持仓：最后决策日选中，权重按波动率倒数
    cur_items = []
    for j in last["picked"]:
        w = w_curve[t_last + 1] if t_last + 1 < T else w_curve[t_last]
        cur_items.append(dict(
            code=CODES[FF.ALL[j]], name=ASSET_CN[FF.ALL[j]], asset=FF.ALL[j],
            role=ROLE.get(FF.ALL[j], "风险"), weight=round(float(w[j] * 100.0), 2),
            capital=round(float(w[j] * 100.0), 2),
            price=_r(c[min(t_last, T - 1), j], 3),
        ))
    has_risk = bool(cur_items)
    if in_circuit or not cur_items:
        cur_items.append(dict(
            code=CODES["BondETF"], name=ASSET_CN["BondETF"], asset="BondETF",
            role="防御", weight=100.0, capital=0.0,
            price=_r(c[min(t_last, T - 1), 5], 3),
        ))
    # 下一调仓日：最后决策日 + rebal 个交易日；超出快照则用工作日外推
    from datetime import date, timedelta
    nxt_idx = t_last + rebal
    if nxt_idx < T:
        next_rebalance = date_strs[nxt_idx]
    else:
        cur = date.fromisoformat(date_strs[-1])
        remaining = nxt_idx - (T - 1)
        while remaining > 0:
            cur += timedelta(days=1)
            if cur.weekday() < 5:
                remaining -= 1
        next_rebalance = cur.isoformat()
    days_to_rebal = max(0, nxt_idx - (T - 1))
    # 因子面板（最后一决策日）
    factor_rows = []
    for j, a in enumerate(FF.ALL):
        raw = F["mom250"][t_last, j] if "mom250" in F else np.nan
        z = Fz["mom250"][t_last, j] if "mom250" in Fz else np.nan
        is_bear = bool(use_bear and j < 5 and bear[t_last, j])
        eligible = bool(np.isfinite(vol[t_last, j]) and vol[t_last, j] <= vol_cap
                        and np.isfinite(c[t_last, j]) and not is_bear) and j < 5
        why = ("合格" if eligible else "避险位" if j == 5 else
               "走熊过滤" if is_bear else
               "波动超限" if np.isfinite(vol[t_last, j]) else "无效价")
        factor_rows.append(dict(
            code=CODES[a], name=ASSET_CN[a], asset=a,
            mom=_pct(raw, 2), vol=_pct(vol[t_last, j], 2),
            score=_r(z, 3), eligible=eligible, why=why,
        ))

    sig = dict(
        model=MODEL_NAME, model_code=MODEL_CODE, params_fp=params_fingerprint(p),
        as_of=date_strs[-1], decision_date=last["date"],
        next_rebalance=next_rebalance, days_to_rebalance=days_to_rebal,
        current=cur_items, target=cur_items,
        current_exposure=round(last_exp * 100.0, 2),
        target_exposure=round(last_exp * 100.0, 2),
        will_change=False, changed=[], circuit=in_circuit,
        reason=("回撤触及熔断线 → 次日全退国债防御，收敛后逢调仓日恢复"
                if in_circuit else
                ("全部风险标的走熊/波动超限 → 空仓退国债防御" if not has_risk else
                 "mom250 横截面动量 · 波动率倒数定仓 · 零杠杆 · 走熊过滤")),
        drawdown=round(ddser[-1], 2), dd_limit=_pct(dd_limit, 2),
        pv=None, factors=factor_rows,
    )

    rebalances = []
    prev_names = None
    for e in rebal_log:
        names = e["names"]
        is_circ = bool(e.get("circuit")) and not names
        kind = "熔断" if is_circ else ("换仓" if (prev_names is None or set(names) != set(prev_names)) else "权重微调")
        rebalances.append(dict(
            date=e["date"], exposure=round(e["exp"] * 100.0, 2), names=" / ".join(ASSET_CN[a] for a in names) if names else "国债ETF",
            changed=(kind in ("换仓", "熔断")), kind=kind,
            reason=("回撤熔断 → 全退国债" if is_circ else
                    (f"mom250 得分最高 {ASSET_CN[names[0]]}" if names else "无合格标的 → 国债ETF")),
            circuit=is_circ,
        ))
        prev_names = names

    # ---- 对账：锚定 REPORT_END ----
    cut = dser.index(REPORT_END) if REPORT_END in dser else len(dser) - 1
    eq_cut = np.asarray(nav[:cut + 1])
    yrs_c = len(eq_cut) / TRADING_DAYS
    cagr_c = (eq_cut[-1] / eq_cut[0]) ** (1.0 / yrs_c) - 1.0
    pk_c = np.maximum.accumulate(eq_cut)
    mdd_c = float(np.min(eq_cut / pk_c - 1.0))
    anchor = cfg["metrics"].get("anchor_1118", cfg["metrics"]["full"])
    target = dict(cagr=anchor["cagr"], maxdd=abs(anchor["maxdd"]))
    actual = dict(cagr=round(cagr_c * 100.0, 4), maxdd=round(abs(mdd_c) * 100.0, 4))
    checks = dict(cagr=round(abs(actual["cagr"] - target["cagr"]), 4),
                  maxdd=round(abs(actual["maxdd"] - target["maxdd"]), 4))

    nav_obj = dict(
        model=MODEL_NAME, model_code=MODEL_CODE, params_fp=params_fingerprint(p),
        basis=f"报告口径 {BASIS_START} 起，起点净值归一化为 1",
        source="factor-evolution fe_engine · mom250 零杠杆单因子回放",
        as_of=dser[-1], start=dser[0], bars=len(dser),
        dates=dser, nav=[_r(x) for x in nav],
        ret=[_r((x - 1.0) * 100.0, 4) for x in nav],
        drawdown=[_r(x, 4) for x in ddser],
        position=[_r(x, 6) for x in pos],
        metrics=perf, yearly=yearly, signal=sig,
        scale=dict(rebalances=len(rebalances), first=dser[0], last=dser[-1]),
        checks=checks,
    )
    nav_obj["signal"]["rebalances"] = rebalances[-12:]

    meta = dict(
        model=MODEL_NAME, model_code=MODEL_CODE, params_fp=params_fingerprint(p),
        source="local_model_replay_v14", basis=nav_obj["basis"],
        snapshot_bars=T, snapshot_from=date_strs[0], snapshot_to=date_strs[-1],
        as_of=dser[-1], bars=len(dser),
        reconcile=dict(
            target=target, window=REPORT_END, as_of=dser[-1],
            extended_bars=len(dser) - cut - 1,
            actual=actual,
            display=dict(cagr=perf["cagr"], maxdd=perf["max_drawdown"], month_win=perf["month_win"]),
            max_abs_diff_pp=max(checks.values()), ok=max(checks.values()) <= 0.01,
        ),
    )

    return dict(
        nav=nav_obj, signal=sig, meta=meta, rebalances=rebalances,
        series=dict(dates=dser, nav=nav, dd=ddser, pos=pos),
        checks=checks, perf=perf, target=target, actual=actual,
    )


def write_all(b: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "nav.json"), b["nav"])
    _write_json(os.path.join(out_dir, "signal.json"), b["signal"])
    _write_json(os.path.join(out_dir, "meta.json"), b["meta"])
    s = b["series"]
    _write_csv(os.path.join(out_dir, "nav_history.csv"),
               ["day", "nav", "ret", "drawdown", "position"],
               [[d, f"{n:.6f}", f"{(n - 1) * 100:.4f}", f"{dd:.4f}", f"{p:.6f}"]
                for d, n, dd, p in zip(s["dates"], s["nav"], s["dd"], s["pos"])])
    _write_csv(os.path.join(out_dir, "rebalances.csv"),
               ["date", "exposure_pct", "holdings", "reason"],
               [[r["date"], f"{r['exposure']:.2f}", r["names"], r["reason"]] for r in b["rebalances"]])


def main():
    ap = argparse.ArgumentParser(description="v1.4 看板数据接口生成器")
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    b = build()
    print(f"[v1.4] 对账 target={b['target']} actual={b['actual']} diff={b['checks']}")
    print(f"[v1.4] 绩效 {b['perf']}")
    if not b["meta"]["reconcile"]["ok"]:
        raise SystemExit("[FAIL] v1.4 对账超差，拒绝产出")
    if args.check:
        return 0
    write_all(b, args.out)
    print(f"[ok] 已写出 5 个接口文件 → {args.out}")
    print(f"[ok] as_of={b['nav']['as_of']} bars={b['nav']['bars']} 最新持仓={[c['name'] for c in b['signal']['current']]}")


if __name__ == "__main__":
    main()
