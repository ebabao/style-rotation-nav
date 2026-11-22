# -*- coding: utf-8 -*-
"""vol_win × vol_cap 参数扫描：找出压回撤不降收益的更优入池过滤。
复用线上 v14_engine 逻辑（mom250/top_n=1/零杠杆/每日熔断18%-0.95-60/10bp），
仅把 vol_win、vol_cap 参数化。数据源=私有生产快照（含 11-19/11-20）。
"""
import sys, csv
import numpy as np
sys.path.insert(0, '/workspace/style-rotation-alert/nav')
import v14_engine as V

rows = list(csv.reader(open('/workspace/style-rotation-alert/data/etf_close_snapshot.csv', encoding='utf-8')))
dates = [str(r[0])[:10] for r in rows[1:]]
close = np.array([[float(r[i+1]) if r[i+1] else np.nan for i in range(6)] for r in rows[1:]])

TRADING_DAYS = 252
BOND = 5
RISK_N = V.RISK_N


def build_vol(ret, win):
    c = ret.shape[0]
    vol = np.full_like(ret, np.nan)
    for j in range(ret.shape[1]):
        for t in range(c):
            seg = ret[max(0, t - win + 1):t + 1, j]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= max(3, win // 2):
                vol[t, j] = np.std(seg, ddof=1) * np.sqrt(TRADING_DAYS)
    return vol


def replay(ret, mom, vol, vol_cap, dd_limit=0.18, dd_recover=0.95, dd_cooldown=60,
           rebal=10, vol_target=0.40, top_n=1, cost_bps=10, warmup=260):
    T, A = ret.shape
    mz = V.zscore_cross(mom)
    cost = cost_bps / 1e4
    equity = 1.0; peak = 1.0
    eqc = np.full(T, np.nan)
    cur_w = np.zeros(A); cur_exp = 0.0
    circuit = False; cd = 0
    cur_picked = []

    def select(t):
        score = np.nan_to_num(mz[t], nan=0.0)
        elig = np.zeros(A, dtype=bool)
        for j in range(RISK_N):
            elig[j] = np.isfinite(vol[t, j]) and vol[t, j] <= vol_cap and np.isfinite(close[t, j])
        if elig.sum() == 0:
            nw = np.zeros(A); nw[BOND] = 1.0
            return nw, 0.0, []
        cand = np.where(elig)[0]
        k = min(top_n, len(cand))
        order = cand[np.argsort(-score[cand])][:k]
        nw = np.zeros(A)
        iv = 1.0 / np.maximum(vol[t, order], 1e-6); iv /= iv.sum()
        for idx, j in enumerate(order):
            nw[j] = iv[idx]
        vsel = np.nan_to_num(vol[t, order], nan=0.0)
        pv = float(np.sqrt(np.sum((iv * vsel) ** 2) + 2e-6))
        raw = np.clip(vol_target / (pv + 1e-9), 0.0, 1.0)
        nw[BOND] = 1.0 - raw
        return nw, raw, [int(j) for j in order]

    for t in range(warmup, T):
        day = 0.0
        for j in range(A):
            if np.isfinite(ret[t, j]) and np.isfinite(cur_w[j]) and cur_w[j] != 0:
                day += cur_w[j] * ret[t, j]
        equity *= (1 + day)
        eqc[t] = equity
        peak = max(peak, equity)
        dd = equity / peak - 1.0
        nw = None; ne = None
        if (not circuit) and dd <= -dd_limit:
            circuit = True; cd = dd_cooldown
            nw = np.zeros(A); nw[BOND] = 1.0; ne = 0.0
        if circuit:
            cd -= 1
            ok = dd >= -dd_limit * dd_recover
            if cd <= 0 and ok and (t - warmup) % rebal == 0:
                circuit = False; nw, ne, cur_picked = select(t)
        if nw is None and (t - warmup) % rebal == 0 and t < T - 1 and (not circuit):
            nw, ne, cur_picked = select(t)
        if nw is not None:
            delta = np.abs(nw - cur_w).sum() + abs(ne - cur_exp)
            equity *= (1 - cost * delta)
            cur_w, cur_exp = nw, ne
    return eqc


def metrics(eqc, warmup=260):
    eq = eqc[warmup:]; eq = eq[np.isfinite(eq)]
    yrs = len(eq) / 252
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1
    pk = np.maximum.accumulate(eq); mdd = np.min(eq / pk - 1)
    dr = np.diff(eq) / eq[:-1]; shp = np.mean(dr) / (np.std(dr) + 1e-9) * np.sqrt(252)
    return cagr * 100, mdd * 100, shp, cagr / abs(mdd) if mdd < 0 else 0


c = close
ret = np.full_like(c, np.nan)
ret[1:] = c[1:] / np.where(c[:-1] != 0, c[:-1], np.nan) - 1.0
mom = np.full_like(c, np.nan)
mom[250:] = c[250:] / np.where(c[:-250] != 0, c[:-250], np.nan) - 1.0

# 基线（线上当前）
vol20 = build_vol(ret, 20)
base = metrics(replay(ret, mom, vol20, 0.25))
print(f"线上基线 (win=20 cap=0.25): 年化={base[0]:.2f}% 回撤={base[1]:.2f}% 夏普={base[2]:.2f} 卡玛={base[3]:.2f}")
print()
print(f"{'win':>4}{'cap':>7}{'年化':>9}{'回撤':>10}{'夏普':>8}{'卡玛':>8}{'Δ回撤':>9}{'Δ卡玛':>8}")
print('-' * 62)
res = []
for win in [10, 15, 20, 30, 40, 60]:
    vol = build_vol(ret, win)
    for cap in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 9.99]:
        m = metrics(replay(ret, mom, vol, cap))
        res.append((win, cap, m))
        mark = ''
        if m[1] > base[1] + 0.3 and m[0] > base[0] - 1.0:
            mark = ' ★'
        print(f"{win:>4}{cap:>7.2f}{m[0]:>8.2f}%{m[1]:>9.2f}%{m[2]:>8.2f}{m[3]:>8.2f}"
              f"{m[1]-base[1]:>+8.2f}{m[3]-base[3]:>+8.2f}{mark}")
