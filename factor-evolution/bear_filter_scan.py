# -*- coding: utf-8 -*-
"""熊市预判过滤因子 · 研究线本地回测（不碰线上 / 不碰冻结 v2.1）
================================================================
问题：mom250 太滞后，黄金 11-09 被满仓选中时 mom250=-2.6%(五个里最不差)
      但 mom120=-14.5%、px<MA200 -10.6%、距60日高点-12.4% 已明确走熊。
思路：在 select(t) 入池时叠加"熊市过滤"——中长周期已转熊的标的直接剔除，
      与 vol_cap 并列，只在调仓日生效（非日频止损，避免重蹈标的级止损覆辙）。
口径：复用线上 v14_engine（mom250/top_n=1/零杠杆/每日熔断18%-0.95-60/10bp），
      无前视（因子只用 t 及之前数据），parity=0。
"""
import sys, csv
import numpy as np
sys.path.insert(0, '/workspace/style-rotation-alert/nav')
import v14_engine as V

rows = list(csv.reader(open('/workspace/style-rotation-alert/data/etf_close_snapshot.csv', encoding='utf-8')))
dates = [r[0] for r in rows[1:]]
close = np.array([[float(r[i]) if r[i] else np.nan for i in range(1, 7)] for r in rows[1:]])
NAMES = ['CSI300', 'NDX100', 'Gold', 'ChiNext', 'STAR50', 'Bond']
BOND = 5
RISK_N = V.RISK_N
TD = 252

ret = np.full_like(close, np.nan)
ret[1:] = close[1:] / np.where(close[:-1] != 0, close[:-1], np.nan) - 1.0
mom = np.full_like(close, np.nan)
mom[250:] = close[250:] / np.where(close[:-250] != 0, close[:-250], np.nan) - 1.0
vol20 = np.full_like(close, np.nan)
for j in range(close.shape[1]):
    for t in range(close.shape[0]):
        seg = ret[max(0, t - 19):t + 1, j]; seg = seg[np.isfinite(seg)]
        if len(seg) >= 10:
            vol20[t, j] = np.std(seg, ddof=1) * np.sqrt(TD)


def _ma(j, t, n):
    seg = close[max(0, t - n + 1):t + 1, j]; seg = seg[np.isfinite(seg)]
    return seg.mean() if len(seg) >= n * 0.8 else np.nan


def build_bear(kind):
    """返回 (T,A) bool：True=该标的在该时点已走熊，应从候选池剔除。"""
    T, A = close.shape
    bear = np.zeros((T, A), dtype=bool)
    for j in range(RISK_N):
        for t in range(T):
            px = close[t, j]
            if not np.isfinite(px):
                continue
            m20, m60, m120, m200 = _ma(j, t, 20), _ma(j, t, 60), _ma(j, t, 120), _ma(j, t, 200)
            mo60 = px / close[t - 60, j] - 1 if t >= 60 and np.isfinite(close[t - 60, j]) else np.nan
            mo120 = px / close[t - 120, j] - 1 if t >= 120 and np.isfinite(close[t - 120, j]) else np.nan
            hi60 = np.nanmax(close[max(0, t - 59):t + 1, j])
            dd60 = px / hi60 - 1 if np.isfinite(hi60) else np.nan
            hi120 = np.nanmax(close[max(0, t - 119):t + 1, j])
            dd120 = px / hi120 - 1 if np.isfinite(hi120) else np.nan
            bear[t, j] = {
                'ma200':   np.isfinite(m200) and px < m200,
                'ma120':   np.isfinite(m120) and px < m120,
                'ma60':    np.isfinite(m60) and px < m60,
                'mom120_0': np.isfinite(mo120) and mo120 < 0,
                'mom60_0':  np.isfinite(mo60) and mo60 < 0,
                'dd60_10': np.isfinite(dd60) and dd60 < -0.10,
                'dd60_15': np.isfinite(dd60) and dd60 < -0.15,
                'dd120_20': np.isfinite(dd120) and dd120 < -0.20,
                'align':   np.isfinite(m20) and np.isfinite(m60) and np.isfinite(m120) and m20 < m60 < m120,
                'ma200_and_mom120': (np.isfinite(m200) and px < m200) and (np.isfinite(mo120) and mo120 < 0),
            }[kind]
    return bear


def replay(bear, vol_cap=0.25, dd_limit=0.18, dd_recover=0.95, dd_cooldown=60,
           rebal=10, vol_target=0.40, top_n=1, cost_bps=10, warmup=260):
    T, A = ret.shape
    mz = V.zscore_cross(mom)
    cost = cost_bps / 1e4
    equity = 1.0; peak = 1.0
    eqc = np.full(T, np.nan)
    cur_w = np.zeros(A); cur_exp = 0.0
    circuit = False; cd = 0
    cur_picked = []
    filter_cnt = 0  # 统计被熊市过滤剔除的次数（调仓日）

    def select(t):
        nonlocal filter_cnt
        score = np.nan_to_num(mz[t], nan=0.0)
        elig = np.zeros(A, dtype=bool)
        for j in range(RISK_N):
            elig[j] = (np.isfinite(vol20[t, j]) and vol20[t, j] <= vol_cap
                       and np.isfinite(close[t, j]) and not bear[t, j])
        if elig.sum() == 0:
            return np.array([0.] * A), 0.0, []
        cand = np.where(elig)[0]
        k = min(top_n, len(cand))
        order = cand[np.argsort(-score[cand])][:k]
        nw = np.zeros(A)
        iv = 1.0 / np.maximum(vol20[t, order], 1e-6); iv /= iv.sum()
        for idx, j in enumerate(order):
            nw[j] = iv[idx]
        vsel = np.nan_to_num(vol20[t, order], nan=0.0)
        pv = float(np.sqrt(np.sum((iv * vsel) ** 2) + 2e-6))
        raw = np.clip(vol_target / (pv + 1e-9), 0.0, 1.0)
        nw[BOND] = 1.0 - raw
        return nw, raw, [int(j) for j in order]

    for t in range(warmup, T):
        day = sum(cur_w[j] * ret[t, j] for j in range(A)
                  if np.isfinite(ret[t, j]) and cur_w[j] != 0)
        equity *= (1 + day); eqc[t] = equity
        peak = max(peak, equity); dd = equity / peak - 1.0
        nw = None; ne = None
        if (not circuit) and dd <= -dd_limit:
            circuit = True; cd = dd_cooldown
            nw = np.zeros(A); nw[BOND] = 1.0; ne = 0.0
        if circuit:
            cd -= 1
            if cd <= 0 and dd >= -dd_limit * dd_recover and (t - warmup) % rebal == 0:
                circuit = False; nw, ne, cur_picked = select(t)
        if nw is None and (t - warmup) % rebal == 0 and t < T - 1 and (not circuit):
            nw, ne, cur_picked = select(t)
            filter_cnt += int((~bear[t, :RISK_N]).sum() < RISK_N)
        if nw is not None:
            delta = np.abs(nw - cur_w).sum() + abs(ne - cur_exp)
            equity *= (1 - cost * delta)
            cur_w, cur_exp = nw, ne
    return eqc, filter_cnt


def metrics(eqc, warmup=260):
    eq = eqc[warmup:]; eq = eq[np.isfinite(eq)]
    yrs = len(eq) / 252
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1
    pk = np.maximum.accumulate(eq); mdd = np.min(eq / pk - 1)
    dr = np.diff(eq) / eq[:-1]; shp = np.mean(dr) / (np.std(dr) + 1e-9) * np.sqrt(252)
    return cagr * 100, mdd * 100, shp, cagr / abs(mdd) if mdd < 0 else 0


base_bear = np.zeros_like(close, dtype=bool)
base_eq, _ = replay(base_bear)
base = metrics(base_eq)
print(f"基线(无熊市过滤): 年化={base[0]:.2f}% 回撤={base[1]:.2f}% 夏普={base[2]:.2f} 卡玛={base[3]:.2f}")
print()
print(f"{'因子':<20}{'年化':>9}{'回撤':>10}{'夏普':>8}{'卡玛':>8}{'Δ收益':>9}{'Δ回撤':>9}{'过滤日':>8}")
print('-' * 80)
results = {}
for kind in ['ma200', 'ma120', 'ma60', 'mom120_0', 'mom60_0', 'dd60_10', 'dd60_15', 'dd120_20', 'align', 'ma200_and_mom120']:
    bear = build_bear(kind)
    eq, fc = replay(bear)
    m = metrics(eq)
    results[kind] = dict(metrics=m, filter_days=fc)
    print(f"{kind:<20}{m[0]:>8.2f}%{m[1]:>9.2f}%{m[2]:>8.2f}{m[3]:>8.2f}"
          f"{m[0]-base[0]:>+8.2f}{m[1]-base[1]:>+8.2f}{fc:>8}")

# 黄金 11-09 能否被各因子挡住
t1109 = dates.index('2026-11-09')
print()
print('=== 黄金 11-09 是否被各因子判定走熊(应剔除) ===')
for kind in results:
    print(f"  {kind:<20}: {'剔除✓' if build_bear(kind)[t1109, 2] else '放行✗'}")
