# -*- coding: utf-8 -*-
"""选券因子进化：兼顾纳指上涨捕获 + 阴跌走熊过滤（研究线，不碰线上/v2.1）
==========================================================================
上一轮问题：MA200 且 mom120<0 会误伤纳指 2025-06（m60=+15.5% 强涨，但 m120
还覆盖旧低点=-2.7%）→ 错过涨幅；且熊转牛踏空 2019 -9.4pp。
本轮候选：
  A 三重确认 px<MA200 且 m60<0 且 m250<0  —— 价格+中期+长期全走熊才剔除
  B 双确认   px<MA200 且 m250<0
  C 中期确认 px<MA200 且 m60<0
  D 趋势减速 px<MA200 且 m60<m120
  E 反弹豁免 px<MA200 且 m60<0 但 m20>0 放行
  F 动态恢复 进入走熊后，一旦 px 上穿 MA20 立即恢复资格（状态机）
  G 走熊打分 多个信号计数>=阈值剔除
口径复用 v14_engine，无前视，parity=0。
重点验证：挡黄金11-09 / 纳指强势期不误伤 / 2019不踏空 / 逐年稳健。
"""
import sys, csv
import numpy as np
sys.path.insert(0, '/workspace/style-rotation-alert/nav')
import v14_engine as V

rows = list(csv.reader(open('/workspace/style-rotation-alert/data/etf_close_snapshot.csv', encoding='utf-8')))
dates = [r[0] for r in rows[1:]]
close = np.array([[float(r[i]) if r[i] else np.nan for i in range(1, 7)] for r in rows[1:]])
NAMES = ['CSI300', 'NDX100', 'Gold', 'ChiNext', 'STAR50', 'Bond']
BOND = 5; RN = V.RISK_N; TD = 252

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

T, A = close.shape


def momn(j, t, n):
    return close[t, j] / close[t - n, j] - 1 if t >= n and np.isfinite(close[t - n, j]) else np.nan


def man(j, t, n):
    seg = close[max(0, t - n + 1):t + 1, j]; seg = seg[np.isfinite(seg)]
    return seg.mean() if len(seg) >= n * 0.8 else np.nan


# 预计算各动量/均线矩阵
M = {n: np.full((T, A), np.nan) for n in (20, 60, 120, 250)}
MA = {n: np.full((T, A), np.nan) for n in (20, 200)}
for j in range(A):
    for t in range(T):
        for n in M:
            if np.isfinite(close[t, j]):
                M[n][t, j] = momn(j, t, n)
        for n in MA:
            if np.isfinite(close[t, j]):
                MA[n][t, j] = man(j, t, n)

below200 = close < MA[200]
m60neg = M[60] < 0
m250neg = M[250] < 0
m20pos = M[20] > 0
decel = M[60] < M[120]
aboveMA20 = close > MA[20]


def build_bear(kind):
    bear = np.zeros((T, A), dtype=bool)
    for j in range(RN):
        # F 动态恢复状态机
        in_bear = False
        for t in range(T):
            if not np.isfinite(close[t, j]):
                continue
            if kind == 'ma200_mom120':
                b = below200[t, j] and m60neg[t, j] and (M[120][t, j] < 0)  # 旧方案近似
            elif kind == 'triple':
                b = below200[t, j] and m60neg[t, j] and m250neg[t, j]
            elif kind == 'double':
                b = below200[t, j] and m250neg[t, j]
            elif kind == 'mid':
                b = below200[t, j] and m60neg[t, j]
            elif kind == 'decel':
                b = below200[t, j] and decel[t, j]
            elif kind == 'exempt':
                b = below200[t, j] and m60neg[t, j] and not m20pos[t, j]
            elif kind == ' dyn_recover':
                kind = 'dyn_recover'
                b = False  # 下方统一处理
            elif kind == 'dyn_recover':
                # 进入：px<MA200 且 m60<0；恢复：px 上穿 MA20
                if not in_bear and below200[t, j] and m60neg[t, j]:
                    in_bear = True
                if in_bear and aboveMA20[t, j]:
                    in_bear = False
                b = in_bear
            elif kind == 'score2':
                sig = int(below200[t, j]) + int(m60neg[t, j]) + int(m250neg[t, j]) + int(decel[t, j])
                b = sig >= 2
            elif kind == 'score3':
                sig = int(below200[t, j]) + int(m60neg[t, j]) + int(m250neg[t, j]) + int(decel[t, j])
                b = sig >= 3
            else:
                b = False
            bear[t, j] = b
    return bear


def replay(bear, vol_cap=0.25, dd_limit=0.18, dd_recover=0.95, dd_cooldown=60,
           rebal=10, vol_target=0.40, top_n=1, cost_bps=10, warmup=260):
    mz = V.zscore_cross(mom); cost = cost_bps / 1e4
    equity = 1.0; peak = 1.0; eqc = np.full(T, np.nan)
    cur_w = np.zeros(A); cur_exp = 0.0; circuit = False; cd = 0; picked = []
    picked_log = {}

    def select(t):
        score = np.nan_to_num(mz[t], nan=0.0)
        elig = np.zeros(A, dtype=bool)
        for j in range(RN):
            elig[j] = (np.isfinite(vol20[t, j]) and vol20[t, j] <= vol_cap
                       and np.isfinite(close[t, j]) and not bear[t, j])
        if elig.sum() == 0:
            return np.zeros(A), 0.0, []
        cand = np.where(elig)[0]; k = min(top_n, len(cand))
        order = cand[np.argsort(-score[cand])][:k]
        nw = np.zeros(A); iv = 1.0 / np.maximum(vol20[t, order], 1e-6); iv /= iv.sum()
        for idx, j in enumerate(order):
            nw[j] = iv[idx]
        vsel = np.nan_to_num(vol20[t, order], nan=0.0)
        pv = float(np.sqrt(np.sum((iv * vsel) ** 2) + 2e-6))
        raw = np.clip(vol_target / (pv + 1e-9), 0.0, 1.0)
        nw[BOND] = 1.0 - raw
        return nw, raw, [int(j) for j in order]

    for t in range(warmup, T):
        day = sum(cur_w[j] * ret[t, j] for j in range(A) if np.isfinite(ret[t, j]) and cur_w[j] != 0)
        equity *= (1 + day); eqc[t] = equity
        peak = max(peak, equity); dd = equity / peak - 1.0
        nw = None; ne = None
        if (not circuit) and dd <= -dd_limit:
            circuit = True; cd = dd_cooldown; nw = np.zeros(A); nw[BOND] = 1.0; ne = 0.0
        if circuit:
            cd -= 1
            if cd <= 0 and dd >= -dd_limit * dd_recover and (t - warmup) % rebal == 0:
                circuit = False; nw, ne, picked = select(t)
        if nw is None and (t - warmup) % rebal == 0 and t < T - 1 and (not circuit):
            nw, ne, picked = select(t)
        if nw is not None:
            delta = np.abs(nw - cur_w).sum() + abs(ne - cur_exp)
            equity *= (1 - cost * delta); cur_w, cur_exp = nw, ne
            for j in range(RN):
                if nw[j] > 0:
                    picked_log.setdefault(j, []).append(dates[t])
    return eqc, picked_log


def metrics(eqc, warmup=260):
    eq = eqc[warmup:]; eq = eq[np.isfinite(eq)]
    yrs = len(eq) / 252
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1
    pk = np.maximum.accumulate(eq); mdd = np.min(eq / pk - 1)
    dr = np.diff(eq) / eq[:-1]; shp = np.mean(dr) / (np.std(dr) + 1e-9) * np.sqrt(252)
    return cagr * 100, mdd * 100, shp, cagr / abs(mdd) if mdd < 0 else 0


base_eq, base_log = replay(np.zeros((T, A), dtype=bool))
base = metrics(base_eq)
print(f"基线: 年化={base[0]:.2f}% 回撤={base[1]:.2f}% 夏普={base[2]:.2f} 卡玛={base[3]:.2f}")
print()
print(f"{'方案':<16}{'年化':>9}{'回撤':>10}{'夏普':>8}{'卡玛':>8}{'Δ收益':>9}{'Δ回撤':>9}{'挡黄金':>8}{'纳指入选':>10}")
print('-' * 92)
t1109 = dates.index('2026-11-09')
for kind in ['ma200_mom120', 'triple', 'double', 'mid', 'decel', 'exempt', 'dyn_recover', 'score2', 'score3']:
    bear = build_bear(kind)
    eq, log = replay(bear)
    m = metrics(eq)
    gold_block = '✓' if bear[t1109, 2] else '✗'
    ndx_cnt = len(log.get(1, []))
    print(f"{kind:<16}{m[0]:>8.2f}%{m[1]:>9.2f}%{m[2]:>8.2f}{m[3]:>8.2f}"
          f"{m[0]-base[0]:>+8.2f}{m[1]-base[1]:>+8.2f}{gold_block:>8}{ndx_cnt:>10}")

# 纳指强势期误伤检查：m60>10% 且被过滤
print()
print('=== 纳指强势期(m60>10%)各方案误伤天数 ===')
strong = [t for t in range(260, T) if np.isfinite(M[60][t, 1]) and M[60][t, 1] > 0.10]
print(f'纳指 m60>10% 的交易日共 {len(strong)} 天')
for kind in ['ma200_mom120', 'triple', 'double', 'mid', 'dyn_recover']:
    bear = build_bear(kind)
    hit = sum(bear[t, 1] for t in strong)
    print(f'  {kind:<16} 被误伤 {hit:>4} 天 ({hit/len(strong)*100:.1f}%)')

# 2019 收益检查
print()
print('=== 2019 收益（熊转牛踏空检验）===')
idx19 = [i for i, d in enumerate(dates) if d.startswith('2019') and np.isfinite(base_eq[i])]
for kind in ['base', 'ma200_mom120', 'triple', 'double', 'dyn_recover', 'score3']:
    eq = base_eq if kind == 'base' else replay(build_bear(kind))[0]
    e = np.array(eq[idx19]); r19 = (e[-1] / e[0] - 1) * 100
    print(f'  {kind:<16} 2019={r19:+.1f}%')
