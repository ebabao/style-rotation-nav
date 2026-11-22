"""
factor-evolution · 独立轮动回测引擎
====================================
重写, 不引用冻结的 v2.1 / research20.py。
机制: 多因子横截面打分选券 → 波动率目标定仓 → 零杠杆 → t-1决策 t生效 → 10bp成本。
支持择时 overlay (因子驱动的仓位缩放, 滞回)。
"""
import numpy as np

RISK = slice(0, 5)  # 前5列风险资产


def softmax(x, temp):
    x = np.clip(x, -50, 50) / temp
    x = x - np.nanmax(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def run(panel, F, params, scores=None):
    """
    params: dict
      factors: {name: weight}  因子权重 (可为负)
      rebal, vol_win, vol_target, vol_cap, top_n, rev_band,
      dd_limit, cash_bond, cost_bps, timing_mode, timing_k, timing_floor, ...
    scores: 可选 (T,A) 外部打分矩阵 (如 ML 预测), 注入到打分里
    返回 dict: equity(逐日净值), cagr, mdd, sharpe, calmar, turn, trades, rebal_days
    """
    c = panel.close
    r = panel.ret
    T, A = c.shape
    p = params
    fw = p['factors']
    rebal = int(p.get('rebal', 10))
    vol_win = int(p.get('vol_win', 20))
    vol_target = p.get('vol_target', 0.40)
    vol_cap = p.get('vol_cap', 0.25)
    top_n = int(p.get('top_n', 1))
    dd_limit = p.get('dd_limit', 0.18)
    dd_recover = p.get('dd_recover', 0.95)
    dd_cooldown = int(p.get('dd_cooldown', 60))
    cost = p.get('cost_bps', 10.0) / 1e4
    warmup = int(p.get('warmup', 260))
    floor = p.get('timing_floor', 0.0)

    # 预计算波动率 (t 日可见)
    ret = r.copy()
    vol = np.full((T, A), np.nan)
    for j in range(A):
        vol[:, j] = pd_roll_std(ret[:, j], vol_win) * np.sqrt(252)

    # 逐因子横截面标准化 (z-score, t 日可见)
    Fz = {}
    for name, w in fw.items():
        if name not in F or abs(w) < 1e-12:
            continue
        Fz[name] = zscore_cross(F[name])

    equity = 1.0
    peak = 1.0
    eq_curve = np.empty(T); eq_curve[:] = np.nan
    cur_w = np.zeros(A)
    cur_exp = 0.0
    turn = 0.0
    trades = 0
    rebal_days = []
    circuit = False; cd = 0
    tstate = 1.0  # 择时状态

    def select(t, score=None):
        """正常选券: 返回 (new_w, new_exp, order, score)。t 日收盘可见。"""
        if score is None:
            score = np.zeros(A)
            for name, w in fw.items():
                if name in Fz:
                    score += w * np.nan_to_num(Fz[name][t], nan=0.0)
            if scores is not None:
                score += np.nan_to_num(scores[t], nan=0.0)
        elig = np.zeros(A, dtype=bool)
        for j in range(5):
            elig[j] = np.isfinite(vol[t, j]) and vol[t, j] <= vol_cap and np.isfinite(c[t, j])
        n_ok = elig.sum()
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
        port_vol = float(np.sqrt(np.sum((iv * vsel)**2) + 2e-6))
        raw_exp = np.clip(vol_target / (port_vol + 1e-9), 0.0, 1.0) * tstate
        new_exp = np.clip(raw_exp, floor, 1.0)
        nw[5] = 1.0 - new_exp
        return nw, new_exp, [int(j) for j in order], score

    for t in range(warmup, T):
        # --- 持仓承担 t 日收益 (t-1 已定好 cur_w, cur_exp) ---
        day_ret = 0.0
        for j in range(A):
            if np.isfinite(r[t, j]) and np.isfinite(cur_w[j]) and cur_w[j] != 0:
                day_ret += cur_w[j] * r[t, j]
        equity *= (1.0 + day_ret)
        eq_curve[t] = equity
        peak = max(peak, equity)
        dd = equity / peak - 1.0

        new_w = None; new_exp = None; acted = False
        # --- 每日熔断检查 (t 日收盘 → t+1 生效) ---
        if (not circuit) and dd <= -dd_limit:
            circuit = True; cd = dd_cooldown
            new_w = np.zeros(A); new_w[5] = 1.0; new_exp = 0.0; acted = True
        if circuit and not acted:
            cd -= 1
            recovered = dd >= -dd_limit * dd_recover
            if cd <= 0 and recovered and (t - warmup) % rebal == 0 and t < T - 1:
                circuit = False
                new_w, new_exp, _, _ = select(t); acted = True
        # --- 常规再平衡 (非熔断、非熔断触发日) ---
        if not acted and (t - warmup) % rebal == 0 and t < T - 1 and (not circuit):
            new_w, new_exp, _, _ = select(t); acted = True

        if acted and new_w is not None:
            delta = np.abs(new_w - cur_w).sum() + abs(new_exp - cur_exp)
            turn += delta
            equity *= (1.0 - cost * delta)
            trades += int(np.sum(np.abs(new_w[:5] - cur_w[:5]) > 1e-6))
            cur_w = new_w
            cur_exp = new_exp
            rebal_days.append(t)
            if p.get('timing_mode', 'none') != 'none':
                tstate = timing_state(panel, F, t, p, tstate)

    # 指标
    eq = eq_curve[warmup:]
    eq = eq[np.isfinite(eq)]
    yrs = len(eq) / 252.0
    cagr = (eq[-1] / eq[0]) ** (1.0 / yrs) - 1.0 if yrs > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq / peak - 1.0))
    dret = np.diff(eq) / eq[:-1]
    sharpe = float(np.mean(dret) / (np.std(dret) + 1e-9) * np.sqrt(252)) if len(dret) > 2 else 0.0
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    return {
        'equity': eq_curve, 'cagr': float(cagr), 'maxdd': mdd,
        'sharpe': sharpe, 'calmar': float(calmar),
        'turn': float(turn), 'trades': trades, 'rebal_days': rebal_days,
        'final': float(eq[-1]), 'n_days': len(eq),
    }


def pd_roll_std(x, w):
    out = np.full(len(x), np.nan)
    for t in range(len(x)):
        seg = x[max(0, t-w+1):t+1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= max(3, w//2):
            out[t] = np.std(seg, ddof=1)
    return out


def zscore_cross(mat):
    """逐行横截面 z-score (T,A)。"""
    out = np.full_like(mat, np.nan)
    for t in range(mat.shape[0]):
        row = mat[t]
        ok = np.isfinite(row)
        if ok.sum() >= 3:
            mu, sd = row[ok].mean(), row[ok].std()
            out[t, ok] = (row[ok] - mu) / (sd + 1e-9)
    return out


def timing_state(panel, F, t, p, prev):
    """多信号投票择时 (t 日可见), 滞回。返回仓位缩放 0..1。"""
    mode = p.get('timing_mode', 'none')
    if mode == 'none':
        return 1.0
    # 用等权风险资产指数作市场代理 (前5列=风险资产, 不是前5行)
    c = panel.close[:, :5]
    idx = np.nanmean(c, axis=1)
    k = int(p.get('timing_k', 120))
    votes = 0.0; total = 0.0
    # 趋势: idx vs MA
    if t >= k:
        ma = np.mean(idx[t-k+1:t+1])
        if np.isfinite(ma):
            total += 1
            votes += 1.0 if idx[t] > ma else 0.0
    # 宽度: 站上 MA 的资产占比
    if t >= k:
        cnt = 0
        for j in range(5):
            win = np.asarray(c[t-k+1:t+1, j], dtype=float).ravel()
            if np.isfinite(c[t, j]) and np.isfinite(win).sum() >= k//2:
                if c[t, j] > np.nanmean(win):
                    cnt += 1
        total += 1
        votes += cnt / 5.0
    # TSMOM: 中期动量符号
    if t >= 60 and np.isfinite(idx[t-60]):
        total += 1
        votes += 1.0 if idx[t] > idx[t-60] else 0.0
    if total == 0:
        return prev
    frac = votes / total
    enter = p.get('vote_enter', 0.60)
    exit_ = p.get('vote_exit', 0.40)
    floor = p.get('timing_floor', 0.0)
    if prev >= 1.0 - 1e-6 and frac < exit_:
        return floor
    if prev <= floor + 1e-6 and frac > enter:
        return 1.0
    return prev
