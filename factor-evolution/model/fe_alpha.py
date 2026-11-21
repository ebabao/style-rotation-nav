# -*- coding: utf-8 -*-
"""factor-evolution · 去漂移 alpha 引擎 + git 机构因子择时
============================================================
目标：把进化目标从"账面 CAGR"改为"去漂移 alpha 的真 Sharpe"，
剥离资产长期 beta（不可外推），只保留选券/择时的真超额。

设计要点：
  1) 去漂移用"严格无前视滚动 mu"：mu_hat[t]=mean(logr[t-756..t-1])，
     修正 timing_data.detrend_close 用全样本 mu 的前视缺陷。
  2) 横截面选券因子在【去漂移价格】上计算 → 选的是相对路径强度，非绝对漂移。
  3) git 机构因子（第29章14因子）在【原始价格】上算市场级择时信号，
     方向由训练段 IC 定（不预设），连续仓位缩放而非硬开关（避开高AUC陷阱）。
  4) 评估用去漂移收益 = 真 alpha；同时给账面收益对照。

git 因子来源（可查证，第29章）：
  RSRS(光大/TenghanZhong) 高阶矩(Harvey&Siddique2000) Diffusion(buckstrdr)
  T-Sharpe(Whitelaw1994) FIP(BravoRush) 波动收缩/自相关(yingwang/trade)
"""
import numpy as np
import pandas as pd

RISK = slice(0, 5)  # 前5列风险资产
BOND = 5


# ---------------- 去漂移（严格无前视）----------------
def detrend_market(close):
    """市场中性去漂移（推荐）：r_alpha[t,j] = r[t,j] - r_mkt[t]。
    r_mkt = 当日 5 个风险资产等权收益（同期基准，仅用于评估真超额，
    不参与任何 t 日决策）。剔除共同 beta，保留资产相对强弱 α。
    去漂移后等权市场净值恒为 1，策略净值即"相对市场的累计超额"。
    """
    with np.errstate(all='ignore'):
        ret = close[1:] / close[:-1] - 1.0
        rmkt = np.nanmean(ret[:, :5], axis=1)
        ra = ret - rmkt[:, None]
    out = np.full_like(close, np.nan)
    out[0] = 1.0
    for t in range(1, close.shape[0]):
        for j in range(close.shape[1]):
            prev = out[t-1, j] if np.isfinite(out[t-1, j]) else 1.0
            out[t, j] = prev * (1.0 + (ra[t-1, j] if np.isfinite(ra[t-1, j]) else 0.0))
    return out


def detrend_rolling(close, win=756, minp=252):
    """【已弃用，仅存档】逐资产减自身漂移会抹掉绝对收益差异，
    导致选券 alpha 与账面收益背离。保留供对照，主引擎请用 detrend_market。"""
    with np.errstate(all='ignore'):
        logr = np.full_like(close, np.nan)
        logr[1:] = np.log(close[1:] / close[:-1])
    out = np.full_like(close, np.nan)
    out[0] = 1.0
    for t in range(1, close.shape[0]):
        for j in range(close.shape[1]):
            if not np.isfinite(close[t, j]) or not np.isfinite(close[t-1, j]):
                continue
            hist = logr[max(1, t-win):t, j]
            hist = hist[np.isfinite(hist)]
            mu = hist.mean() if len(hist) >= minp else 0.0
            prev = out[t-1, j] if np.isfinite(out[t-1, j]) else 1.0
            out[t, j] = prev * np.exp(logr[t, j] - mu)
    return out


# ---------------- git 市场因子（自包含，原始价格）----------------
def build_git_factors(panel):
    """5 风险资产等权合成市场级 high/low/close，计算第29章14因子。
    返回 {name: (T,) ndarray}，仅用 t 及之前数据（零前视）。"""
    c = panel.close[:, :5]
    h = panel.high[:, :5]
    l = panel.low[:, :5]
    with np.errstate(all='ignore'):
        nrm = c / c[0]
        mkt_c = np.nanmean(nrm, axis=1)
        mkt_h = np.nanmean(h / c[0], axis=1)
        mkt_l = np.nanmean(l / c[0], axis=1)
    n = len(mkt_c)
    F = {}

    def _roll(a, w, how, minp=None):
        return getattr(pd.Series(a).rolling(w, min_periods=minp or max(3, w//2)), how)().to_numpy()

    # 1. RSRS: high = a + b*low 斜率
    win = 18
    beta = np.full(n, np.nan); r2 = np.full(n, np.nan)
    for t in range(win, n):
        x, y = l[t-win+1:t+1], h[t-win+1:t+1]
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        xm, ym = x.mean(), y.mean()
        sxx = ((x-xm)**2).sum()
        if sxx <= 0: continue
        b = ((x-xm)*(y-ym)).sum()/sxx
        a = ym - b*xm
        yy = y-(a+b*x); sst = ((y-ym)**2).sum()
        beta[t] = b; r2[t] = 1-((yy**2).sum())/sst if sst > 0 else 0
    F['rsrs_slope'] = beta
    def _zlast(v):
        v = np.asarray(v); v = v[np.isfinite(v)]
        return (v[-1]-v.mean())/(v.std()+1e-9) if len(v) > 1 else np.nan
    bz = pd.Series(beta).rolling(252, min_periods=60).apply(_zlast, raw=True).to_numpy()
    F['rsrs_adj'] = bz*r2

    # 2. 高阶矩
    ret = np.full(n, np.nan)
    ret[1:] = mkt_c[1:]/mkt_c[:-1]-1
    rs = pd.Series(ret)
    for w2, tag in ((20,'20'),(60,'60')):
        F[f'skew{tag}'] = rs.rolling(w2, min_periods=w2).skew().to_numpy()
        F[f'kurt{tag}'] = rs.rolling(w2, min_periods=w2).kurt().to_numpy()
    m5 = rs.rolling(40, min_periods=40).apply(
        lambda v: np.mean(((v-np.mean(v))/(np.std(v)+1e-9))**5), raw=True)
    F['m5_z'] = m5.rolling(252, min_periods=60).apply(_zlast, raw=True).to_numpy()

    # 3. Diffusion
    above = []
    for m in (20, 60, 120):
        ma = pd.DataFrame(c).rolling(m, min_periods=m).mean().to_numpy()
        above.append(np.nanmean((c > ma).astype(float), axis=1))
    F['diffusion20'] = np.nanmean(np.vstack(above), axis=0)
    mkt_up = mkt_c/np.roll(mkt_c, 20)-1
    dif_s = pd.Series(F['diffusion20']).rolling(5).mean().to_numpy()
    F['diffusion_div'] = np.where(np.isfinite(mkt_up)&np.isfinite(dif_s), mkt_up-dif_s, np.nan)

    # 4. T-Sharpe
    mu60 = _roll(ret, 60, 'mean'); sd60 = _roll(ret, 60, 'std')
    F['tsharpe'] = mu60/(sd60+1e-9)*np.sqrt(252)

    # 5. FIP
    pos = np.where(np.isfinite(ret), (ret > 0).astype(float), np.nan)
    F['fip252'] = pd.Series(pos).rolling(252, min_periods=252).mean().to_numpy()

    # 6. 波动收缩 + 自相关
    v10 = _roll(ret, 10, 'std'); v63 = _roll(ret, 63, 'std')
    F['vol_contraction'] = v10/(v63+1e-9)
    trend = np.where(mkt_c > np.roll(mkt_c, 63), 1.0, -1.0)
    F['vol_trend'] = F['vol_contraction']*trend
    def _ac1(v):
        v = np.asarray(v); v = v[np.isfinite(v)]
        if len(v) < 3 or v.std() == 0: return np.nan
        return float(np.corrcoef(v[:-1], v[1:])[0, 1])
    F['autocorr21'] = rs.rolling(21, min_periods=21).apply(_ac1, raw=True).to_numpy()

    for k in F:
        F[k] = np.nan_to_num(F[k], nan=np.nan, posinf=np.nan, neginf=np.nan)
    return F


GIT_NAMES = ['rsrs_slope','rsrs_adj','skew20','kurt20','skew60','kurt60','m5_z',
             'diffusion20','diffusion_div','tsharpe','fip252','vol_contraction',
             'vol_trend','autocorr21']
GIT_SOURCE = {
    'rsrs_slope':'RSRS/光大', 'rsrs_adj':'RSRS×R²/光大',
    'skew20':'高阶矩20d/Harvey-Siddique2000', 'kurt20':'高阶矩峰度20d',
    'skew60':'高阶矩60d', 'kurt60':'高阶矩峰度60d', 'm5_z':'5阶矩z',
    'diffusion20':'Diffusion扩散宽度/buckstrdr', 'diffusion_div':' Diffusion背离',
    'tsharpe':'时变夏普/Whitelaw1994', 'fip252':'Frog-in-the-Pan/BravoRush',
    'vol_contraction':'波动收缩/yingwang', 'vol_trend':'波动收缩+趋势',
    'autocorr21':'21d自相关/yingwang'}


# ---------------- 方向判定（训练段 IC，无前视）----------------
def git_directions(F_git, dret_mkt, lo, hi, fwd_h=20):
    """用训练段算每个 git 因子与未来 h 日市场去漂移收益的秩相关，定择时方向。"""
    from scipy.stats import spearmanr
    dirs = {}
    fwd = pd.Series(dret_mkt).rolling(fwd_h).sum().shift(-fwd_h).to_numpy()
    for name in GIT_NAMES:
        x, y = F_git[name], fwd
        m = np.isfinite(x)&np.isfinite(y)
        m[:lo] = False; m[hi:] = False
        if m.sum() < 60:
            dirs[name] = 0.0; continue
        r = spearmanr(x[m], y[m]).correlation
        dirs[name] = float(np.sign(r)) if np.isfinite(r) else 0.0
    return dirs


# ---------------- 去漂移 alpha 引擎 ----------------
def zscore_cross(mat):
    out = np.full_like(mat, np.nan)
    for t in range(mat.shape[0]):
        row = mat[t]; ok = np.isfinite(row)
        if ok.sum() >= 3:
            out[t, ok] = (row[ok]-row[ok].mean())/(row[ok].std()+1e-9)
    return out


def run_alpha(panel, dclose, F_cross, F_git, git_W, params, git_dirs=None):
    """在去漂移价格上选券+择时，评估真 alpha；同时给账面对照。
    F_cross: {name:(T,A)} 在去漂移价格上算的横截面因子
    F_git:   {name:(T,)} 原始价格市场因子
    git_W:   {name:weight} 择时权重（可为负，方向由 git_dirs 或权重符号定）
    返回 dict: alpha_equity, book_equity, metrics...
    """
    p = params
    dret = np.zeros_like(dclose)
    dret[1:] = dclose[1:]/dclose[:-1]-1
    dret[0] = np.nan
    oret = panel.ret
    T, A = dclose.shape
    fw = p['factors']
    rebal = int(p.get('rebal', 10))
    vol_win = int(p.get('vol_win', 20))
    vol_target = p.get('vol_target', 0.40)
    vol_cap = p.get('vol_cap', 0.25)
    top_n = int(p.get('top_n', 1))
    dd_limit = p.get('dd_limit', 0.45)
    cost = p.get('cost_bps', 10)/1e4
    warmup = int(p.get('warmup', 260))
    floor = p.get('timing_floor', 0.0)
    timing_on = bool(git_W) and abs(sum(abs(w) for w in git_W.values())) > 1e-9

    # 去漂移波动（决策时可见）
    vol = np.full((T, A), np.nan)
    for j in range(A):
        vol[:, j] = pd.Series(dret[:, j]).rolling(vol_win, min_periods=max(3, vol_win//2)).std().to_numpy()*np.sqrt(252)

    Fz = {name: zscore_cross(F_cross[name]) for name, w in fw.items()
          if name in F_cross and abs(w) > 1e-12}

    # git 择时信号：z-score 后按方向加权，再映射连续仓位 [floor,1]
    if timing_on:
        gz = {}
        for name, w in git_W.items():
            if name in F_git and abs(w) > 1e-12:
                s = pd.Series(F_git[name])
                z = (s - s.rolling(252, min_periods=60).mean())/(s.rolling(252, min_periods=60).std()+1e-9)
                gz[name] = z.to_numpy()
        wsum = sum(abs(w) for w in git_W.values())
        raw_sig = np.zeros(T)
        for name, w in git_W.items():
            if name in gz:
                d = git_dirs.get(name, 1.0) if git_dirs else (1.0 if w >= 0 else -1.0)
                raw_sig += w*d*np.nan_to_num(gz[name], nan=0.0)
        raw_sig = raw_sig/(wsum+1e-9)

    eq_a = 1.0; eq_b = 1.0; peak_a = 1.0
    ea = np.full(T, np.nan); eb = np.full(T, np.nan)
    cur_w = np.zeros(A); cur_exp = 0.0; trades = 0; turn = 0.0
    cooldown = 0
    for t in range(warmup, T):
        # 承担 t 日收益
        da = 0.0; db = 0.0
        for j in range(A):
            if np.isfinite(cur_w[j]) and cur_w[j] != 0:
                if np.isfinite(dret[t, j]): da += cur_w[j]*dret[t, j]
                if np.isfinite(oret[t, j]): db += cur_w[j]*oret[t, j]
        eq_a *= (1+da); eq_b *= (1+db)
        ea[t] = eq_a; eb[t] = eq_b
        peak_a = max(peak_a, eq_a); dd = eq_a/peak_a-1

        if (t-warmup) % rebal == 0 and t < T-1:
            score = np.zeros(A)
            for name, w in fw.items():
                if name in Fz:
                    score += w*np.nan_to_num(Fz[name][t], nan=0.0)
            elig = np.array([np.isfinite(vol[t, j]) and vol[t, j] <= vol_cap and np.isfinite(dclose[t, j])
                             for j in range(5)])
            if elig.sum() == 0:
                nw = np.zeros(A); nw[BOND] = 1.0; ne = 0.0
            else:
                cand = np.where(elig)[0]; k = min(top_n, len(cand))
                order = cand[np.argsort(-score[cand])][:k]
                nw = np.zeros(A)
                iv = 1/np.maximum(np.nan_to_num(vol[t, order], nan=1.0), 1e-6)
                iv = iv/iv.sum()
                for idx, j in enumerate(order): nw[j] = iv[idx]
                pv = float(np.sqrt(np.sum((iv*np.nan_to_num(vol[t, order], nan=0.0))**2)+2e-6))
                exp = np.clip(vol_target/(pv+1e-9), 0, 1)
                if timing_on:
                    sig = np.clip(raw_sig[t], -1, 1)
                    scale = np.clip((sig+1)/2, 0, 1)  # [-1,1] -> [0,1]
                    exp *= (floor + (1-floor)*scale)
                exp = np.clip(exp, floor if not timing_on else 0.0, 1.0)
                if dd <= -dd_limit: cooldown = int(p.get('dd_cooldown', 10))
                if cooldown > 0: exp *= 0.5; cooldown -= 1
                nw[BOND] = 1-exp; ne = exp
            delta = np.abs(nw-cur_w).sum()+abs(ne-cur_exp)
            eq_a *= (1-cost*delta); eq_b *= (1-cost*delta)
            turn += delta
            trades += int(np.sum(np.abs(nw[:5]-cur_w[:5]) > 1e-6))
            cur_w, cur_exp = nw, ne

    def metrics(curve):
        s = curve[warmup:]; s = s[np.isfinite(s)]
        if len(s) < 60: return dict(cagr=0, maxdd=0, sharpe=0, calmar=0)
        s = s/s[0]; yrs = len(s)/252
        cagr = s[-1]**(1/yrs)-1
        pk = np.maximum.accumulate(s); mdd = float(np.min(s/pk-1))
        dr = np.diff(s)/s[:-1]
        sh = float(np.mean(dr)/(np.std(dr)+1e-9)*np.sqrt(252))
        return dict(cagr=float(cagr), maxdd=mdd, sharpe=sh,
                    calmar=float(cagr/abs(mdd)) if mdd < 0 else 0)
    return dict(alpha_equity=ea, book_equity=eb,
                alpha=metrics(ea), book=metrics(eb),
                turn=float(turn), trades=trades)


def seg_metrics(curve, warm, segs):
    out = {}
    for k, (a, b) in segs.items():
        s = curve[a:b+1]; s = s[np.isfinite(s)]
        if len(s) < 60: out[k] = dict(cagr=0, maxdd=0, sharpe=0); continue
        s = s/s[0]; yrs = len(s)/252
        cagr = s[-1]**(1/yrs)-1
        pk = np.maximum.accumulate(s); mdd = float(np.min(s/pk-1))
        dr = np.diff(s)/s[:-1]
        sh = float(np.mean(dr)/(np.std(dr)+1e-9)*np.sqrt(252))
        out[k] = dict(cagr=float(cagr), maxdd=float(mdd), sharpe=float(sh))
    return out
