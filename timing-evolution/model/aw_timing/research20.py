# -*- coding: utf-8 -*-
"""
v2.0 INSTITUTIONAL TIMING research: 6-asset zero-leverage, single-engine.

Upgrades over v1.9 (all in the timing overlay; core alpha engine unchanged):
  Timing used to be a binary MA/breadth gate (idx>MA ? full : floor) which
  whipsaws in choppy markets and got naturally eliminated. v2.0 replaces it
  with a SIGNAL LIBRARY + CONTINUOUS position scaling + HYSTERESIS:

  timing_mode:
    none          no overlay (bit-identical to v1.9)
    vol_managed   Moreira-Muir(2017): scale inversely with realized-vol percentile
                  (high-vol regimes deliver worse risk-adjusted returns -> cut).
    tsmom         Moskowitz-Ooi-Pedersen(2012): continuous tanh of index momentum,
                  positive trend -> full, negative -> de-risk smoothly.
    multi_vote    weighted vote of {trend, breadth, tsmom, vol-regime}, with a
                  hysteresis state machine (enter/exit thresholds) to cut whipsaw.
    ma_smooth     v1.9-style MA gate but with smoothing buffer (enter above ma*(1+b),
                  exit below ma*(1-b)) -> fewer regime flips.

  All timing signals are computed from past data only and lagged by signal_lag
  (t-1 decision, t execution): strictly no look-ahead.

Architecture: ONE Engine20 class powers backtest + all stress tests, so parity
is structurally 0. With timing_mode='none' and w_ts=0 and weight_mode in
{inv_vol,equal,score}, output is bit-identical to research19/v1.7.
"""
import sys, json, math, time, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import pandas as pd
from engine13 import (FactorBank13, ASSETS13, RISK_ASSETS, MOM_LBS13)
from engine import TRADING_DAYS, CASH_DAILY

# 自包含：数据面板随子工程一起发布，不依赖 /workspace 外部路径
DATA = os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'data', 'etf_close_snapshot.csv')
COST = 0.0010
FIN_ANNUAL = 0.030
RF = 0.0
BLEND_KEYS = [(20, 0), (60, 0), (120, 0)]
VOL_TIMING_WINS = (20, 40, 60)
TSMOM_LBS = (40, 60, 90, 120, 160, 200)


def zrow(M):
    m = np.nanmean(M, axis=1, keepdims=True)
    s = np.nanstd(M, axis=1, keepdims=True)
    return np.nan_to_num((M - m) / np.where(s > 1e-9, s, 1.0),
                         nan=0.0, posinf=0.0, neginf=0.0)


class Engine20:
    """Single source of truth for v2.0 backtest and stress tests."""

    def __init__(self, close, risk_assets=None):
        self.fb = FactorBank13(close)
        self.close = self.fb.close
        self.dates = self.fb.dates
        self.T, self.A = self.fb.T, self.fb.A
        self.assets = self.fb.assets
        self.ret1 = self.fb.ret1
        _risk = set(RISK_ASSETS) if risk_assets is None else set(risk_assets)
        self.risk = np.array([a in _risk for a in self.assets])
        self.bond = self.fb.bond_col
        fb = self.fb
        self.ZM = {k: zrow(v) for k, v in fb.mom.items()}
        self.ZV = {k: zrow(v) for k, v in fb.vol.items()}
        self.ZT = {k: zrow(v) for k, v in fb.trend.items()}
        self.ZA = {k: zrow(v) for k, v in fb.accel.items()}
        self.ZS = {k: zrow(v) for k, v in fb.sharpe_mom.items()}
        self.ZBLEND = np.nanmean(np.stack([self.ZM[k] for k in BLEND_KEYS]), axis=0)
        self.TSM = {lb: np.tanh(np.nan_to_num(fb.sharpe_mom[lb], nan=0.0))
                    for lb in MOM_LBS13}
        self.FH = fb.from_high
        # ---- equal-weight risk-asset index (regime timing base) ----
        risk_r = np.nanmean(np.where(np.isfinite(self.ret1[:, self.risk]),
                                     self.ret1[:, self.risk], 0.0), axis=1)
        self.idx = np.cumprod(1.0 + np.nan_to_num(risk_r, nan=0.0))
        ser = pd.Series(self.idx)
        self.IDX_MA = {k: ser.rolling(k, min_periods=k).mean().to_numpy()
                       for k in (60, 120, 200, 250)}
        self.BREADTH = {}
        for k in (60, 120, 200, 250):
            ma = pd.DataFrame(fb.close).rolling(k, min_periods=k).mean().to_numpy()
            with np.errstate(invalid='ignore'):
                above = (fb.close / ma - 1.0) > 0
            self.BREADTH[k] = np.nanmean(
                np.where(np.isfinite(above[:, self.risk]), above[:, self.risk], False), axis=1)
        # NEW: rolling realized volatility of the index (annualized) + its trailing
        # percentile rank. Percentile is computed on a trailing window only (no look-ahead).
        self.IDX_VOL = {}
        self.IDX_VOL_PCTL = {}
        idxret = pd.Series(np.nan_to_num(risk_r, nan=0.0))
        for w in VOL_TIMING_WINS:
            rv = (idxret.rolling(w, min_periods=w).std() * np.sqrt(TRADING_DAYS)).to_numpy()
            self.IDX_VOL[w] = rv
            # trailing percentile of current vol among past 252 observations
            s = pd.Series(rv)
            self.IDX_VOL_PCTL[w] = s.rolling(252, min_periods=120).rank(pct=True).to_numpy()
        # NEW: continuous time-series momentum of the index over several lookbacks
        self.TSMOM = {}
        for lb in TSMOM_LBS:
            self.TSMOM[lb] = np.tanh(ser / ser.shift(lb) - 1.0).to_numpy()  # -1..1
        self.warmup = fb.warmup
        # v2.1 可选: 正交新因子叠加钩子 (默认关闭, 不影响历史结果)
        self.extra_factors = None   # dict factor_name -> weight
        self.extra_z = {}           # dict factor_name -> z-score matrix (T,A)

    def set_extra(self, weights, zmat):
        """注入新因子权重与z矩阵; weights=None 关闭叠加。"""
        self.extra_factors = weights
        self.extra_z = zmat

    def _timing_scale(self, p, d, state):
        """Return exposure scale in [floor, 1] and (possibly updated) hysteresis state.
        All signals indexed at d (already lagged by the caller)."""
        mode = p.get('timing_mode', 'none')
        if mode == 'none':
            return 1.0, state
        floor = float(p.get('timing_floor', 0.5))
        k = int(p.get('timing_k', 200))
        # pick nearest supported window
        def near(keys, target):
            return min(keys, key=lambda x: abs(x - target))
        if mode == 'vol_managed':
            # Moreira-Muir: inverse realized-volatility scaling.
            # High vol percentile -> cut exposure; low -> full. Continuous.
            wv = near(VOL_TIMING_WINS, k)
            pctl = self.IDX_VOL_PCTL[wv][d]
            if not np.isfinite(pctl):
                return 1.0, state
            # aggressiveness: how strongly we cut as vol rises
            aggr = float(p.get('timing_aggr', 1.0))
            # scale = 1 - aggr*floor*pctl^gamma  (monotone down in vol percentile)
            gamma = float(p.get('timing_gamma', 1.0))
            scale = 1.0 - aggr * (1.0 - floor) * (pctl ** gamma)
            return float(np.clip(scale, floor, 1.0)), state
        if mode == 'tsmom':
            # continuous momentum: tanh signal in [-1,1] mapped to [floor,1]
            lb = near(TSMOM_LBS, k)
            s = self.TSMOM[lb][d]
            if not np.isfinite(s):
                return 1.0, state
            # map (-1..1) -> (floor..1): (s+1)/2 in [0,1]
            scale = floor + (1.0 - floor) * (s + 1.0) / 2.0
            return float(np.clip(scale, floor, 1.0)), state
        if mode == 'ma_smooth':
            # hysteresis around MA: enter risk-on when idx > ma*(1+band),
            # stay until idx < ma*(1-band). state=1 risk-on, 0 risk-off.
            ma = self.IDX_MA[near((60, 120, 200, 250), k)][d]
            if not np.isfinite(ma):
                return 1.0, state
            band = float(p.get('timing_band', 0.02))
            ratio = self.idx[d] / ma - 1.0
            if state == 1:
                if ratio < -band:
                    state = 0
            else:
                if ratio > band:
                    state = 1
            return (1.0 if state == 1 else floor), state
        if mode == 'multi_vote':
            # weighted vote of 4 signals with hysteresis on the aggregate score.
            # signals each contribute in [0,1] (1 = risk-on).
            w_trend = float(p.get('vote_trend', 1.0))
            w_breath = float(p.get('vote_breadth', 1.0))
            w_ts = float(p.get('vote_tsmom', 1.0))
            w_vol = float(p.get('vote_vol', 1.0))
            wsum = w_trend + w_breath + w_ts + w_vol + 1e-9
            parts = []
            # trend
            ma = self.IDX_MA[near((60, 120, 200, 250), k)][d]
            tr = 1.0 if (np.isfinite(ma) and self.idx[d] > ma) else 0.0
            parts.append(w_trend * tr)
            # breadth
            b = self.BREADTH[near((60, 120, 200, 250), k)][d]
            br = 0.0
            if np.isfinite(b):
                thr = float(p.get('breadth_thr', 0.5))
                br = float(np.clip((b - (thr - 0.2)) / 0.4, 0.0, 1.0))
            parts.append(w_breath * br)
            # tsmom
            lb = near(TSMOM_LBS, k)
            s = self.TSMOM[lb][d]
            tsv = (np.clip(s, -1.0, 1.0) + 1.0) / 2.0 if np.isfinite(s) else 0.5
            parts.append(w_ts * tsv)
            # vol-regime (low vol percentile = risk-on)
            wv = near(VOL_TIMING_WINS, k)
            pctl = self.IDX_VOL_PCTL[wv][d]
            vv = (1.0 - np.clip(pctl, 0.0, 1.0)) if np.isfinite(pctl) else 0.5
            parts.append(w_vol * vv)
            score = sum(parts) / wsum  # [0,1], higher = risk-on
            # hysteresis on aggregate: risk-on if score>enter, stay until score<exit
            enter = float(p.get('vote_enter', 0.60))
            exit_ = float(p.get('vote_exit', 0.40))
            if state == 1:
                if score < exit_:
                    state = 0
            else:
                if score > enter:
                    state = 1
            # continuous blend: soften the binary state with the score itself
            soft = float(p.get('vote_soft', 0.35))  # 0=pure binary,1=pure continuous
            lvl = (1.0 - soft) * (1.0 if state == 1 else 0.0) + soft * score
            return float(np.clip(floor + (1.0 - floor) * lvl, floor, 1.0)), state
        return 1.0, state

    def run(self, p, start=None, end=None, initial=1.0, cost=None, fin_annual=None,
            signal_lag=None):
        cost = COST if cost is None else float(cost)
        fin_annual = FIN_ANNUAL if fin_annual is None else float(fin_annual)
        start = self.warmup if start is None else max(start, self.warmup)
        end = self.T if end is None else min(end, self.T)
        lag = int(p.get('signal_lag', 1)) if signal_lag is None else int(signal_lag)
        mom_lb, skip = int(p['mom_lb']), int(p['skip'])
        vol_win, high_win = int(p['vol_win']), int(p['high_win'])
        ma_len = int(p['ma_len'])
        M = self.fb.mom[(mom_lb, skip)]
        V = self.fb.vol[vol_win]
        TR = self.fb.trend[ma_len] if ma_len else None
        ZTR = self.ZT[ma_len] if ma_len else None
        F = self.FH[high_win]
        zM = self.ZBLEND if p.get('blend') else self.ZM[(mom_lb, skip)]
        zV = self.ZV[vol_win]
        zA = self.ZA[mom_lb]
        zS = self.ZS[mom_lb]
        tsm = self.TSM[mom_lb]

        wv = np.array([p['w_mom'], p['w_vol'], p['w_accel'], p['w_sharpe'],
                       p['w_trend'], p.get('w_ts', 0.0)], float)
        wv = wv / (np.abs(wv).sum() + 1e-9)

        top_n, rebal = int(p['top_n']), int(p['rebal'])
        vol_target, lev_cap = float(p['vol_target']), float(p['lev_cap'])
        abs_thr, vol_cap = float(p['abs_thr']), float(p['vol_cap'])
        dd_limit, dd_cd = float(p['dd_limit']), int(p['dd_cooldown'])
        high_dd = float(p['high_dd_limit'])
        wmode = p['weight_mode']
        cash_mode = p['cash_mode']
        softmax_temp = max(1e-3, float(p.get('softmax_temp', 0.15)))
        blend_alpha = float(np.clip(p.get('blend_alpha', 0.5), 0.0, 1.0))

        w = np.zeros(self.A)
        exp = 0.0
        equity = [initial]
        exposures = []
        turnover_total, trades = 0.0, 0
        circuit_until, peak = -1, initial
        timing_mode = p.get('timing_mode', 'none')
        tstate = 1  # hysteresis state: 1 = risk-on initially

        for t in range(start, end):
            if (t - start) % rebal == 0:
                d = t - lag
                score = (wv[0] * zM[d] - wv[1] * zV[d] + wv[2] * zA[d]
                         + wv[3] * zS[d]
                         + (wv[4] * ZTR[d] if ZTR is not None else 0.0)
                         + (wv[5] * tsm[d] if wv[5] != 0 else 0.0))
                # v2.1: 叠加正交新因子 (默认关闭)
                if self.extra_factors:
                    for fname, fw in self.extra_factors.items():
                        if abs(fw) > 1e-9 and fname in self.extra_z:
                            score = score + fw * np.nan_to_num(self.extra_z[fname][d], nan=0.0)
                dd = equity[-1] / peak - 1.0
                if dd < -dd_limit:
                    circuit_until = t + dd_cd
                eligible = np.isfinite(M[d]) & np.isfinite(V[d]) & self.risk
                eligible &= (M[d] > abs_thr) & (V[d] < vol_cap) & (V[d] > 0)
                eligible &= (F[d] > -high_dd)
                if TR is not None:
                    eligible &= np.isfinite(TR[d]) & (TR[d] > 0)
                trail = (t <= circuit_until) or (not eligible.any())
                if trail:
                    if cash_mode == 'bond' and self.bond >= 0 and np.isfinite(self.close[d, self.bond]):
                        new_w = np.zeros(self.A); new_w[self.bond] = 1.0; new_exp = 1.0
                    else:
                        new_w, new_exp = np.zeros(self.A), 0.0
                else:
                    idxs = np.argsort(-np.where(eligible, score, -np.inf))[:top_n]
                    idxs = idxs[eligible[idxs]]
                    if len(idxs) == 0:
                        new_w, new_exp = np.zeros(self.A), 0.0
                    else:
                        if wmode == 'inv_vol':
                            iv = 1.0 / np.maximum(V[d, idxs], 0.02); wi = iv / iv.sum()
                        elif wmode == 'score':
                            s = np.clip(score[idxs] - np.nanmin(score[idxs]) + 0.02, 0.02, None)
                            wi = s / s.sum()
                        elif wmode == 'softmax':
                            s = score[idxs] / softmax_temp
                            s = np.clip(s - np.max(s), -30, 30)
                            ew = np.exp(s); wi = ew / ew.sum()
                        elif wmode == 'blend':
                            iv = 1.0 / np.maximum(V[d, idxs], 0.02); iv = iv / iv.sum()
                            eqw = np.ones(len(idxs)) / len(idxs)
                            wi = blend_alpha * eqw + (1.0 - blend_alpha) * iv
                        else:
                            wi = np.ones(len(idxs)) / len(idxs)
                        new_w = np.zeros(self.A); new_w[idxs] = wi
                        lo = max(0, d - vol_win + 1)
                        pr = np.nansum(self.ret1[lo:d + 1][:, idxs] * wi, axis=1)
                        pv = float(np.nanstd(pr) * np.sqrt(TRADING_DAYS))
                        if not np.isfinite(pv) or pv < 1e-6:
                            pv = float(np.nansum(V[d, idxs] * wi))
                        new_exp = float(np.clip(vol_target / max(pv, 1e-6), 0.0, lev_cap))
                # ---- timing overlay (signal library + continuous scaling) ----
                if timing_mode != 'none' and not trail and new_exp > 0:
                    scale, tstate = self._timing_scale(p, d, tstate)
                    new_exp *= scale
                turn = np.abs(new_w * new_exp - w * exp).sum()
                turnover_total += turn
                if turn > 1e-6:
                    equity[-1] *= (1.0 - turn * cost)
                    trades += 1
                w, exp = new_w, new_exp
            r = (float(np.nansum(w * self.ret1[t]) * exp + (1.0 - exp) * CASH_DAILY)
                 - max(0.0, exp - 1.0) * fin_annual / TRADING_DAYS)
            equity.append(equity[-1] * (1.0 + r))
            peak = max(peak, equity[-1])
            exposures.append(exp)
        eq = np.asarray(equity)
        dd_idx = self.dates[start - 1:start - 1 + len(eq)]
        return eq, dd_idx, np.asarray(exposures), turnover_total, trades


def metrics(eq, dts, exposures, turn, trades):
    n = len(eq)
    yrs = (dts[-1] - dts[0]).days / 365.25
    total = eq[-1] / eq[0] - 1.0
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1.0
    r = eq[1:] / eq[:-1] - 1.0
    vol = float(np.std(r) * np.sqrt(TRADING_DAYS))
    sharpe = (float(np.mean(r)) * TRADING_DAYS - RF) / vol if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq / peak - 1.0))
    calmar = cagr / abs(mdd) if mdd < -1e-9 else 0.0
    s = pd.Series(eq, index=pd.to_datetime(dts))
    m = s.resample('ME').last().pct_change().dropna()
    mwin = float((m > 0).mean())
    return dict(total=round(total * 100, 2), cagr=round(cagr * 100, 2),
                vol=round(vol * 100, 2), sharpe=round(sharpe, 4),
                maxdd=round(mdd * 100, 2), calmar=round(calmar, 3),
                mwin=round(mwin * 100, 2), trades=int(trades),
                turnover=round(turn, 1), avg_exp=round(float(np.mean(exposures)), 3),
                lev_time=round(float(np.mean(exposures > 1.001)) * 100, 1),
                years=round(yrs, 2))


def yearly(eq, dts):
    s = pd.Series(eq, index=pd.to_datetime(dts))
    out = {}
    for y, g in s.groupby(s.index.year):
        if len(g) > 1:
            out[int(y)] = round((g.iloc[-1] / g.iloc[0] - 1) * 100, 2)
    return out


df = pd.read_csv(DATA, parse_dates=['day']).set_index('day').sort_index()
close = df[ASSETS13].astype(float)
ENGINE = Engine20(close)
dates = ENGINE.dates
T, A = ENGINE.T, ENGINE.A
fb = ENGINE.fb
warmup = ENGINE.warmup

# C_PARAM extends v1.9 default; timing_mode='none' => bit-identical to v1.9
C_PARAM = dict(mom_lb=160, skip=0, vol_win=40, ma_len=0, high_win=120,
               w_mom=0.6567748460792409, w_vol=0.6850954511718137,
               w_accel=0.009366340742048944, w_sharpe=0.35812088191863223,
               w_trend=0.1368760859874372, w_ts=0.0,
               top_n=2, rebal=10, vol_target=0.1, lev_cap=1.0,
               abs_thr=-0.010882335508737578, vol_cap=0.7393585375186374,
               dd_limit=0.13005634764030602, dd_cooldown=40,
               high_dd_limit=0.15, weight_mode='inv_vol', cash_mode='bond',
               softmax_temp=0.15, blend_alpha=0.5,
               # timing library
               timing_mode='none', timing_floor=0.5, timing_k=120,
               timing_aggr=1.0, timing_gamma=1.0, timing_band=0.02,
               vote_trend=1.0, vote_breadth=1.0, vote_tsmom=1.0, vote_vol=1.0,
               vote_enter=0.60, vote_exit=0.40, vote_soft=0.35,
               breadth_thr=0.5)


def run(p, **kw):
    return ENGINE.run(p, **kw)


if __name__ == '__main__':
    t0 = time.time()
    m = metrics(*run(C_PARAM))
    print('BASELINE v2.0', json.dumps(m))
    print('PERIOD', str(dates[warmup - 1].date()), str(dates[-1].date()),
          'warmup', warmup, 'assets', A, 'risk', int(ENGINE.risk.sum()))
