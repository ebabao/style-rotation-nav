# -*- coding: utf-8 -*-
"""
六指数风格轮动 · 因子库与回测引擎

设计要点（避免前视偏差）：
  - 所有因子在 t 日收盘后可得（t 行只用 <= t 的数据）
  - 决策在 t 日收盘后做出，t+1 日生效
  - 交易成本按权重变动的绝对值之和（换手）收取
  - 现金仓位按年化 1.5% 货币基金近似，标注在报告中
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

ASSETS = ["SSE50", "ChiNext", "CSI300", "CSI500", "CSI1000", "STAR50"]
ASSET_CN = {
    "SSE50": "上证50", "ChiNext": "创业板指", "CSI300": "沪深300",
    "CSI500": "中证500", "CSI1000": "中证1000", "STAR50": "科创50",
    "Bond5Y": "国债ETF", "Bond10Y": "十年国债ETF", "GoldETF": "黄金ETF",
}

MOM_LBS = [10, 20, 30, 40, 60, 90, 120, 160, 200, 250]
SKIPS = [0, 5, 10, 21]
VOL_WINS = [20, 40, 60, 120]
MA_LENS = [60, 120, 200]
HIGH_WINS = [120, 252]

TRADING_DAYS = 252
CASH_ANNUAL = 0.015
CASH_DAILY = CASH_ANNUAL / TRADING_DAYS
COST_RATE = 0.0008          # 单边 8bp（ETF 佣金 + 冲击成本近似）
MIN_PERIODS = 20


def load_data(path: str = "/workspace/quant/data/index_close.csv"):
    """读取行情面板；列由 CSV 决定，支持主方案(6)与增强方案(9)。"""
    df = pd.read_csv(path, parse_dates=["day"]).set_index("day").sort_index()
    return df.astype(float)


class FactorBank:
    """预计算所有候选因子矩阵，供进化算法高速查表使用。"""

    def __init__(self, close: pd.DataFrame):
        self.dates = close.index
        self.close = close.to_numpy(dtype=float)
        self.T, self.A = self.close.shape
        self.assets = list(close.columns)

        px = pd.DataFrame(self.close, columns=self.assets)
        r1 = px / px.shift(1) - 1.0
        self.ret1 = r1.to_numpy(dtype=float)          # (T,A)

        self.mom = {}
        for lb in MOM_LBS:
            for sk in SKIPS:
                # t-(lb+sk) → t-sk 的区间收益；sk>0 即跳过最近 sk 日
                m = (px.shift(sk) / px.shift(lb + sk) - 1.0).to_numpy(dtype=float)
                self.mom[(lb, sk)] = m

        self.vol = {}
        for w in VOL_WINS:
            self.vol[w] = (r1.rolling(w, min_periods=w).std() * np.sqrt(TRADING_DAYS)).to_numpy(dtype=float)

        self.trend = {}
        for ma in MA_LENS:
            self.trend[ma] = (px / px.rolling(ma, min_periods=ma).mean() - 1.0).to_numpy(dtype=float)

        self.rev5 = (px / px.shift(5) - 1.0).to_numpy(dtype=float)

        self.from_high = {}
        for w in HIGH_WINS:
            self.from_high[w] = (px / px.rolling(w, min_periods=MIN_PERIODS).max() - 1.0).to_numpy(dtype=float)

        self.warmup = max(max(MOM_LBS) + max(SKIPS), max(MA_LENS), max(HIGH_WINS)) + 5

    def zscore(self, row: np.ndarray) -> np.ndarray:
        """截面标准化，全 NaN 或零方差时返回零向量。"""
        m = np.nanmean(row)
        s = np.nanstd(row)
        out = (row - m) / (s if s > 1e-9 else 1.0)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class BTResult:
    equity: np.ndarray
    dates: pd.DatetimeIndex
    exposure: np.ndarray
    holdings: list
    turnover: float
    metrics: dict


def _metrics(equity: np.ndarray, dates, exposure, turnover, trades) -> dict:
    eq = np.asarray(equity, dtype=float)
    rets = eq[1:] / eq[:-1] - 1.0
    years = len(rets) / TRADING_DAYS
    total = eq[-1] / eq[0] - 1.0
    cagr = (eq[-1] / eq[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    vol = np.std(rets, ddof=1) * np.sqrt(TRADING_DAYS) if len(rets) > 2 else 0.0
    sharpe = (np.mean(rets) * TRADING_DAYS - CASH_ANNUAL) / vol if vol > 1e-9 else 0.0
    downside = np.std(rets[rets < 0], ddof=1) * np.sqrt(TRADING_DAYS) if (rets < 0).sum() > 2 else 0.0
    sortino = (np.mean(rets) * TRADING_DAYS - CASH_ANNUAL) / downside if downside > 1e-9 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    maxdd = float(max(0.0, -dd.min()))
    calmar = cagr / maxdd if maxdd > 1e-9 else 0.0
    # 胜率口径：一律用「相对现金的超额」衡量，避免空仓吃利息造成虚假高胜率
    excess = rets - CASH_DAILY
    day_win = float((excess > 0).mean())
    active = np.asarray(exposure)[:len(rets)] > 1e-6
    active_win = float((rets[active] > 0).mean()) if active.sum() > 10 else 0.0

    # 月度胜率同样按超额计算
    s = pd.Series(eq, index=dates)
    mret = s.resample("ME").last().pct_change().dropna()
    month_win = float((mret > CASH_ANNUAL / 12).mean()) if len(mret) else 0.0

    return dict(
        total_return=float(total), cagr=float(cagr), vol=float(vol),
        sharpe=float(sharpe), sortino=float(sortino), maxdd=float(maxdd),
        calmar=float(calmar), day_win=day_win, day_win_active=active_win,
        month_win=month_win,
        turnover=float(turnover), trades=int(trades),
        avg_exposure=float(np.mean(exposure)),
        final_equity=float(eq[-1]),
    )


def backtest(p: dict, fb: FactorBank, start: int | None = None,
             end: int | None = None, initial: float = 1.0) -> BTResult:
    """在 [start, end) 区间回测。start/end 为整数索引。"""
    start = fb.warmup if start is None else max(start, fb.warmup)
    end = fb.T if end is None else min(end, fb.T)
    if end - start < 60:
        return BTResult(np.array([initial]), fb.dates[start:end], np.array([0.0]), [], 0.0, {})

    mom_key = (int(p["mom_lb"]), int(p["skip"]))
    M = fb.mom[mom_key]
    V = fb.vol[int(p["vol_win"])]
    TR = fb.trend[int(p["ma_len"])] if p["ma_len"] else np.zeros_like(V)
    REV = fb.rev5
    FH = fb.from_high[int(p["high_win"])]
    R = fb.ret1

    w = np.zeros(fb.A)
    exp = 0.0
    equity = [initial]
    exposures = []
    holdings_log = []
    turnover_total = 0.0
    trades = 0
    circuit_until = -1
    peak = initial
    rebal = int(p["rebal"])
    vol_target = float(p["vol_target"])
    abs_thr = float(p["abs_thr"])
    vol_cap = float(p["vol_cap"])
    dd_limit = float(p["dd_limit"])
    dd_cooldown = int(p["dd_cooldown"])
    top_n = int(p["top_n"])
    wm = np.array([p["w_mom"], p["w_vol"], p["w_trend"], p["w_rev"]], dtype=float)
    wm = wm / (np.abs(wm).sum() + 1e-9)

    for t in range(start, end):
        # ---- 调仓日：用 t-1 收盘可得的信息决策，t 日生效 ----
        if (t - start) % rebal == 0:
            d = t - 1
            score = (wm[0] * fb.zscore(M[d])
                     + wm[1] * (-fb.zscore(V[d]))
                     + wm[2] * fb.zscore(TR[d])
                     + wm[3] * (-fb.zscore(REV[d])))

            cur_eq = equity[-1]
            dd = cur_eq / peak - 1.0
            if dd < -dd_limit:
                circuit_until = t + dd_cooldown

            eligible = np.ones(fb.A, dtype=bool)
            eligible &= np.isfinite(M[d]) & np.isfinite(V[d])
            eligible &= (M[d] > abs_thr)                       # 绝对动量过滤
            if p["ma_len"]:
                eligible &= np.isfinite(TR[d]) & (TR[d] > 0)  # 趋势过滤
            eligible &= (V[d] < vol_cap) & (V[d] > 0)          # 波动率上限
            eligible &= (FH[d] > -float(p["high_dd_limit"]))  # 距高点过滤

            if t <= circuit_until or eligible.sum() == 0:
                new_w = np.zeros(fb.A)
                new_exp = 0.0
            else:
                idx = np.argsort(-np.where(eligible, score, -np.inf))[:top_n]
                idx = idx[eligible[idx]]
                if len(idx) == 0:
                    new_w = np.zeros(fb.A); new_exp = 0.0
                else:
                    if p["weight_mode"] == "inv_vol":
                        iv = 1.0 / np.maximum(V[d, idx], 0.02)
                        wi = iv / iv.sum()
                    else:
                        wi = np.ones(len(idx)) / len(idx)
                    new_w = np.zeros(fb.A); new_w[idx] = wi

                    # 波动率靶向：用近 vol_win 日协方差估计组合波动，只允许降杠杆
                    lo = max(0, d - int(p["vol_win"]) + 1)
                    cov = np.cov(R[lo:d + 1, idx].T) if len(idx) > 1 else np.array([[V[d, idx[0]] ** 2 / TRADING_DAYS]])
                    cov = np.atleast_2d(np.nan_to_num(cov, nan=0.0))
                    pv = float(np.sqrt(max(wi @ cov @ wi, 1e-12) * TRADING_DAYS))
                    new_exp = float(np.clip(vol_target / max(pv, 1e-6), 0.0, 1.0))

            turn = np.abs(new_w * new_exp - w * exp).sum()
            turnover_total += turn
            if turn > 1e-6:
                equity[-1] *= (1.0 - turn * COST_RATE)
                trades += 1
            w, exp = new_w, new_exp
            holdings_log.append((str(fb.dates[t].date()),
                                 [fb.assets[i] for i in np.flatnonzero(w > 0)],
                                 round(exp, 3)))

        # ---- 当日收益 ----
        r = float(np.nansum(w * R[t]) * exp + (1.0 - exp) * CASH_DAILY)
        equity.append(equity[-1] * (1.0 + r))
        peak = max(peak, equity[-1])
        exposures.append(exp)

    eq = np.asarray(equity)
    dates = fb.dates[start:end]
    if len(eq) != len(dates):
        dates = fb.dates[start - 1:end]
    met = _metrics(eq, dates, np.asarray(exposures), turnover_total, trades)
    return BTResult(eq, dates, np.asarray(exposures), holdings_log, turnover_total, met)


DEFAULT_PARAMS = dict(
    mom_lb=60, skip=5, vol_win=40, ma_len=120, high_win=252,
    w_mom=0.45, w_vol=0.15, w_trend=0.30, w_rev=0.10,
    top_n=2, rebal=10, vol_target=0.12, abs_thr=0.0, vol_cap=0.45,
    dd_limit=0.12, dd_cooldown=10, high_dd_limit=0.30,
    weight_mode="inv_vol",
)
