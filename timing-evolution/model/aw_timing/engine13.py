# -*- coding: utf-8 -*-
"""
1.3 版引擎 · 大类资产 ETF 轮动（学习「藏富于基·择时君」重制版）

与 1.2 的差别，全部来自那篇文章明确交代的三条经验：
  A. 因子要少。作者把因子砍到老策略的 1/3 才稳住实盘 → 这里只用 5 个：
     动量 / 低波 / Γ加速度 / 风险调整动量 / 趋势，不再叠加残差Alpha、路径质量等。
  B. 标的要少且低相关。池子从 9 个缩到 6 个（文章 5 个 + 国债ETF 作择时避险位）：
     沪深300、纳指100、黄金、创业板、科创50、国债 —— 覆盖 A股/美股/贵金属/债券。
  C. 右侧动量 + 择时。只在趋势向上时持仓，趋势不在就退到国债或现金，
     绝不逆势加仓，这就是作者说的「不陷熊市」。

数据纪律（比文章更严）：
  - 上市前留 NaN，不伪造回填。科创50ETF 2020-11-16 才有，之前自动排除，
    所以本文无法复现「2013.7 起 5 个 ETF」的说法 —— 那 5 只里有 1 只当时不存在。
  - 纳指100ETF 2022-01-14 的 1:5 份额拆分已做前复权，否则会制造 -80% 假崩盘。
  - t-1 收盘决策、t 日生效；单边千一（与文章一致）；现金 1.5%/年。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from engine import _metrics, BTResult, TRADING_DAYS, CASH_ANNUAL, CASH_DAILY, MIN_PERIODS

ASSETS13 = ["CSI300ETF", "NDX100ETF", "GoldETF", "ChiNextETF", "STAR50ETF", "BondETF"]
ASSET_CN13 = {
    "CSI300ETF": "沪深300ETF", "NDX100ETF": "纳指100ETF", "GoldETF": "黄金ETF",
    "ChiNextETF": "创业板ETF", "STAR50ETF": "科创50ETF", "BondETF": "国债ETF",
}
RISK_ASSETS = ["CSI300ETF", "NDX100ETF", "GoldETF", "ChiNextETF", "STAR50ETF"]

MOM_LBS13 = [10, 20, 40, 60, 90, 120, 160]
SKIPS13 = [0, 5, 10, 21]
VOL_WINS13 = [20, 40, 60, 120]
MA_LENS13 = [60, 120]
HIGH_WINS13 = [120, 252]

COST13 = 0.0010          # 单边千一（文章口径）
FIN_SPREAD_ANNUAL = 0.030
FIN_SPREAD_DAILY = FIN_SPREAD_ANNUAL / TRADING_DAYS


def load_data13(path: str = "/workspace/quant/data/etf_close_13.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["day"]).set_index("day").sort_index()
    return df[ASSETS13].astype(float)


class FactorBank13:
    """只保留 5 个因子的矩阵，短回看窗口，压低自由度。"""

    def __init__(self, close: pd.DataFrame):
        self.dates = close.index
        self.close = close.to_numpy(float)
        self.T, self.A = self.close.shape
        self.assets = list(close.columns)

        px = pd.DataFrame(self.close, columns=self.assets)
        r1 = px / px.shift(1) - 1.0
        self.ret1 = r1.to_numpy(float)

        self.mom = {}
        for lb in MOM_LBS13:
            for sk in SKIPS13:
                self.mom[(lb, sk)] = (px.shift(sk) / px.shift(lb + sk) - 1.0).to_numpy(float)

        self.vol = {w: (r1.rolling(w, min_periods=w).std() * np.sqrt(TRADING_DAYS)).to_numpy(float)
                    for w in VOL_WINS13}

        self.trend = {ma: (px / px.rolling(ma, min_periods=ma).mean() - 1.0).to_numpy(float)
                      for ma in MA_LENS13}

        self.from_high = {w: (px / px.rolling(w, min_periods=MIN_PERIODS).max() - 1.0).to_numpy(float)
                          for w in HIGH_WINS13}

        # Γ 动量加速度：近半段 − 前半段（Sornette 2020）
        self.accel = {}
        # 风险调整动量：区间收益 / (日波动 × √lb)
        self.sharpe_mom = {}
        for lb in MOM_LBS13:
            half = max(5, lb // 2)
            self.accel[lb] = (px / px.shift(half) - px.shift(half) / px.shift(lb)).to_numpy(float)
            dv = r1.rolling(lb, min_periods=max(10, lb // 3)).std()
            with np.errstate(invalid="ignore", divide="ignore"):
                sm = (px / px.shift(lb) - 1.0) / (dv * np.sqrt(lb))
            self.sharpe_mom[lb] = sm.to_numpy(float)

        self.warmup = max(max(MOM_LBS13) + max(SKIPS13), max(MA_LENS13)) + 5
        self.bond_col = self.assets.index("BondETF") if "BondETF" in self.assets else -1

    def zscore(self, row: np.ndarray) -> np.ndarray:
        m, s = np.nanmean(row), np.nanstd(row)
        out = (row - m) / (s if s > 1e-9 else 1.0)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


SPACE13 = {
    "mom_lb":       ("choice", MOM_LBS13),
    "skip":         ("choice", SKIPS13),
    "vol_win":      ("choice", VOL_WINS13),
    "ma_len":       ("choice", [0] + MA_LENS13),
    "high_win":     ("choice", HIGH_WINS13),
    "w_mom":        ("float", 0.0, 1.0),
    "w_vol":        ("float", 0.0, 1.0),
    "w_accel":      ("float", 0.0, 1.0),
    "w_sharpe":     ("float", 0.0, 1.0),
    "w_trend":      ("float", 0.0, 1.0),
    "top_n":        ("int", 1, 3),
    "rebal":        ("choice", [5, 10, 15, 20]),
    "vol_target":   ("float", 0.10, 0.45),
    "lev_cap":      ("float", 1.0, 1.5),
    "abs_thr":      ("float", -0.05, 0.10),
    "vol_cap":      ("float", 0.30, 0.95),
    "dd_limit":     ("float", 0.08, 0.25),
    "dd_cooldown":  ("choice", [5, 10, 20, 40]),
    "high_dd_limit": ("float", 0.15, 0.60),
    "weight_mode":  ("choice", ["inv_vol", "equal", "score"]),
    "cash_mode":    ("choice", ["bond", "cash"]),
}
KEYS13 = list(SPACE13)

DEFAULT_PARAMS13 = dict(
    mom_lb=60, skip=0, vol_win=40, ma_len=120, high_win=252,
    w_mom=1.0, w_vol=0.0, w_accel=0.0, w_sharpe=0.0, w_trend=0.0,
    top_n=1, rebal=10, vol_target=0.20, lev_cap=1.0,
    abs_thr=0.0, vol_cap=0.60, dd_limit=0.15, dd_cooldown=10,
    high_dd_limit=0.35, weight_mode="equal", cash_mode="bond",
)


def backtest13(p: dict, fb: FactorBank13, start: int | None = None,
               end: int | None = None, initial: float = 1.0) -> BTResult:
    start = fb.warmup if start is None else max(start, fb.warmup)
    end = fb.T if end is None else min(end, fb.T)
    if end - start < 60:
        return BTResult(np.array([initial]), fb.dates[start:end], np.array([0.0]), [], 0.0, {})

    mom_lb, skip = int(p["mom_lb"]), int(p["skip"])
    vol_win, high_win = int(p["vol_win"]), int(p["high_win"])
    ma_len = int(p["ma_len"])
    M = fb.mom[(mom_lb, skip)]
    V = fb.vol[vol_win]
    TR = fb.trend[ma_len] if ma_len else None
    FH = fb.from_high[high_win]
    AC, SM = fb.accel[mom_lb], fb.sharpe_mom[mom_lb]
    R = fb.ret1
    z = fb.zscore

    wv = np.array([p["w_mom"], p["w_vol"], p["w_accel"], p["w_sharpe"], p["w_trend"]], float)
    wv = wv / (np.abs(wv).sum() + 1e-9)

    top_n, rebal = int(p["top_n"]), int(p["rebal"])
    vol_target, lev_cap = float(p["vol_target"]), float(p["lev_cap"])
    abs_thr, vol_cap = float(p["abs_thr"]), float(p["vol_cap"])
    dd_limit, dd_cd = float(p["dd_limit"]), int(p["dd_cooldown"])
    high_dd = float(p["high_dd_limit"])
    wmode, cash_mode = p["weight_mode"], p["cash_mode"]

    risk_mask = np.array([a in RISK_ASSETS for a in fb.assets])

    w = np.zeros(fb.A)
    exp = 0.0
    equity = [initial]
    exposures, holdings_log = [], []
    turnover_total, trades = 0.0, 0
    circuit_until, peak = -1, initial
    bond = fb.bond_col

    for t in range(start, end):
        if (t - start) % rebal == 0:
            d = t - 1
            score = (wv[0] * z(M[d]) - wv[1] * z(V[d])
                     + wv[2] * z(AC[d]) + wv[3] * z(SM[d])
                     + (wv[4] * z(TR[d]) if TR is not None else 0.0))

            dd = equity[-1] / peak - 1.0
            if dd < -dd_limit:
                circuit_until = t + dd_cd

            eligible = np.isfinite(M[d]) & np.isfinite(V[d]) & risk_mask
            eligible &= (M[d] > abs_thr) & (V[d] < vol_cap) & (V[d] > 0)
            eligible &= (FH[d] > -high_dd)
            if TR is not None:
                eligible &= np.isfinite(TR[d]) & (TR[d] > 0)

            trail = (t <= circuit_until) or (not eligible.any())
            if trail:
                if cash_mode == "bond" and bond >= 0 and np.isfinite(fb.close[d, bond]):
                    new_w = np.zeros(fb.A); new_w[bond] = 1.0
                    new_exp = 1.0
                else:
                    new_w, new_exp = np.zeros(fb.A), 0.0
            else:
                idx = np.argsort(-np.where(eligible, score, -np.inf))[:top_n]
                idx = idx[eligible[idx]]
                if len(idx) == 0:
                    new_w, new_exp = np.zeros(fb.A), 0.0
                else:
                    if wmode == "inv_vol":
                        iv = 1.0 / np.maximum(V[d, idx], 0.02); wi = iv / iv.sum()
                    elif wmode == "score":
                        s = np.clip(score[idx] - np.nanmin(score[idx]) + 0.02, 0.02, None)
                        wi = s / s.sum()
                    else:
                        wi = np.ones(len(idx)) / len(idx)
                    new_w = np.zeros(fb.A); new_w[idx] = wi

                    lo = max(0, d - vol_win + 1)
                    pr = np.nansum(R[lo:d + 1][:, idx] * wi, axis=1)
                    pv = float(np.nanstd(pr) * np.sqrt(TRADING_DAYS))
                    if not np.isfinite(pv) or pv < 1e-6:
                        pv = float(np.nansum(V[d, idx] * wi))
                    new_exp = float(np.clip(vol_target / max(pv, 1e-6), 0.0, lev_cap))

            turn = np.abs(new_w * new_exp - w * exp).sum()
            turnover_total += turn
            if turn > 1e-6:
                equity[-1] *= (1.0 - turn * COST13)
                trades += 1
            w, exp = new_w, new_exp
            holdings_log.append((str(fb.dates[t].date()),
                                 [fb.assets[i] for i in np.flatnonzero(w > 1e-9)],
                                 round(exp, 3)))

        r = float(np.nansum(w * R[t]) * exp + (1.0 - exp) * CASH_DAILY
                  - max(0.0, exp - 1.0) * FIN_SPREAD_DAILY)
        equity.append(equity[-1] * (1.0 + r))
        peak = max(peak, equity[-1])
        exposures.append(exp)

    eq = np.asarray(equity)
    dates = fb.dates[start:end]
    if len(eq) != len(dates):
        dates = fb.dates[start - 1:end]
    met = _metrics(eq, dates, np.asarray(exposures), turnover_total, trades)
    return BTResult(eq, dates, np.asarray(exposures), holdings_log, turnover_total, met)
