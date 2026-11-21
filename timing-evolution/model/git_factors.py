# -*- coding: utf-8 -*-
"""第29章 · 从 GitHub/论文学习的市场级择时/预警因子库（零前视，逐条标注来源）

来源可查证（文档第29.1节）：
  RSRS 阻力支撑         github.com/TenghanZhong/RSRS；光大证券《基于阻力支撑相对强度的市场择时》
  高阶矩择时            Harvey & Siddique (2000)；buckstrdr/.../Higher_Moment_Timing.md
  Diffusion 扩散宽度    buckstrdr/.../Diffusion_Indicator.md
  时变夏普 T-Sharpe     Whitelaw (1994/97)；国信；buckstrdr/.../Time_Varying_Sharpe.md
  Frog-in-the-Pan       github.com/BravoRush/Momentum-Investing（趋势平滑度/自满度）
  波动收缩+趋势持续     github.com/yingwang/trade（10d/63d波动比、21d自相关）
  动量加速度 m5_z       文档28节已有 stress 对照

数据：本工程的 etf_ohlc.csv 含真实 high/low（优于文档"全收益指数需ETF合成high/low"），
      5 个风险资产等权合成市场级 high/low/close，全部因子仅用 t 日及之前数据。
零前视：所有 rolling 只向后看；因子值 f[t] 在 t 日收盘后可得，预测 t+1 起的未来收益。
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from timing_data import load_config, load_close, OUT_DATA  # noqa: E402

CFG = load_config()
RISK = CFG["universe"]["RISK"]

SOURCE = {
    "rsrs_slope": "RSRS / 光大证券·TenghanZhong/RSRS",
    "rsrs_adj": "RSRS z-score×R² / 光大证券",
    "skew20": "高阶矩偏度 / Harvey&Siddique2000",
    "kurt20": "高阶矩超额峰度 / Harvey&Siddique2000",
    "skew60": "高阶矩偏度(60d) / Harvey&Siddique2000",
    "kurt60": "高阶矩超额峰度(60d) / Harvey&Siddique2000",
    "m5_z": "5阶矩z-score / Harvey&Siddique2000",
    "diffusion20": "Diffusion站上多均线占比 / buckstrdr",
    "diffusion_div": "指数涨/宽度降背离 / buckstrdr",
    "tsharpe": "时变夏普 / Whitelaw1994·国信",
    "fip252": "Frog-in-the-Pan正收益天数占比 / BravoRush",
    "vol_contraction": "10d/63d波动比 / yingwang/trade",
    "vol_trend": "波动收缩+趋势持续 / yingwang/trade",
    "autocorr21": "21d自相关 / yingwang/trade",
}


def load_market_panel():
    """加载 5 风险资产的 OHLC，等权合成市场级 high/low/close + 面板。

    返回 dict: close/high/low(市场级 Series, DatetimeIndex), asset_close(DataFrame)
    """
    ohlc_path = os.path.join(os.path.dirname(_HERE), "data", "etf_ohlc.csv")
    ohlc_path = os.path.normpath(ohlc_path)
    p = pd.read_csv(ohlc_path, parse_dates=["day"]).set_index("day").sort_index()
    # 与引擎主数据严格对齐日历（close snapshot 的日期）
    snap = load_close()
    p = p.reindex(snap.index)
    ac = p[[f"{a}_close" for a in RISK]].astype(float)
    ac.columns = RISK
    hi = p[[f"{a}_high" for a in RISK]].astype(float)
    hi.columns = RISK
    lo = p[[f"{a}_low" for a in RISK]].astype(float)
    lo.columns = RISK
    # 等权归一化后合成市场级 high/low（各资产归一到首值，消除量纲）
    nrm = ac / ac.iloc[0]
    mkt_close = nrm.mean(axis=1)
    mkt_high = (hi / ac.iloc[0]).mean(axis=1)   # high 相对各自首值
    mkt_low = (lo / ac.iloc[0]).mean(axis=1)
    return dict(close=mkt_close, high=mkt_high, low=mkt_low,
                asset_close=ac, asset_high=hi, asset_low=lo)


def build_factors(panel=None) -> pd.DataFrame:
    """构造全部第29章因子，返回 DataFrame（DatetimeIndex，列=因子名，全部零前视）。"""
    if panel is None:
        panel = load_market_panel()
    close, high, low = panel["close"], panel["high"], panel["low"]
    ac = panel["asset_close"]
    n = len(close)
    F = pd.DataFrame(index=close.index)

    # ---- 1. RSRS 阻力支撑（High = α + β·Low 回归斜率 → z-score → ×R²）----
    win = 18
    beta = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    h = high.to_numpy(float)
    l = low.to_numpy(float)
    for t in range(win, n):
        x = l[t - win + 1:t + 1]
        y = h[t - win + 1:t + 1]
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        xm, ym = x.mean(), y.mean()
        sxx = ((x - xm) ** 2).sum()
        if sxx <= 0:
            continue
        b = ((x - xm) * (y - ym)).sum() / sxx
        a = ym - b * xm
        yy = y - (a + b * x)
        sst = ((y - ym) ** 2).sum()
        beta[t] = b
        r2[t] = 1 - (yy ** 2).sum() / sst if sst > 0 else 0.0
    F["rsrs_slope"] = beta
    def _zlast(v):
        v = v[np.isfinite(v)]
        return (v[-1] - v.mean()) / (v.std() + 1e-9) if len(v) > 1 else np.nan
    bz = pd.Series(beta).rolling(252, min_periods=60).apply(_zlast, raw=True)
    F["rsrs_adj"] = bz.to_numpy() * r2

    # ---- 2. 高阶矩（偏度/超额峰度/5阶矩z）----
    ret = close.pct_change()
    for win2, tag in ((20, "20"), (60, "60")):
        F[f"skew{tag}"] = ret.rolling(win2, min_periods=win2).skew().to_numpy()
        F[f"kurt{tag}"] = ret.rolling(win2, min_periods=win2).kurt().to_numpy()
    m5 = ret.rolling(40, min_periods=40).apply(
        lambda v: np.mean(((v - np.mean(v)) / (np.std(v) + 1e-9)) ** 5), raw=True)
    F["m5_z"] = m5.rolling(252, min_periods=60).apply(_zlast, raw=True).to_numpy()

    # ---- 3. Diffusion 扩散宽度（站上多均线资产占比 + 背离）----
    mas = [20, 60, 120]
    above = pd.DataFrame(index=ac.index)
    for m in mas:
        ma = ac.rolling(m, min_periods=m).mean()
        above[f"m{m}"] = (ac > ma).mean(axis=1)
    F["diffusion20"] = above.mean(axis=1).to_numpy()
    mkt_up = (close / close.shift(20) - 1).to_numpy()
    dif = F["diffusion20"].to_numpy()
    F["diffusion_div"] = np.where(np.isfinite(mkt_up) & np.isfinite(dif),
                                  mkt_up - pd.Series(dif).rolling(5).mean().to_numpy(),
                                  np.nan)

    # ---- 4. 时变夏普 T-Sharpe（滚动 risk-adjusted 恶化预警）----
    rv = ret.rolling(60, min_periods=60).mean() / (ret.rolling(60, min_periods=60).std() + 1e-9) * np.sqrt(252)
    F["tsharpe"] = rv.to_numpy()

    # ---- 5. Frog-in-the-Pan（252日正收益天数占比=趋势平滑度/自满度）----
    pos = (ret > 0).astype(float)
    F["fip252"] = pos.rolling(252, min_periods=252).mean().to_numpy()

    # ---- 6. 波动收缩+趋势持续（10d/63d波动比、21d自相关）----
    v10 = ret.rolling(10, min_periods=10).std()
    v63 = ret.rolling(63, min_periods=63).std()
    F["vol_contraction"] = (v10 / (v63 + 1e-9)).to_numpy()
    trend = np.where(close > close.shift(63), 1.0, -1.0)
    F["vol_trend"] = (F["vol_contraction"].to_numpy() * trend)
    def _ac1(v):
        v = v[np.isfinite(v)]
        if len(v) < 3 or v.std() == 0:
            return np.nan
        return float(np.corrcoef(v[:-1], v[1:])[0, 1])
    F["autocorr21"] = ret.rolling(21, min_periods=21).apply(_ac1, raw=True).to_numpy()

    return F.replace([np.inf, -np.inf], np.nan)


def forward_drop_label(close=None, h=10, thr=-0.08) -> pd.Series:
    """标签：未来 h 日市场级组合收益 ≤ thr（-8%）。仅评估用，shift 保证 t 时刻标签来自未来。"""
    if close is None:
        close = load_market_panel()["close"]
    fwd = close.shift(-h) / close - 1.0
    lab = (fwd <= thr).astype(float)
    lab[fwd.isna()] = np.nan
    return lab


if __name__ == "__main__":
    F = build_factors()
    lab = forward_drop_label()
    print(f"因子数 {F.shape[1]}，样本 {len(F)}，区间 {F.index[0].date()}~{F.index[-1].date()}")
    print("列:", list(F.columns))
    print(f"事件数(未来10日≤-8%) {int(lab.sum())} / {int(lab.notna().sum())} "
          f"({lab.sum()/lab.notna().sum():.1%})")
    print("\n各因子非空率:")
    for c in F.columns:
        print(f"  {c:16s} {F[c].notna().mean():.1%}  来源 {SOURCE.get(c,'')}")
