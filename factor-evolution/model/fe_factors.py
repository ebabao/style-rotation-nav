"""
factor-evolution · 独立因子库 (数据层 + 因子计算)
==================================================
站在 style-rotation-nav 工程内，从 zero 起构建，不引用冻结的 v2.1 / quant/research20.py。

因子来源（可查证）:
  - WorldQuant Alpha101  : Kakushadze 2015《101 Formulaic Alphas》— 量价复合
  - Qlib Alpha158        : Microsoft Qlib 默认技术因子集 — K线形态/滚动统计
  - 第29章机构因子        : RSRS(光大)/高阶矩(Harvey-Siddique2000)/FIP(Bravo-Rush)/波动收缩
  - 经典                  : 动量/反转/TSMOM/路径效率(Kaufman)/量价背离

严格无前视：所有 rolling 仅用 t 及以前数据；决策日用 d=t-1 的因子值，承担 t 日收益。
"""
import numpy as np
import pandas as pd

ALL = ['CSI300ETF', 'NDX100ETF', 'GoldETF', 'ChiNextETF', 'STAR50ETF', 'BondETF']
RISK = ['CSI300ETF', 'NDX100ETF', 'GoldETF', 'ChiNextETF', 'STAR50ETF']
BOND = 'BondETF'


def load(data_dir):
    """返回 (Panel, dates)。Panel 含每个资产的 OHLCV (T,) Series。"""
    ohlc = pd.read_csv(f'{data_dir}/etf_ohlc.csv', parse_dates=['day']).set_index('day')
    P = {}
    for a in ALL:
        P[a] = {
            'open': ohlc[f'{a}_open'], 'high': ohlc[f'{a}_high'],
            'low': ohlc[f'{a}_low'], 'close': ohlc[f'{a}_close'],
            'vol': ohlc[f'{a}_vol'],
        }
    return Panel(P), ohlc.index


class Panel:
    def __init__(self, P):
        self.P = P
        self.cols = ALL
        self.T = len(next(iter(P.values()))['close'])
        # 矩阵 (T, A)
        self.close = np.column_stack([P[a]['close'].values for a in ALL])
        self.open = np.column_stack([P[a]['open'].values for a in ALL])
        self.high = np.column_stack([P[a]['high'].values for a in ALL])
        self.low = np.column_stack([P[a]['low'].values for a in ALL])
        self.vol = np.column_stack([P[a]['vol'].values for a in ALL])
        self.ret = np.zeros_like(self.close)
        self.ret[1:] = self.close[1:] / self.close[:-1] - 1.0
        self.ret[0] = np.nan

    def _df(self, x):
        return pd.DataFrame(x)


def _roll(x, w, how):
    """滚动聚合: how ∈ {mean,std,sum,skew,kurt,max,min}。返回 (T,A)。"""
    df = pd.DataFrame(x)
    r = df.rolling(w, min_periods=max(2, w // 2))
    return getattr(r, how)().values


def _rc(a, w, b):
    """a,b 均 (T,A)，逐列滚动相关。返回 (T,A)。"""
    A = pd.DataFrame(a); B = pd.DataFrame(b)
    out = np.full_like(a, np.nan)
    for j in range(a.shape[1]):
        out[:, j] = A.iloc[:, j].rolling(w, min_periods=max(2, w // 2)).corr(B.iloc[:, j]).values
    return out


def build_factors(panel, source=None):
    """
    构建因子字典: name -> (T, A) 矩阵。
    每个因子仅用 t 及以前数据；NaN 保留，由引擎处理。
    """
    src = source or {}
    F = {}
    c, o, h, l, v, r = panel.close, panel.open, panel.high, panel.low, panel.vol, panel.ret
    logv = np.log(v + 1.0)

    # ---------- 1. 动量 / 反转 (Alpha101 #001/#054, 经典) ----------
    for w in (5, 10, 20, 60, 120, 250):
        m = np.full_like(c, np.nan)
        m[w:] = c[w:] / c[:-w] - 1.0
        F[f'mom{w}'] = m
    for w in (1, 3, 5):
        rv = np.full_like(c, np.nan)
        rv[w:] = -(c[w:] / c[:-w] - 1.0)
        F[f'rev{w}'] = rv
    # 时序动量 TSMOM (Baz et al 2015): sign * 波动缩放
    for w in (20, 60, 120):
        s = np.sign(F[f'mom{w}'])
        vol = _roll(r, w, 'std') * np.sqrt(252)
        with np.errstate(divide='ignore', invalid='ignore'):
            F[f'tsmom{w}'] = s / np.maximum(vol, 1e-8)

    # ---------- 2. K线形态 (Alpha158 KBAR) ----------
    with np.errstate(divide='ignore', invalid='ignore'):
        F['kmid'] = (c - o) / o                 # 实体/开盘
        F['klen'] = (h - l) / o                 # 振幅
        F['ksft'] = (2*c - h - l) / o           # 收盘在当日区间位置
        F['kup'] = (np.maximum(o, c) - h) / o   # 上影线(负)
        F['klow'] = (l - np.minimum(o, c)) / o  # 下影线(负)

    # ---------- 3. 均线偏离 / 趋势质量 (Alpha158 MA/RSQR/BETA) ----------
    for w in (5, 10, 20, 60):
        ma = _roll(c, w, 'mean')
        with np.errstate(divide='ignore', invalid='ignore'):
            F[f'ma{w}'] = c / ma - 1.0

    # RSQR 稳健实现（逐资产滚动回归）
    for w in (20, 60):
        rs = np.full_like(c, np.nan)
        beta = np.full_like(c, np.nan)
        idx = np.arange(w, dtype=float); idx = idx - idx.mean(); den = (idx**2).sum()
        for j in range(c.shape[1]):
            ys = pd.DataFrame(c[:, j])
            for t in range(w - 1, len(c)):
                y = c[t-w+1:t+1, j]
                if np.all(np.isfinite(y)):
                    ym = y - y.mean()
                    b = (ym * idx).sum() / den
                    resid = ym - b*idx
                    sst = (ym**2).sum()
                    rs[t, j] = 1 - (resid**2).sum()/sst if sst > 0 else np.nan
                    beta[t, j] = b / (np.mean(y) + 1e-8)
        F[f'rsqr{w}'] = rs
        F[f'beta{w}'] = beta

    # ---------- 4. 波动族 (Alpha158 STD / 第29章波动收缩) ----------
    for w in (20, 60):
        F[f'std{w}'] = _roll(r, w, 'std') * np.sqrt(252)
    v10 = _roll(r, 10, 'std')
    v63 = _roll(r, 63, 'std')
    with np.errstate(divide='ignore', invalid='ignore'):
        F['vol_contraction'] = v10 / (v63 + 1e-9)
        F['down_vol20'] = _roll(np.minimum(r, 0)**2, 20, 'mean') * np.sqrt(252)

    # ---------- 5. 价量关系 (Alpha101 #004/#006/#007, Alpha158 CORR) ----------
    for w in (5, 20):
        F[f'corr{w}'] = _rc(c, w, logv)
    with np.errstate(divide='ignore', invalid='ignore'):
        F['vol_surge'] = v / (_roll(v, 20, 'mean') + 1e-9)   # 量能突增
    F['vstd20'] = _roll(v, 20, 'std') / (v + 1e-9)           # 量能波动

    # ---------- 6. 涨跌结构 (Alpha158 CNTP/SUMP, 第29章 FIP) ----------
    for w in (20, 60):
        F[f'cntp{w}'] = _roll((r > 0).astype(float), w, 'mean')
        up = _roll(np.maximum(r, 0), w, 'sum')
        dn = _roll(np.maximum(-r, 0), w, 'sum')
        with np.errstate(divide='ignore', invalid='ignore'):
            F[f'sumd{w}'] = (up - dn) / (up + dn + 1e-9)   # 类 RSI
    F['fip252'] = _roll((r > 0).astype(float), 252, 'mean')

    # ---------- 7. 高阶矩 (第29章 Harvey-Siddique 2000) ----------
    for w in (20, 60):
        F[f'skew{w}'] = _roll(r, w, 'skew')
        F[f'kurt{w}'] = _roll(r, w, 'kurt')

    # ---------- 8. 路径效率 / 流动性 (Kaufman ER / Amihud) ----------
    for w in (20, 60):
        direction = np.abs(F.get(f'mom{w}', np.full_like(c, np.nan)))
        path = _roll(np.abs(r), w, 'sum')
        with np.errstate(divide='ignore', invalid='ignore'):
            F[f'eff{w}'] = direction / (path + 1e-9)
    with np.errstate(divide='ignore', invalid='ignore'):
        F['amihud20'] = _roll(np.abs(r) / (v + 1e-9), 20, 'mean') * 1e8

    # ---------- 9. 极值位置 (Alpha158 RANK/IMAX, 距高点) ----------
    for w in (20, 60):
        F[f'rank{w}'] = pd.DataFrame(c).rolling(w, min_periods=max(2, w//2)).apply(
            lambda s: s.rank(pct=True).iloc[-1] if np.all(np.isfinite(s)) else np.nan, raw=False).values
        hh = _roll(h, w, 'max')
        with np.errstate(divide='ignore', invalid='ignore'):
            F[f'ddhigh{w}'] = c / hh - 1.0

    # 清洗：inf -> nan
    for k in F:
        F[k] = np.where(np.isfinite(F[k]), F[k], np.nan)
    return F


SOURCE = {
    'mom5':'经典动量','mom10':'经典动量','mom20':'Alpha101#054/动量','mom60':'经典动量',
    'mom120':'经典动量','mom250':'经典动量(年度)','rev1':'短期反转','rev3':'短期反转',
    'rev5':'Alpha101#004/反转','tsmom20':'Baz2015时序动量','tsmom60':'Baz2015时序动量',
    'tsmom120':'Baz2015时序动量','kmid':'Alpha158 KBAR','klen':'Alpha158 KBAR',
    'ksft':'Alpha158 KBAR','kup':'Alpha158 KBAR','klow':'Alpha158 KBAR',
    'ma5':'Alpha158 MA','ma10':'Alpha158 MA','ma20':'Alpha158 MA','ma60':'Alpha158 MA',
    'rsqr20':'Alpha158 RSQR','rsqr60':'Alpha158 RSQR','beta20':'Alpha158 BETA',
    'beta60':'Alpha158 BETA','std20':'Alpha158 STD','std60':'Alpha158 STD',
    'vol_contraction':'第29章/波动收缩','down_vol20':'下行波动','corr5':'Alpha158 CORR',
    'corr20':'Alpha101量价相关','vol_surge':'Alpha101#007量能','vstd20':'Alpha158 VSTD',
    'cntp20':'Alpha158 CNTP','cntp60':'Alpha158 CNTP','sumd20':'Alpha158 SUMD(RSI)',
    'sumd60':'Alpha158 SUMD(RSI)','fip252':'第29章/Frog-in-the-Pan','skew20':'第29章/高阶矩',
    'skew60':'第29章/高阶矩','kurt20':'第29章/高阶矩','kurt60':'第29章/高阶矩',
    'eff20':'Kaufman效率','eff60':'Kaufman效率','amihud20':'Amihud2002流动性',
    'rank20':'Alpha158 RANK','rank60':'Alpha158 RANK','ddhigh20':'Alpha158 距高点',
    'ddhigh60':'Alpha158 距高点',
}
