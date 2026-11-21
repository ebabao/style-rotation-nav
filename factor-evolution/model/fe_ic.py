"""
严格分离标签 IC 检验
=====================
教训来源: v2.1 诊断发现 overnight_20/intraday_20 的 IC+0.95 是机械重叠虚高。
本模块: 因子在 t 日收盘算好, 标签 = t+1 到 t+1+h 的收益 (完全不含 t 及以前的价格路径)。
用 RankIC(时序 Spearman) + 去漂移(减去同期风险资产等权均值) 双重口径。
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RISK = ['CSI300ETF', 'NDX100ETF', 'GoldETF', 'ChiNextETF', 'STAR50ETF']


def forward_ret(panel, h=10, start=1, detrend=True):
    """
    标签矩阵 (T,A): t 日 → 未来 [t+start, t+start+h-1] 的累计收益。
    start=1 表示从次日开始, 与 t 日因子完全错开。
    detrend: 减去当日5风险资产等权均值(去漂移alpha)。
    """
    c = panel.close
    T, A = c.shape
    lab = np.full((T, A), np.nan)
    for t in range(T - start - h + 1):
        lab[t] = c[t + start + h - 1] / c[t + start - 1] - 1.0
    if detrend:
        mkt = np.nanmean(lab[:, :5], axis=1, keepdims=True)
        lab = lab - mkt
    return lab


def rank_ic(F, lab, warmup=300, min_periods=200):
    """
    对每个因子: 逐日横截面 Spearman(因子_t, 标签_t), 再取时序均值/ICIR/胜率。
    只在风险资产(前5列)上算。
    """
    out = {}
    for name, mat in F.items():
        ics = []
        for t in range(warmup, len(lab) - 1):
            x, y = mat[t, :5], lab[t, :5]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 4:
                rho = spearmanr(x[ok], y[ok]).correlation
                if np.isfinite(rho):
                    ics.append(rho)
        ics = np.array(ics)
        if len(ics) < min_periods:
            continue
        out[name] = {
            'ic': float(np.mean(ics)),
            'icir': float(np.mean(ics) / (np.std(ics) + 1e-9)),
            'pos': float(np.mean(ics > 0)),
            'n': len(ics),
        }
    return out


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import fe_factors as FF
    p, dates = FF.load(os.path.join(os.path.dirname(__file__), '..', 'data'))
    F = FF.build_factors(p)
    lab = forward_ret(p, h=10, start=1, detrend=True)
    res = rank_ic(F, lab, warmup=300)
    res_sorted = sorted(res.items(), key=lambda kv: -abs(kv[1]['ic']))
    print('=== 严格分离标签 RankIC (去漂移, h=10, t+1 起) ===')
    print('%-14s %8s %8s %8s' % ('因子', 'IC', 'ICIR', '胜率'))
    for k, v in res_sorted:
        print('%-14s %+8.4f %+8.4f %8.3f' % (k, v['ic'], v['icir'], v['pos']))
