# -*- coding: utf-8 -*-
"""机构版择时工程 · 引擎层（单引擎架构，parity 结构性 = 0）

包装 vendor 的 Engine20：
  - make_engine(): 从配置构建主引擎（冻结 alpha 基线 + 择时 overlay）
  - run_window(): 在指定日期窗口回测，返回净值与指标
  - detrend engine: 去漂移市场，用于诚实性检验
  - 全部路径经同一个 Engine20.run，杜绝"回测/验证/递归"多引擎口径漂移
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing_data import load_close, detrend_close, load_config  # noqa: E402
from aw_timing import research20 as R20  # noqa: E402


@lru_cache(maxsize=1)
def _engines():
    close = load_close()
    dclose = detrend_close(close)
    main = R20.Engine20(close)
    det = R20.Engine20(dclose)
    return close, main, det


def main_engine():
    return _engines()[1]


def detrend_engine():
    return _engines()[2]


def frozen_params() -> dict:
    """冻结的 alpha 基线参数（含 timing_mode=none 起点）。"""
    cfg = load_config()
    p = dict(R20.C_PARAM)
    p.update(cfg["alpha_baseline"]["params"])
    p["lev_cap"] = 1.0
    p.pop("timing", None)
    return p


def lock_params(p: dict) -> dict:
    """锁参：甄别类型、钳制零杠杆、清除 legacy 字段。"""
    out = dict(R20.C_PARAM)
    out.update(dict(p))
    out.pop("timing", None)
    out["lev_cap"] = 1.0
    for k in ("mom_lb", "skip", "vol_win", "ma_len", "high_win", "top_n",
              "rebal", "dd_cooldown", "timing_k"):
        if k in out:
            out[k] = int(out[k])
    for k in ("timing_floor", "timing_aggr", "timing_gamma", "timing_band",
              "vote_trend", "vote_breadth", "vote_tsmom", "vote_vol",
              "vote_enter", "vote_exit", "vote_soft", "breadth_thr",
              "vol_target", "abs_thr", "vol_cap", "dd_limit", "high_dd_limit",
              "lev_cap", "w_ts", "softmax_temp", "blend_alpha"):
        if k in out:
            out[k] = float(out[k])
    for k in ("timing_mode", "weight_mode", "cash_mode"):
        if k in out:
            out[k] = str(out[k])
    if "blend" in out:
        out["blend"] = bool(out["blend"])
    return out


def idx_of(dates, iso: str) -> int:
    """日期 → 引擎整数索引（取 >= iso 的首个）。"""
    return int(np.searchsorted(pd.DatetimeIndex(dates), pd.Timestamp(iso)))


def run_full(p: dict, detrend: bool = False) -> dict:
    eng = detrend_engine() if detrend else main_engine()
    eq, dts, expo, turn, trades = eng.run(lock_params(p))
    m = R20.metrics(eq, dts, expo, turn, trades)
    return {"equity": eq, "dates": dts, "exposures": expo,
            "turnover": turn, "trades": trades, "metrics": m}


def run_window(p: dict, start_iso: str, end_iso: str, detrend: bool = False) -> dict:
    """在 [start, end) 窗口回测（净值起点归一化为 1）。"""
    eng = detrend_engine() if detrend else main_engine()
    s = max(idx_of(eng.dates, start_iso), eng.warmup)
    e = min(idx_of(eng.dates, end_iso), eng.T)
    if e - s < 30:
        return {"metrics": {}, "equity": np.array([1.0]), "dates": eng.dates[s:e]}
    eq, dts, expo, turn, trades = eng.run(lock_params(p), start=s, end=e)
    m = R20.metrics(eq, dts, expo, turn, trades)
    return {"equity": eq, "dates": dts, "exposures": expo,
            "turnover": turn, "trades": trades, "metrics": m}


def window_metrics(p: dict, start_iso: str, end_iso: str, detrend: bool = False) -> dict:
    return run_window(p, start_iso, end_iso, detrend=detrend)["metrics"]


def yearly_returns(p: dict, detrend: bool = False) -> dict:
    r = run_full(p, detrend=detrend)
    return R20.yearly(r["equity"], r["dates"])


if __name__ == "__main__":
    p = frozen_params()
    m = run_full(p)["metrics"]
    md = run_full(p, detrend=True)["metrics"]
    print("冻结 alpha 基线:", {k: m[k] for k in ("cagr", "maxdd", "sharpe", "avg_exp")})
    print("去漂移:", {k: md[k] for k in ("cagr", "maxdd", "sharpe")})
