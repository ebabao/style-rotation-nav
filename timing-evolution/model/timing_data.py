# -*- coding: utf-8 -*-
"""机构版择时工程 · 数据层（对齐 sr-nav/all-weather/data14.py 范式）

职责：
  1) 配置驱动加载（yaml，日期统一规范化为 ISO 字符串）
  2) 面板加载（前复权收盘价 + OHLCV），严格自包含于子工程 data/
  3) 数据审计哈希（文件 sha256 + 面板指纹 frame_hash），防数据漂移
  4) 去漂移面板（detrend_close），用于诚实性检验（真 alpha vs beta 漂移）

数据纪律：不联网、不编造、不拼接；全部落盘子工程 data/，可离线复现。
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent          # timing-evolution/
MODEL = ROOT / "model"
DATA = ROOT / "data"
OUT = MODEL / "timing_out"
OUT_DATA = OUT / "data"
CONFIG = MODEL / "timing_config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def norm(x):
        if hasattr(x, "isoformat"):
            return x.isoformat()
        if isinstance(x, dict):
            return {k: norm(v) for k, v in x.items()}
        if isinstance(x, list):
            return [norm(v) for v in x]
        return x

    return norm(cfg)


def dump_json(obj, path: Path) -> None:
    def conv(x):
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if hasattr(x, "isoformat"):
            return x.isoformat()
        return str(x)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv) + "\n",
                    encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=True).encode("utf-8") + repr(list(df.columns)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_close() -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_csv(DATA / cfg["data"]["close_panel"], parse_dates=["day"])
    df = df.set_index("day").sort_index()
    assets = cfg["universe"]["ASSETS"]
    return df[assets].astype(float)


def load_ohlc() -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_csv(DATA / cfg["data"]["ohlc_panel"], parse_dates=["day"])
    return df.set_index("day").sort_index()


def detrend_close(close: pd.DataFrame) -> pd.DataFrame:
    """剔除每个资产的长期漂移（对数收益均值），保留波动与路径结构。

    与 quant/panel_run14.detrend_close 逐行一致，用于检验收益是否只是 beta 漂移。
    """
    px = close.to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        logr = np.diff(np.log(np.where(np.isfinite(px) & (px > 0), px, np.nan)), axis=0)
    mu = np.nanmean(logr, axis=0)
    out = np.full_like(px, np.nan)
    out[0] = np.where(np.isfinite(px[0]), 1.0, np.nan)
    for t in range(1, len(px)):
        for j in range(px.shape[1]):
            if not np.isfinite(px[t, j]):
                continue
            if np.isfinite(out[t - 1, j]) and np.isfinite(logr[t - 1, j]):
                out[t, j] = out[t - 1, j] * np.exp(logr[t - 1, j] - mu[j])
            else:
                out[t, j] = 1.0
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def audit_data() -> dict:
    """数据审计：文件哈希 + 面板指纹 + 区间/缺失统计，落盘防漂移。"""
    cfg = load_config()
    close = load_close()
    ohlc = load_ohlc()
    risk = cfg["universe"]["RISK"]
    audit = {
        "version": cfg["version"],
        "files": {
            cfg["data"]["close_panel"]: {
                "sha256": sha256(DATA / cfg["data"]["close_panel"]),
                "rows": int(len(close)),
                "frame_hash": frame_hash(close),
            },
            cfg["data"]["ohlc_panel"]: {
                "sha256": sha256(DATA / cfg["data"]["ohlc_panel"]),
                "rows": int(len(ohlc)),
                "frame_hash": frame_hash(ohlc[["day"]] if "day" in ohlc else ohlc.iloc[:, :1]),
            },
        },
        "close_panel": {
            "start": str(close.index.min().date()),
            "end": str(close.index.max().date()),
            "n_rows": int(len(close)),
            "assets": list(close.columns),
            "nan_counts": {c: int(close[c].isna().sum()) for c in close.columns},
        },
        "risk_assets": risk,
        "cost_bps": cfg["data"]["cost_bps"],
        "fin_annual": cfg["data"]["fin_annual"],
        "note": "全部数据随子工程发布，离线可复现；哈希用于检测数据被外部改写。",
    }
    # OHLC 指纹单独按全列计算（避免上面截断）
    audit["files"][cfg["data"]["ohlc_panel"]]["frame_hash"] = frame_hash(ohlc)
    dump_json(audit, OUT_DATA / "data_audit.json")
    return audit


if __name__ == "__main__":
    a = audit_data()
    print("数据审计完成：", a["close_panel"]["start"], "→", a["close_panel"]["end"],
          a["close_panel"]["n_rows"], "行")
    for f, m in a["files"].items():
        print(" ", f, "sha256", m["sha256"][:16], "rows", m["rows"])
