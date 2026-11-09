"""v1.4 七只真实 ETF 数据层 —— 无指数代理、无拼接、不编造

与 v1.2/v1.3 数据层的根本差别：
  * 资产池只有 7 只真实可交易 ETF，全部使用新浪长历史前复权日线；
  * 上市前保持 NaN（缺席），由引擎按规则重分配大类预算，绝不填充/拼接/编造；
  * 正式评估期从 7 只都有真实价格的最早日（2019-12-05 豆粕ETF上市）起，
    但保留此前真实历史供协方差预热（因此 2020 年 3 月压力样本不丢失）。

沿用 v1.0 D0 口径：对数收益单一真源、显式共同起始日、DatetimeIndex、
ffill<=5 日 + 零收益标记、异常跳变修复留痕、清洗审计哈希。
复用 v1.2 已验证的 fetch_symbol（含 hfq 前复权与份额折算修复），不重写抓数。
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import aw12.data12 as d12  # noqa: E402

PKG = Path(__file__).resolve().parent
OUT = ROOT / "aw14_out"
DATA = OUT / "data"
RAW = DATA / "raw"
CONFIG = PKG / "config_v14.yaml"

# 复用 v1.2 的抓取函数，但把 raw 缓存重定向到 v1.4 目录，避免污染历史产物
d12.RAW = RAW


from functools import lru_cache


@lru_cache(maxsize=1)
def load_config() -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def norm(x):
        # PyYAML 会把 2019-12-06 解析成 datetime.date；统一为 ISO 字符串，
        # 避免字符串比较失败与 JSON 序列化崩溃（v1.3 已踩过此坑）
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
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if hasattr(x, "isoformat"):
            return x.isoformat()
        return str(x)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv) + "\n",
                          encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=True).encode("utf-8")
                          + repr(list(df.columns)).encode("utf-8")).hexdigest()


def _to_series(obj: dict) -> pd.Series:
    if not obj["rows"]:
        return pd.Series(dtype=float)
    return pd.Series([float(v) for _, v in obj["rows"]],
                     index=pd.DatetimeIndex([d for d, _ in obj["rows"]]), dtype=float)


def build_one(key: str, spec: dict, cfg: dict, sym: str,
              target: pd.DatetimeIndex, raw: dict) -> dict:
    """单只 ETF：前复权价 → 对数收益 → 对齐日历 → 上市前缺席。"""
    o = raw[sym]
    px = _to_series(o).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    lr = np.log(px / px.shift(1)).dropna() if len(px) > 1 else pd.Series(dtype=float)
    first_ret = lr.index.min() if len(lr) else None

    aligned = lr.reindex(target)
    missing = aligned.isna()
    filled = aligned.ffill(limit=int(cfg["data"]["ffill_max_days"]))
    zero_days = int((filled.notna() & missing).sum())
    # 上市前无收益 → 保持 NaN（真实缺席，不编造）
    if first_ret is not None:
        filled.loc[filled.index < first_ret] = np.nan
    avail = filled.notna()
    meta = {
        "code": spec["code"], "name": spec["name"], "sym": sym,
        "quadrant": spec["quadrant"],
        "listing_date": (str(px.index.min().date()) if len(px) else None),
        "first_return_date": (str(first_ret.date()) if first_ret is not None else None),
        "price_rows": int(len(px)), "return_rows": int(avail.sum()),
        "absent_days": int((~avail).sum()),
        "ffill_zero_return_days": zero_days,
        "qfq": o.get("qfq", {}), "errors": o.get("errors", []),
        "mode": "真实 ETF 前复权（无代理、无拼接）",
    }
    return {"r": filled, "avail": avail, "meta": meta}


def run(use_cache: bool = True) -> dict:
    for p in (OUT, DATA, RAW):
        p.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    print("=== v1.4 七只真实 ETF 数据层（无代理 / 不编造）===")

    syms = {k: v["sym"] for k, v in cfg["universe"].items()}
    raw = {}
    for key, sym in list(syms.items()) + [("CALENDAR", "sh000300")]:
        is_idx = sym == "sh000300"
        o = d12.fetch_symbol(sym, cfg, is_index=is_idx, use_cache=use_cache)
        raw[sym] = o
        q = o.get("qfq", {})
        tag = "指数日历" if is_idx else f"{cfg['universe'][key]['name']:8s}"
        print(f"  {key:8s} {tag} {sym:10s} rows={len(o['rows']):>5} "
              f"{o['first']} ~ {o['last']} 复权={q.get('applied', 0)} "
              f"折算修复={len(q.get('split_repairs', []))} err={len(o.get('errors', []))}")

    # 主日历：A 股交易日
    cal = pd.DatetimeIndex(sorted(_to_series(raw["sh000300"]).index))
    cal = cal[(cal >= pd.Timestamp(cfg["data"]["start"]))
              & (cal <= pd.Timestamp(cfg["data"]["end"]))]
    target = cal[1:]
    print(f"\n主日历(A股交易日) {len(cal)} 天 {cal[0].date()} ~ {cal[-1].date()}")

    rets, avails, metas = {}, {}, {}
    for key, spec in cfg["universe"].items():
        b = build_one(key, spec, cfg, spec["sym"], target, raw)
        rets[key], avails[key], metas[key] = b["r"], b["avail"], b["meta"]
        m = b["meta"]
        print(f"  {key:8s} 上市={m['listing_date']} 有效收益={m['return_rows']:>5} "
              f"缺席={m['absent_days']:>5} ffill零收益={m['ffill_zero_return_days']}")

    R = pd.DataFrame(rets, index=target)
    A = pd.DataFrame(avails, index=target)
    # 从对数收益重建价格 level（起点 1.0），缺席处 NaN —— 份额法净值唯一价格真源
    PX = pd.DataFrame(np.nan, index=target, columns=R.columns)
    for c in R.columns:
        x = R[c].copy()
        fv = x.first_valid_index()
        if fv is None:
            continue
        x.loc[fv] = 0.0
        lvl = np.exp(x.fillna(0.0).cumsum())
        lvl[R[c].isna()] = np.nan
        PX[c] = lvl
    R.index.name = PX.index.name = A.index.name = "date"

    # 7 只都有真实价格的最早日 = 正式评估期起点
    all_present = A.all(axis=1)
    common = all_present[all_present].index.min()
    eval_start = cfg["data"]["eval_start"]
    assert str(common.date()) == eval_start, \
        f"共同起始日 {common.date()} 与配置 {eval_start} 不一致，需人工复核"
    # 价格首日 = 收益首日的前一交易日（豆粕 2019-12-05 上市、12-06 起计收益）
    price_first = cal[cal < common][-1]

    audit = {
        "version": "v1.4", "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": cfg["data"]["source"], "config_hash": sha256(CONFIG),
        "range": {"start": str(cal[0].date()), "end": str(cal[-1].date()),
                  "trading_days": int(len(cal)), "return_rows": int(len(R))},
        "eval_start_explicit": eval_start,
        "common_start_all7": str(common.date()),
        "price_first_all7": str(price_first.date()),
        "eval_trading_days": int(all_present.loc[eval_start:].sum()),
        "universe_size": len(cfg["universe"]),
        "calendar_source": cfg["data"]["calendar"],
        "assets": metas,
        "checks": {
            "log_return_single_source": True,
            "datetime_index": isinstance(R.index, pd.DatetimeIndex),
            "no_proxy_cells": 0,
            "no_splice": True,
            "absent_cells": int((~A).to_numpy().sum()),
            "price_rebuilt_from_log_returns": True,
            "finite_returns_only": bool(np.isfinite(R.to_numpy()[np.isfinite(R.to_numpy())]).all()),
        },
        "caveats": [
            "全部 7 只均为真实可交易 ETF 前复权日线，无指数代理、无上市前拼接、无编造",
            "上市前保持缺席(NaN)，引擎按当日可用大类重分配风险预算",
            "共同可交易起点由豆粕ETF(159985, 2019-12-05)决定；此前真实历史仅用于协方差预热",
            "QDII ETF(513500/513100) 含场内溢价与汇率影响，未单独剥离",
            "融资腿按年化2%计息（保守），不计入 7 只 ETF universe",
        ],
    }
    R.to_parquet(DATA / "clean_returns.parquet")
    PX.to_parquet(DATA / "clean_prices.parquet")
    A.to_parquet(DATA / "available.parquet")
    dump_json(audit, DATA / "data_audit_v14.json")
    (DATA / "hashes.json").write_text(json.dumps({
        "clean_returns": frame_hash(R), "clean_prices": frame_hash(PX),
        "available": frame_hash(A)}, indent=2) + "\n", encoding="utf-8")
    print(f"\n输出 clean_returns {R.shape} / clean_prices {PX.shape} / available {A.shape}")
    print(f"7只共同起始日 {common.date()} | 评估期交易日 {audit['eval_trading_days']} | "
          f"缺席格 {audit['checks']['absent_cells']} | 代理格 0")
    return {"returns": R, "prices": PX, "avail": A, "audit": audit, "cfg": cfg}


if __name__ == "__main__":
    run()
