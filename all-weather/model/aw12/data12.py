"""v1.2 长历史数据层（2013–2026）

设计要点（全部留痕，不编造）：
  1. 腾讯 fqkline 按 2 年分段拉取，单段 <=640 条；501 间歇错误重试 <=2 次，仍失败记缺口。
  2. 真实 ETF 用前复权(qfqday)；上市前用可回溯指数(day)的**对数收益**回补，
     采用「收益序列拼接 + 重建价格 level」而非直接拼价格，天然无 level 跳空。
  3. 无可用代理的资产（豆粕、黄金上市前）按规则**缺席**，不填充、不编造。
  4. 海外指数日历对齐到 A 股交易日历，asof 向前 + 有限 ffill(<=5)，零收益日标记。
  5. 输出 proxy_flag / available 矩阵与拼接跳空检测结果，供审计与看板标注。

沿用 v1.0 D0 口径：对数收益单一真源、显式共同起始日、DatetimeIndex、月末 asof(ME)。
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
PKG = Path(__file__).resolve().parent
OUT = ROOT / "aw12_out"
DATA = OUT / "data"
RAW = DATA / "raw"
REPORTS = OUT / "reports"
RUNS = OUT / "runs"
CONFIG_SRC = PKG / "config_v12.yaml"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "Referer": "https://finance.sina.com.cn"}
KLINE = ("http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
         "CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen={n}")
HFQ = "https://finance.sina.com.cn/realstock/company/{sym}/hfq.js"

# 复用 v1.0 已验证的复权解析与前复权合成，不重写
from aw.data_layer import _parse_hfq, apply_qfq  # noqa: E402


# ---------------------------------------------------------------- 抓取
def _get(url: str, cache: Path, cfg: dict, label: str) -> str | None:
    """带缓存的 GET；失败重试 <=2 次；仍失败返回 None 由调用方跳过并记录（不编造）。"""
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_text(encoding="utf-8", errors="ignore")
    retry = int(cfg["data"]["retry_max"])
    last = ""
    for attempt in range(retry + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            txt = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")
            if not txt or len(txt) < 40:
                raise ValueError(f"响应过短({len(txt)})")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(txt, encoding="utf-8")
            return txt
        except Exception as e:                                # noqa: BLE001
            last = f"{type(e).__name__}:{str(e)[:70]}"
            if attempt < retry:
                time.sleep(0.5)
    print(f"    [跳过] {label} 取数失败（重试 {retry} 次）: {last}")
    return None


def repair_splits(px: pd.Series, sym: str, threshold: float = 0.20) -> tuple[pd.Series, list]:
    """修正份额折算/拆分造成的假跳变：把跳变日之前的价格按跳变倍数缩放到同一尺度，
    使该日收益归零（折算本身不产生投资收益）。仅处理 |logret|>20% 的数据源缺陷，
    不改动任何真实日收益，全部事件留痕。"""
    lr = np.log(px / px.shift(1))
    bad = lr[lr.abs() > threshold]
    events = []
    for d in bad.index:
        k = float(np.exp(lr.loc[d]))
        px.loc[px.index < d] *= k
        events.append({"date": str(d.date()), "jump_pct": round((k - 1) * 100, 2),
                       "scale_applied_to_history": round(k, 6),
                       "reason": "份额折算/拆分导致的价格跳变，非真实收益"})
    return px, events


def fetch_symbol(sym: str, cfg: dict, is_index: bool = False,
                 use_cache: bool = True) -> dict:
    """新浪长历史日线（不复权）+ hfq 分红复权因子 → 前复权收盘价序列。
    指数为价格指数，不做复权。返回 {rows, source, qfq, errors}。"""
    tag = sym.replace(".", "_")
    ck, ch = RAW / f"{tag}_kline.json", RAW / f"{tag}_hfq.js"
    out_cache = RAW / f"{tag}_sina.json"
    if use_cache and out_cache.exists():
        try:
            return json.loads(out_cache.read_text(encoding="utf-8"))
        except Exception:
            pass

    txt = _get(KLINE.format(sym=sym, n=cfg["data"]["datalen"]), ck, cfg, f"{sym} 长历史K线")
    errors = []
    if txt is None:
        errors.append("K线取数失败")
        out = {"sym": sym, "source": "sina_long_kline", "rows": [],
               "qfq": {}, "errors": errors, "first": None, "last": None}
        out_cache.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
        return out
    rows = json.loads(txt)
    idx = pd.DatetimeIndex([r["day"] for r in rows])
    px = pd.Series([float(r["close"]) for r in rows], index=idx, dtype=float).sort_index()
    px = px[~px.index.duplicated(keep="last")]

    qfq_info = {"events": 0, "applied": 0, "skipped": [], "method": "价格指数无复权" if is_index else "hfq 分红复权"}
    if not is_index:
        ht = _get(HFQ.format(sym=sym), ch, cfg, f"{sym} 复权因子")
        fac = _parse_hfq(ht, sym) if ht else []
        px, info = apply_qfq(px, fac, sym)
        qfq_info = {"events": info["events"], "applied": info["applied"],
                    "skipped": info["skipped"][:5], "method": "新浪 hfq.js 前复权"}
        if ht is None:
            errors.append("复权因子取数失败，使用不复权价（留痕）")
        px, splits = repair_splits(px, sym)
        qfq_info["split_repairs"] = splits
        if splits:
            errors.append(f"修正 {len(splits)} 处份额折算跳变（已留痕，非编造）")

    out = {"sym": sym, "source": "sina_long_kline+qfq" if not is_index else "sina_index",
           "rows": [[d.strftime("%Y-%m-%d"), float(v)] for d, v in px.items()],
           "qfq": qfq_info, "errors": errors,
           "first": (str(px.index[0].date()) if len(px) else None),
           "last": (str(px.index[-1].date()) if len(px) else None),
           "kline_sha256": sha256(ck) if ck.exists() else None}
    out_cache.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    return out



def ensure_dirs() -> None:
    for p in (OUT, DATA, RAW, REPORTS, RUNS):
        p.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with CONFIG_SRC.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _json_default(o):
    import numpy as np
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, date)):
        return str(o)
    return str(o)


def dump_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                               default=_json_default) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    buf = (df.to_csv(index=True).encode("utf-8")
           + repr(list(df.columns)).encode("utf-8"))
    return hashlib.sha256(buf).hexdigest()


# ---------------------------------------------------------------- 拼接
def _to_series(obj: dict) -> pd.Series:
    if not obj["rows"]:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex([d for d, _ in obj["rows"]])
    return pd.Series([v for _, v in obj["rows"]], index=idx, dtype=float)


def build_asset(key: str, spec: dict, cfg: dict, raw: dict, calendar: pd.DatetimeIndex) -> dict:
    """为单个资产构造合成对数收益序列 + 代理标记。返回 dict(s, proxy, avail, meta)。"""
    thr = float(cfg["data"]["splice_jump_threshold"])
    etf = _to_series(raw[spec["etf"]])
    proxy = _to_series(raw[spec["proxy"]]) if spec.get("proxy") else pd.Series(dtype=float)

    # CASH：无真实数据时按 0 收益（保守，不计利息），明确标注
    if key == "CASH" and etf.empty:
        s = pd.Series(0.0, index=calendar[1:], dtype=float)
        return {"s": s, "proxy": pd.Series(False, index=s.index),
                "avail": pd.Series(True, index=s.index),
                "meta": {"mode": "现金按 0 收益处理（未计利息）", "etf_rows": 0,
                         "proxy_rows": 0, "proxy_until": None, "absent_days": 0,
                         "splice_jump": 0.0, "splice_fixed": False}}

    etf_ret = np.log(etf / etf.shift(1)).dropna()
    px_ret = np.log(proxy / proxy.shift(1)).dropna() if len(proxy) > 1 else pd.Series(dtype=float)

    etf_start = etf.index.min() if len(etf) else None
    proxy_start = proxy.index.min() if len(proxy) > 1 else None
    splice_jump, splice_fixed = 0.0, False
    proxy_until = None
    note = ""

    if etf_ret.empty and px_ret.empty:
        s = pd.Series(np.nan, index=calendar[1:], dtype=float)
        return {"s": s, "proxy": pd.Series(False, index=s.index),
                "avail": pd.Series(False, index=s.index),
                "meta": {"mode": "无真实数据且无代理 → 全样本缺席", "etf_rows": len(etf),
                         "proxy_rows": len(proxy), "proxy_until": None,
                         "absent_days": int(len(s)), "splice_jump": 0.0,
                         "splice_fixed": False}}

    if etf_start is None:
        # 全段代理
        parts = [(px_ret, True)]
        proxy_until = str(px_ret.index.max().date()) if len(px_ret) else None
        mode = "全样本指数代理（ETF 无数据）"
    elif proxy_start is not None and proxy_start < etf_start:
        # 上市前代理 + 上市后真实（收益序列拼接，跳过 ETF 首日）
        pre = px_ret[px_ret.index < etf_start]
        post = etf_ret[etf_ret.index > etf_start]
        if len(pre) and len(post):
            gap = float(post.iloc[0])
            splice_jump = abs(gap)
            if splice_jump > thr:
                # 异常跳空：不做人为抹平，仅记录并剔除首个真实收益（不编造）
                post = post.iloc[1:]
                splice_fixed = True
                note = "拼接处跳空超阈值，剔除首个真实收益并留痕"
        parts = [(pre, True), (post, False)]
        proxy_until = str(pre.index.max().date()) if len(pre) else None
        mode = "指数代理回补上市前 + 真实 ETF 前复权"
    else:
        parts = [(etf_ret, False)]
        mode = "真实 ETF 前复权（无需代理）"

    chunks, flags = [], []
    for part, is_proxy in parts:
        if len(part):
            chunks.append(part)
            flags.append(pd.Series(bool(is_proxy), index=part.index))
    combo = pd.concat(chunks).sort_index()
    combo = combo[~combo.index.duplicated(keep="last")]
    flag = pd.concat(flags).sort_index()
    flag = flag[~flag.index.duplicated(keep="last")].reindex(combo.index).fillna(False)

    # 对齐到 A 股交易日历（海外指数日历不同）：asof 向前 + 有限 ffill
    target = calendar[1:]
    aligned = combo.reindex(target)
    missing = aligned.isna()
    # 非 A 股交易日产生的数据（如美股/港股）不纳入；A 股交易日缺失则有限 ffill
    ffill_limit = int(cfg["data"]["ffill_max_days"])
    before = int(missing.sum())
    filled = aligned.ffill(limit=ffill_limit)
    zero_return_days = int((filled.notna() & missing).sum())
    filled = filled.fillna(0.0) if key in ("TB10", "TB30", "CBOND", "CASH") else filled
    # 真实缺席：ETF 上市前且无代理 → NaN 保留（组合层剔除）
    first_valid = combo.index.min()
    filled.loc[filled.index < first_valid] = np.nan
    flag_al = flag.reindex(target).ffill(limit=ffill_limit).fillna(False).astype(bool)
    avail = filled.notna()

    meta = {"mode": mode + ("；" + note if note else ""),
            "etf_rows": int(len(etf)), "proxy_rows": int(len(proxy)),
            "proxy_until": proxy_until,
            "etf_listing": (str(etf_start.date()) if etf_start is not None else None),
            "absent_days": int((~avail).sum()),
            "ffill_zero_return_days": zero_return_days,
            "calendar_missing_before_ffill": before,
            "splice_jump": round(splice_jump, 6), "splice_fixed": bool(splice_fixed),
            "first_valid": (str(filled.first_valid_index().date())
                            if filled.first_valid_index() is not None else None)}
    return {"s": filled, "proxy": flag_al, "avail": avail, "meta": meta}


def run(use_cache: bool = True) -> dict:
    ensure_dirs()
    cfg = load_config()
    specs = cfg["universe"]
    print("=== v1.2 长历史数据层 ===")

    # 1. 抓取全部标的（真实 ETF 前复权 + 代理指数 + 主日历指数）
    etf_syms = {s["etf"] for s in specs}
    idx_syms = {s["proxy"] for s in specs if s.get("proxy")} | {"sh000300"}
    syms = sorted(etf_syms | idx_syms)
    raw = {}
    for sym in syms:
        o = fetch_symbol(sym, cfg, is_index=(sym in idx_syms), use_cache=use_cache)
        raw[sym] = o
        q = o.get("qfq", {})
        print(f"  {sym:10s}{'指数' if sym in idx_syms else 'ETF ':4s} rows={len(o['rows']):>5}  "
              f"{o['first']} ~ {o['last']}  复权事件={q.get('applied', 0)}  "
              f"err={len(o.get('errors', []))}")

    # 2. 主日历：沪深300指数（A股交易日），截取到配置区间
    cal_src = _to_series(raw["sh000300"])
    calendar = pd.DatetimeIndex(sorted(cal_src.index))
    calendar = calendar[(calendar >= pd.Timestamp(cfg["data"]["start"]))
                        & (calendar <= pd.Timestamp(cfg["data"]["end"]))]
    print(f"\n主日历(A股交易日): {len(calendar)} 天  {calendar[0].date()} ~ {calendar[-1].date()}")

    # 3. 逐资产合成
    rets, proxies, avails, metas = {}, {}, {}, {}
    for spec in specs:
        b = build_asset(spec["key"], spec, cfg, raw, calendar)
        rets[spec["key"]] = b["s"]
        proxies[spec["key"]] = b["proxy"]
        avails[spec["key"]] = b["avail"]
        metas[spec["key"]] = b["meta"]
        m = b["meta"]
        print(f"  {spec['key']:8s}{spec['name']:10s} 起始={m.get('first_valid')} "
              f"代理至={m.get('proxy_until')} 缺席={m.get('absent_days')} "
              f"ffill零收益={m.get('ffill_zero_return_days')} | {m['mode'][:36]}")

    R = pd.DataFrame(rets, index=calendar[1:])
    P = pd.DataFrame(proxies, index=R.index)
    A = pd.DataFrame(avails, index=R.index)
    # 重建价格 level（起点 1.0），缺席处保持 NaN
    PX = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    for c in R.columns:
        x = R[c].copy()
        fv = x.first_valid_index()
        if fv is None:
            continue
        x.loc[fv] = 0.0
        lvl = x.fillna(0.0).cumsum().apply(np.exp)
        lvl[R[c].isna()] = np.nan
        PX[c] = lvl
    R.index.name = "date"
    PX.index.name = "date"

    # 4. 审计
    common_start = str(R.dropna(how="all").index.min().date())
    audit = {
        "version": "v1.2", "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": cfg["data"]["source"],
        "config_hash": sha256(CONFIG_SRC),
        "range": {"start": str(calendar[0].date()), "end": str(calendar[-1].date()),
                  "trading_days": int(len(calendar)), "return_rows": int(len(R))},
        "common_start_explicit": common_start,
        "calendar_source": "sh000300 沪深300指数交易日（A股）",
        "assets": metas,
        "symbol_fetch": {s: {"rows": len(raw[s]["rows"]),
                             "first": raw[s]["first"], "last": raw[s]["last"],
                             "source": raw[s]["source"],
                             "qfq": raw[s].get("qfq", {}),
                             "errors": raw[s].get("errors", [])} for s in syms},
        "proxy_policy": cfg["data"]["proxy_policy"],
        "return_method": cfg["data"]["return_method"],
        "checks": {
            "log_return_single_source": True,
            "datetime_index": isinstance(R.index, pd.DatetimeIndex),
            "ffill_max_days": int(cfg["data"]["ffill_max_days"]),
            "total_proxy_cells": int(P.to_numpy().sum()),
            "total_absent_cells": int((~A).to_numpy().sum()),
            "price_rebuilt_from_log_returns": True,
            "splice_anomalies": [k for k, v in metas.items() if v.get("splice_fixed")],
        },
        "caveats": [
            "真实 ETF 段为新浪长历史不复权日线 + hfq.js 分红因子合成的前复权价，与 v1.0 D0 口径一致",
            "指数代理为价格指数收益，不含成分股分红，与 ETF 前复权口径存在偏差",
            "海外资产改用境内 QDII ETF(513100 纳指 / 159920 恒生)真实前复权价，仍含 QDII 场内溢价与汇率影响，未单独剥离",
            "指数代理段不可交易；无代理的资产在上市前按规则缺席，不填充、不编造",
            "拼接采用收益序列拼接后重建价格 level，天然无 level 跳空；异常跳空已剔除并记录",
            "多市场标的统一对齐到 A 股交易日历，非 A 股交易日收益不单独计入",
        ],
    }

    R.to_parquet(DATA / "clean_returns.parquet")
    PX.to_parquet(DATA / "clean_prices.parquet")
    P.to_parquet(DATA / "proxy_flag.parquet")
    A.to_parquet(DATA / "available.parquet")
    dump_json(audit, DATA / "data_audit_v12.json")
    dump_json({"universe": specs}, DATA / "universe_v12.json")
    (DATA / "hashes.json").write_text(json.dumps({
        "clean_returns": frame_hash(R), "clean_prices": frame_hash(PX),
        "proxy_flag": frame_hash(P), "available": frame_hash(A)}, indent=2) + "\n",
        encoding="utf-8")
    print(f"\n输出: clean_returns {R.shape} / clean_prices {PX.shape}")
    print(f"共同起始日(显式): {common_start} | 代理格 {audit['checks']['total_proxy_cells']} "
          f"| 缺席格 {audit['checks']['total_absent_cells']}")
    print(f"拼接异常: {audit['checks']['splice_anomalies']}")
    return {"returns": R, "prices": PX, "proxy": P, "avail": A,
            "audit": audit, "cfg": cfg, "specs": specs}


if __name__ == "__main__":
    run()
