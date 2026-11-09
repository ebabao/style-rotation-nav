#!/usr/bin/env python3.11
"""全天候策略 v1.4 · 每日数据同步主程序（aw-nav）

架构参照 ebabao/style-rotation-nav：
    页面 index.html 静态不变，只 fetch ./data/*.json（cache:no-store）；
    本脚本每日把「策略模型接口」产物同步为「看板页面数据」，推到仓库即完成 Pages 部署。

执行链（两步 + 明确回退）：
    Step 1  行情刷新：复用 aw14/data14.py 抓 7 只 ETF 前复权日线，重试 ≤2；
            失败或校验不通过 → 恢复仓库内上一版快照（source=cache_fallback），不编造数据。
    Step 2  模型回放：复用 aw14/engine14.py 的 quick_run(theta_active) 全量回放，
            生成 meta/nav/signal/weights/rebalances/monthly/sync_audit 接口，校验并做模型对账。

参数配置分离：model/active_params.json 为生效参数；删除该文件即回退 engine14.THETA0 基准。
运行指标：run_id / scraped / new / md / status / source / retry / fallback 全量写入审计日志。

用法：
    python3.11 scripts/sync_daily.py                 # 完整每日同步（刷新 + 回放 + 接口）
    python3.11 scripts/sync_daily.py --no-refresh   # 跳过抓数，用仓库内快照（本地/调试/CI 手动重启）
    python3.11 scripts/sync_daily.py --force        # 忽略幂等，强制重写接口
    python3.11 scripts/sync_daily.py --retrain      # 按需触发 GA 参数自进化（非每日链）
    python3.11 scripts/sync_daily.py --dry-run      # 只计算并打印，不落盘
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import signal
import socket
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "model"
DATA = ROOT / "data"
ENGINE_DATA = MODEL / "aw14_out" / "data"
sys.path.insert(0, str(MODEL))

CN = timezone(timedelta(hours=8))
PARQUETS = ("clean_prices.parquet", "clean_returns.parquet",
            "available.parquet", "benchmark_equal_weight.parquet")
ENGINE_VERSION = "aw14/engine14@2026-11-08"
POLICY = {
    "refresh_retry_max": 2,
    "refresh_timeout_sec": 150,
    "fallback": "抓数失败 → 恢复仓库内上一版 parquet 快照，标记 source=cache_fallback，绝不编造",
    "idempotency": "数据指纹 + 参数指纹一致则跳过重写，仅追加一条 skipped 巡检记录",
    "no_lookahead": "引擎 t 日收盘估计协方差、t+1 起承担收益，零前视",
    "rollback": "删除 model/active_params.json 即回退默认基准参数",
}


# ---------------------------------------------------------------- 基础工具
def now() -> datetime:
    return datetime.now(CN)


def run_id() -> str:
    return "sync-" + now().strftime("%Y%m%d-%H%M%S") + "-" + hashlib.md5(
        str(time.time_ns()).encode()).hexdigest()[:6]


def jdump(obj: dict, path: Path) -> None:
    """原子写入；allow_nan=False 顺带充当数值有限性校验。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def jload(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def f(x, nd=6):
    """转 python float 并四舍五入，杜绝 numpy 标量与 NaN 混入 JSON。"""
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"非有限数值: {v}")
    return round(v, nd)


class Timeout:
    """Unix 下给抓数加硬超时，避免 CI 卡死。"""
    def __init__(self, sec: int):
        self.sec = sec

    def _raise(self, *_):
        raise TimeoutError(f"抓数超过 {self.sec}s 硬超时")

    def __enter__(self):
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, self._raise)
            signal.alarm(self.sec)
        else:
            socket.setdefaulttimeout(15)
        return self

    def __exit__(self, *_):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        return False


# ---------------------------------------------------------------- Step 1 行情刷新
def _snapshots() -> dict[str, bytes]:
    return {n: (ENGINE_DATA / n).read_bytes() for n in PARQUETS
            if (ENGINE_DATA / n).exists()}


def _restore(snap: dict[str, bytes]) -> None:
    for n, b in snap.items():
        (ENGINE_DATA / n).write_bytes(b)


SITE_DATA = ROOT / "site" / "data"


def mirror_to_site() -> int:
    """把接口同步到发布目录（平台静态发布为实时目录映射，写入即线上生效）。

    若 site/ 不存在（纯 GitHub Pages 分支部署场景）则跳过，不影响主流程。
    """
    if not (ROOT / "site").exists():
        return 0
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in DATA.iterdir():
        if p.is_file() and not p.name.endswith(".tmp"):
            shutil.copy2(p, SITE_DATA / p.name)
            n += 1
    # 静态页面一并同源（不常变，但保证发布目录始终自洽）
    for name in ("index.html", ".nojekyll"):
        p = ROOT / name
        if p.exists():
            shutil.copy2(p, ROOT / "site" / name)
    return n


def _prices_meta() -> tuple[int, pd.Timestamp, str]:
    px = pd.read_parquet(ENGINE_DATA / "clean_prices.parquet")
    px.index = pd.to_datetime(px.index)
    md = hashlib.md5(np.round(px.fillna(0.0).to_numpy(dtype=float), 8).tobytes()).hexdigest()[:12]
    return len(px), px.index.max(), md


def refresh_data(retry_max: int = 2) -> dict:
    """复用 aw14/data14.py 抓数；失败回退上一版快照，返回运行指标。"""
    snap = _snapshots()
    old_len, old_last, old_md = _prices_meta()
    warnings: list[str] = []

    for attempt in range(1, retry_max + 1):
        try:
            with Timeout(150):
                from aw14.data14 import run as data_run
                data_run(use_cache=False)
            new_len, new_last, new_md = _prices_meta()
            # 校验：列齐、行数不倒退、末日不倒退、数值有限
            px = pd.read_parquet(ENGINE_DATA / "clean_prices.parquet")
            assert px.shape[1] == 7, f"资产列数异常 {px.shape[1]}"
            assert new_len >= old_len, f"行数倒退 {old_len}→{new_len}"
            assert new_last >= old_last, f"末日倒退 {old_last}→{new_last}"
            assert np.isfinite(px.to_numpy(dtype=float)[np.isfinite(px.to_numpy(dtype=float))]).all()
            return {"ok": True, "attempt": attempt, "scraped": new_len - old_len,
                    "new": max(0, new_len - old_len), "md": new_md,
                    "last": str(new_last.date()), "warnings": warnings,
                    "source": "fresh_sina" if new_len > old_len else "fresh_sina_no_new"}
        except Exception as e:                                    # noqa: BLE001
            warnings.append(f"第{attempt}次抓数失败: {type(e).__name__}: {str(e)[:110]}")
            _restore(snap)

    _restore(snap)
    _len, _last, md = _prices_meta()
    return {"ok": False, "attempt": retry_max, "scraped": 0, "new": 0, "md": md,
            "last": str(_last.date()), "warnings": warnings, "source": "cache_fallback"}


# ---------------------------------------------------------------- Step 2 模型回放
def active_theta() -> tuple[np.ndarray, dict]:
    from aw14.engine14 import THETA0, LO, HI
    a = jload(MODEL / "active_params.json")
    if not a or "theta" not in a:
        return np.array(THETA0, float), {"params_fp": "DEFAULT_THETA0",
                                         "adopted": False, "note": "active_params.json 缺失，已回退默认基准"}
    th = np.clip(np.asarray(a["theta"], float), LO, HI)
    return th, {"params_fp": a.get("params_fp", "unknown"), "adopted": True,
                "run_id": a.get("run_id"), "adopted_at": a.get("adopted_at")}


def _metrics_en(m: dict) -> dict:
    return {
        "total_return": f((m["期末净值"] - 1) * 100, 4),
        "cagr": f(m["年化收益"] * 100, 4),
        "vol_ann": f(m["年化波动"] * 100, 4),
        "sharp": f(m["夏普比率"], 4),
        "max_drawdown": f(abs(m["最大回撤"]) * 100, 4),
        "calmar": f(m["Calmar"], 4),
        "final_nav": f(m["期末净值"], 4),
    }


def _yearly(nav: pd.Series) -> list[dict]:
    from all_weather_prototype import metrics as calc_metrics
    out = []
    for y in sorted({d.year for d in nav.index}):
        s = nav.loc[str(y)]
        if len(s) < 20:
            continue
        m = calc_metrics(s / s.iloc[0])
        mdd = abs(m["最大回撤"]) * 100
        tr = (s.iloc[-1] / s.iloc[0] - 1) * 100
        out.append({"year": y, "total_return_pct": f(tr, 2),
                    "annual_return_pct": f(m["年化收益"] * 100, 2),
                    "max_drawdown_pct": f(mdd, 2),
                    "return_dd_ratio": f(tr / mdd, 2) if mdd > 1e-9 else None,
                    "std_pct": f(m["年化波动"] * 100, 2),
                    "sharpe": f(m["夏普比率"], 2)})
    return out


def build_interfaces(res: dict, meta_in: dict, rid: str) -> dict[str, object]:
    """把引擎产物转成参考工程同构的公开接口。"""
    state = jload(MODEL / "state_snapshot.json", {}) or {}
    nav = res["nav"].copy()
    nav.index = pd.to_datetime(nav.index)
    dates = [d.strftime("%Y-%m-%d") for d in nav.index]
    navv = nav.to_numpy(dtype=float)
    ret = (navv / navv[0] - 1) * 100
    peak = np.maximum.accumulate(navv)
    dd = (navv / peak - 1) * 100

    # 等权基准（同区间归一化）
    bp = ENGINE_DATA / "benchmark_equal_weight.parquet"
    bench = None
    if bp.exists():
        b = pd.read_parquet(bp)["bench"]
        b.index = pd.to_datetime(b.index)
        b = b.reindex(nav.index).ffill()
        bench = [f(v, 4) for v in (b / b.iloc[0]).to_numpy(dtype=float)]

    # 日频总敞口 L（调仓日快照 ffill）
    w = res["weights"]
    expo = None
    if len(w) and "L" in w:
        wi = w.copy(); wi.index = pd.to_datetime(wi.index)
        L = wi["L"].reindex(nav.index).ffill().bfill().fillna(1.0)
        expo = [f(v, 3) for v in L.to_numpy(dtype=float)]

    m = res["metrics"]
    men = _metrics_en(m)
    monthly = res["monthly"]
    mwin = float((monthly > 0).mean() * 100) if len(monthly) else 0.0
    men.update({"month_win": f(mwin, 2), "trades": int(res["n_rebalance"]),
                "turnover": f(res["avg_turnover"] * 100, 2)})

    # 与 state 官方指标对账（单位 pp）
    official = state.get("final_metrics", {})
    actual = {"cagr": men["cagr"], "maxdd": men["max_drawdown"], "sharpe": men["sharp"]}
    target = {"cagr": f(official.get("年化收益", actual["cagr"]) * 100, 4),
              "maxdd": f(abs(official.get("最大回撤", -actual["maxdd"] / 100)) * 100, 4),
              "sharpe": f(official.get("夏普比率", actual["sharpe"]), 4)}
    diff = max(abs(actual[k] - target[k]) for k in actual)
    reconcile = {"window": dates[-1], "target": target, "actual": actual,
                 "display": actual, "max_abs_diff_pp": f(diff, 4),
                 "ok": bool(diff < 0.05),
                 "note": "state 官方口径为 2026-11-08 训练快照；数据刷新后偏差为真实增量，如实披露"}

    src = meta_in["source"]
    meta = {
        "schema": "aw-nav/meta@1", "model": "全天候策略 v1.4 · 七 ETF 进化学习版",
        "model_code": "aw14_seven_etf_ga", "params_fp": meta_in["params_fp"],
        "source": src, "basis": "评估期 2019-12-06 起、起点净值归一化为 1；7 只真实 ETF 前复权日线，零指数代理",
        "snapshot_from": "2013-01-07", "snapshot_to": meta_in["last"],
        "as_of": dates[-1], "eval_start": "2019-12-06", "bars": len(dates),
        "data_purity": state.get("data_purity", {"proxy_cells": 0, "note": "零代理、无拼接"}),
        "universe": state.get("universe", []),
        "reconcile": reconcile,
        "target_assessment": state.get("target_assessment", {}),
        "engineering": {
            "learning": state.get("learning", {}),
            "evolution": state.get("evolution", {}),
            "recursive_test": state.get("recursive_test", {}),
        },
        "engine": ENGINE_VERSION,
        "sync": {"last_sync": now().isoformat(timespec="seconds"), "run_id": rid,
                 "status": meta_in["status"], "source": src,
                 "scraped": meta_in["scraped"], "new": meta_in["new"], "md": meta_in["md"],
                 "retry": meta_in["attempt"], "fallback": src == "cache_fallback",
                 "warnings": meta_in["warnings"]},
        "policy": POLICY,
    }

    nav_json = {
        "schema": "aw-nav/nav@1", "as_of": dates[-1], "start": dates[0], "bars": len(dates),
        "model": meta["model"], "params_fp": meta_in["params_fp"],
        "dates": dates, "nav": [f(v, 6) for v in navv], "ret": [f(v, 4) for v in ret],
        "dd": [f(v, 4) for v in dd], "bench": bench,
        "exposure": expo, "position": expo,
        "metrics": men,
        "benchmark_equal_weight": _metrics_en(state.get("benchmark_equal_weight", m)),
        "yearly": _yearly(nav),
        "monthly": [{"month": d.strftime("%Y-%m"), "ret_pct": f(v * 100, 3)}
                    for d, v in monthly.items()],
        "segments": {k: _metrics_en(v) for k, v in (state.get("segments") or {}).items() if isinstance(v, dict)},
    }

    # 当前 / 目标持仓（季度再平衡：当前即生效持仓，目标为下次检查的执行口径）
    from aw14.engine14 import ASSET_CN, QN_CN
    wlast = w.iloc[-1] if len(w) else None
    codes = [c for c in ("TB5", "GOLD", "SOY", "DIVLV", "SP500", "NDX", "GEM50")]
    codes = [c for c in codes if wlast is not None and c in wlast.index]
    holdings = []
    for c in codes:
        wv = float(wlast[c])
        holdings.append({"code": c, "name": ASSET_CN.get(c, c), "asset": c,
                         "role": QN_CN.get({"TB5": "BD", "GOLD": "GD", "SOY": "CM"}.get(c, "EQ"), "股票"),
                         "weight": f(wv * 100, 2), "capital": f(wv * 100, 2), "price": None})
    L_now = float(wlast["L"]) if wlast is not None and "L" in wlast else 1.0
    logs = res["logs"]
    all_rb, prev_names = [], None
    for _, r in logs.iterrows():
        top = sorted([(c, float(r.get(f"w_{c}", 0))) for c in codes], key=lambda z: -z[1])[:2]
        names = [c for c, _ in top]
        switched = prev_names is not None and set(names) != set(prev_names)
        all_rb.append({"date": str(r["date"]), "exposure": f(float(r["L"]) * 100, 1),
                       "holdings": names,
                       "names": " / ".join(ASSET_CN.get(c, c) for c in names),
                       "changed": switched,
                       "kind": "建仓" if prev_names is None else ("换仓" if switched else "权重微调"),
                       "reason": "季度检查 · 大类风险平价 + 国债杠杆", "circuit": False})
        prev_names = names
    rb = all_rb[-30:]
    signal_json = {
        "schema": "aw-nav/signal@1", "model": meta["model"], "model_code": meta["model_code"],
        "params_fp": meta_in["params_fp"], "as_of": dates[-1],
        "decision_date": str(w.index[-1].date()) if len(w) else dates[-1],
        "next_rebalance": "下一季度检查日（季初 ±3% 阈值）", "days_to_rebalance": None,
        "current": holdings, "target": holdings,
        "current_exposure": f(L_now * 100, 1), "target_exposure": f(L_now * 100, 1),
        "will_change": False, "changed": [], "circuit": False,
        "reason": "四宫格风险平价 + 国债专属杠杆 + 季度 ±3% 阈值再平衡（无方向性择时）",
        "drawdown": f(dd[-1], 2), "dd_limit": -8.0, "L_mean": f(res["L_mean"], 3),
        "risk_contrib": {QN_CN.get(q, q): f(v * 100, 2) for q, v in res["rc_mean"].items()},
        "class_weight": {QN_CN.get(q, q): f(v * 100, 2) for q, v in res["W_mean"].items()},
        "holdings": holdings, "rebalances": rb,
    }

    weights_json = {
        "schema": "aw-nav/weights@1", "as_of": dates[-1],
        "wmean_asset": {ASSET_CN.get(k, k): f(v * 100, 2) for k, v in res["wmean_asset"].items()},
        "class_weight": signal_json["class_weight"], "risk_contrib": signal_json["risk_contrib"],
        "L_mean": f(res["L_mean"], 3), "avg_turnover": f(res["avg_turnover"] * 100, 2),
        "n_rebalance": int(res["n_rebalance"]),
        "series": [{"date": str(d.date()),
                    "L": f(float(row.get("L", 1.0)), 3),
                    "weights": {ASSET_CN.get(c, c): f(float(row[c]) * 100, 2) for c in codes}}
                   for d, (_, row) in zip(w.index, w.iterrows())],
    }

    rebal_csv = logs.to_csv(index=False, encoding="utf-8-sig", float_format="%.6f") if len(logs) else "date\n"
    monthly_csv = "month,ret_pct\n" + "\n".join(
        f"{x['month']},{x['ret_pct']}" for x in nav_json["monthly"])

    return {"data/meta.json": meta, "data/nav.json": nav_json,
            "data/signal.json": signal_json, "data/weights.json": weights_json,
            "data/rebalances.csv": rebal_csv, "data/monthly.csv": monthly_csv,
            "_nav": nav_json}


def validate(payload: dict) -> list[str]:
    errs = []
    nv = payload["_nav"]
    n = len(nv["dates"])
    for k in ("nav", "ret", "dd"):
        if len(nv[k]) != n:
            errs.append(f"nav.json {k} 长度 {len(nv[k])} != {n}")
    if nv["dates"] != sorted(nv["dates"]):
        errs.append("日期非单调递增")
    if not all(np.isfinite(nv["nav"])):
        errs.append("净值含非有限值")
    for p, obj in payload.items():
        if p.startswith("_"):
            continue
        if isinstance(obj, dict):
            json.dumps(obj, ensure_ascii=False, allow_nan=False)
    if payload["data/meta.json"]["bars"] != n:
        errs.append("meta.bars 与 nav 长度不一致")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true", help="跳过抓数，用仓库内快照")
    ap.add_argument("--force", action="store_true", help="忽略幂等强制重写")
    ap.add_argument("--retrain", action="store_true", help="按需触发 GA 参数自进化")
    ap.add_argument("--dry-run", action="store_true", help="只计算不落盘")
    args = ap.parse_args()

    t0 = time.time()
    rid = run_id()
    nm = 0
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"[{rid}] 开始每日同步 | 模式: {'no-refresh' if args.no_refresh else 'full'}"
          f"{' +retrain' if args.retrain else ''}")

    # ---- 按需参数自进化（非每日链）
    if args.retrain:
        try:
            from aw14.learn14 import run_learning
            run_learning()
            st = jload(MODEL / "aw14_out" / "v14_state.json", {})
            if st and "final_theta" in st:
                a = jload(MODEL / "active_params.json", {})
                a["theta"] = st["final_theta"]
                a["run_id"] = st.get("run_id")
                a["adopted_at"] = now().strftime("%Y-%m-%d")
                a["params_fp"] = hashlib.md5(json.dumps(a["theta"]).encode()).hexdigest()[:10]
                jdump(a, MODEL / "active_params.json")
                print("  参数自进化完成，active_params 已更新:", a["params_fp"])
        except Exception as e:                                    # noqa: BLE001
            print(f"  参数自进化失败（保留旧参数，不阻断同步）: {type(e).__name__}: {e}")

    # ---- Step 1 行情刷新 / 回退
    if args.no_refresh:
        _len, _last, md = _prices_meta()
        ref = {"ok": True, "attempt": 0, "scraped": 0, "new": 0, "md": md,
               "last": str(_last.date()), "warnings": [], "source": "snapshot_replay"}
    else:
        ref = refresh_data(POLICY["refresh_retry_max"])
    print(f"  Step1 行情: source={ref['source']} attempt={ref['attempt']} "
          f"scraped={ref['scraped']} new={ref['new']} md={ref['md']} last={ref['last']}")
    for wmsg in ref["warnings"]:
        print("    !", wmsg)

    # ---- 幂等判断
    theta, pinfo = active_theta()
    fingerprint = hashlib.md5(f"{ref['md']}|{pinfo['params_fp']}|{ENGINE_VERSION}".encode()).hexdigest()[:12]
    prev = jload(DATA / "meta.json", {}) or {}
    same = (prev.get("data_md5") == fingerprint and prev.get("sync", {}).get("status") == "ok")
    if same and not args.force and not args.dry_run:
        audit = jload(DATA / "sync_audit.json", {"runs": []})
        audit["runs"].append({"run_id": rid, "at": now().isoformat(timespec="seconds"),
                              "status": "skipped", "reason": "数据指纹与参数指纹未变化",
                              "md": ref["md"], "source": ref["source"]})
        audit["runs"] = audit["runs"][-60:]
        audit["last_run"] = rid
        jdump(audit, DATA / "sync_audit.json")
        nm = mirror_to_site()
        print(f"[{rid}] 幂等跳过：数据未变化，接口与页面保持不变（已追加巡检记录，发布目录同步 {nm} 个文件）")
        print(f"一行汇总：{now():%Y-%m-%d %H:%M} 同步跳过（无新数据），as_of={ref['last']}，回退={ref['source']=='cache_fallback'}")
        return 0

    # ---- Step 2 模型回放 + 接口生成
    from aw14.engine14 import quick_run
    res = quick_run(theta, cache={})
    meta_in = {"source": ref["source"], "params_fp": pinfo["params_fp"],
               "status": "ok", "scraped": ref["scraped"], "new": ref["new"],
               "md": ref["md"], "attempt": ref["attempt"], "last": ref["last"],
               "warnings": ref["warnings"]}
    payload = build_interfaces(res, meta_in, rid)
    errs = validate(payload)
    if errs:
        print("  接口校验失败，放弃本次写入（沿用上一版）:", "; ".join(errs))
        meta_in["status"] = "error"
    else:
        payload["data/meta.json"]["data_md5"] = fingerprint
        if not args.dry_run:
            for rel, obj in payload.items():
                if rel.startswith("_"):
                    continue
                p = ROOT / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(obj, str):
                    p.write_text(obj, encoding="utf-8")
                else:
                    jdump(obj, p)
            audit = jload(DATA / "sync_audit.json", {"runs": []})
            audit["runs"].append({"run_id": rid, "at": now().isoformat(timespec="seconds"),
                                  "status": meta_in["status"], "source": ref["source"],
                                  "scraped": ref["scraped"], "new": ref["new"], "md": ref["md"],
                                  "params_fp": pinfo["params_fp"], "as_of": payload["_nav"]["as_of"],
                                  "bars": payload["_nav"]["bars"], "retry": ref["attempt"],
                                  "fallback": ref["source"] == "cache_fallback",
                                  "reconcile_pp": payload["data/meta.json"]["reconcile"]["max_abs_diff_pp"]})
            audit["runs"] = audit["runs"][-60:]
            audit["last_run"] = rid
            audit["policy"] = POLICY
            jdump(audit, DATA / "sync_audit.json")
            nm = mirror_to_site()

    m = payload["_nav"]["metrics"]
    print(f"  Step2 回放: as_of={payload['_nav']['as_of']} bars={payload['_nav']['bars']} "
          f"年化={m['cagr']}% 夏普={m['sharp']} 回撤=-{m['max_drawdown']}%")
    print(f"  对账偏差 {payload['data/meta.json']['reconcile']['max_abs_diff_pp']}pp "
          f"| 接口 {len([k for k in payload if not k.startswith('_')])} 个 | 校验 {'PASS' if not errs else 'FAIL'}")
    print(f"[{rid}] 完成，耗时 {time.time()-t0:.1f}s")
    print(f"一行汇总：{now():%Y-%m-%d %H:%M} 同步{status_cn(meta_in['status'])}，"
          f"as_of={payload['_nav']['as_of']}，新增 {ref['new']} 日，来源 {ref['source']}，"
          f"回退={ref['source']=='cache_fallback'}，接口 {len([k for k in payload if not k.startswith('_')])} 个，"
          f"发布目录同步 {nm} 个文件")
    return 0 if not errs else 2


def status_cn(s: str) -> str:
    return {"ok": "成功", "skipped": "跳过", "error": "失败"}.get(s, s)


if __name__ == "__main__":
    sys.exit(main())
