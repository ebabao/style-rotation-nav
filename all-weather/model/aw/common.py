"""全天候策略模型 v1.0 · 工程公共基础（路径 / 配置 / 哈希 / 中文字体 / 运行指标）"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent          # /workspace
PKG = Path(__file__).resolve().parent                  # /workspace/aw
OUT = ROOT / "aw_out"
DATA = OUT / "data"
RAW = DATA / "raw"
REPORTS = OUT / "reports"
RUNS = OUT / "runs"
CONFIG_SRC = PKG / "config_v1.yaml"
CONFIG_OUT = OUT / "config" / "data_v1.yaml"

CN_TZ = timezone(timedelta(hours=8))


def ensure_dirs() -> None:
    for p in (OUT, DATA, RAW, REPORTS, RUNS, CONFIG_OUT.parent):
        p.mkdir(parents=True, exist_ok=True)


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def stamp() -> str:
    return now_cn().strftime("%Y-%m-%d %H:%M:%S")


def run_id() -> str:
    t = now_cn().strftime("%Y%m%d-%H%M%S")
    return f"run-{t}-{hashlib.sha1(t.encode()).hexdigest()[:6]}"


def load_config() -> dict:
    with CONFIG_SRC.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict) -> None:
    CONFIG_OUT.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _json_default(o):
    """numpy 标量显式转换，避免 bool_ 被 default=str 序列化成字符串 "True"。"""
    import numpy as np
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                               default=_json_default) + "\n",
                    encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_hash(df) -> str:
    """DataFrame 内容哈希（含索引与列名），用于可复现性验收。"""
    buf = (df.to_csv(index=True).encode("utf-8")
           + repr(list(df.columns)).encode("utf-8")
           + repr(df.index.dtype).encode("utf-8"))
    return hashlib.sha256(buf).hexdigest()


def setup_matplotlib() -> None:
    """中文字体方案；找不到时由调用方降级为英文标签。"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
        "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = 130


class RunLog:
    """分阶段运行指标记录器：run_id / stage / status / evidence / retries / fallback。"""

    def __init__(self, rid: str):
        self.rid = rid
        self.started = stamp()
        self.stages: list[dict] = []

    def add(self, stage: str, status: str, **kw) -> None:
        self.stages.append({"stage": stage, "status": status, "at": stamp(), **kw})

    def save(self) -> None:
        dump_json({
            "run_id": self.rid, "started_at": self.started, "finished_at": stamp(),
            "stages": self.stages,
            "summary": {
                "total": len(self.stages),
                "passed": sum(s["status"] == "PASS" for s in self.stages),
                "blocked": sum(s["status"] == "BLOCKED" for s in self.stages),
                "failed": sum(s["status"] == "FAIL" for s in self.stages),
            },
        }, RUNS / f"{self.rid}.json")
