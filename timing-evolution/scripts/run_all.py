#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""一键复现：数据审计 → 冻结基线对账 → Q-learning 监管层 → 看板生成。

机构版主线为移植 sr-nav/evolution_guard 的 Q-learning 监管范式。
GA 三段进化（timing_learn.py）作为可选的离线参数研究保留，不参与主线。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def step(title, args, required=True):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)
    r = subprocess.run([PY, *args], cwd=ROOT)
    if r.returncode != 0 and required:
        print(f"[失败] {title}（exit {r.returncode}）", file=sys.stderr)
        sys.exit(r.returncode)


if __name__ == "__main__":
    step("[1/6] 数据审计（哈希落盘）", ["-m", "model.timing_data"])
    step("[2/6] 冻结基线对账（32.78% / 去漂移16.93%）", ["-m", "model.timing_engine"])
    step("[3/6] Q-learning 监管层（影子台账 / 信誉Q / 预警）",
         ["-m", "model.timing_regulator"])
    step("[4/6] 第29章 GitHub 因子库自检", ["-m", "model.git_factors"])
    step("[5/6] 第29章 零前视 walk-forward 验证 + 触发器对照回测",
         ["-m", "model.git_factor_validate"])
    step("[5/6b] 第29章 高AUC≠好触发器回测", ["-m", "model.git_factor_backtest"])
    step("[6/6] 生成看板 + 第29章研究报告",
         ["scripts/build_page.py"])
    step("[6/6b] 生成第29章报告", ["scripts/build_git_report.py"])
    print("\n[完成] 全部流程通过；打开 index.html（Q-learning 监管看板）与 第29章GitHub择时因子研究报告.html。")
    print("（可选）离线 GA 三段进化：python3.11 -m model.timing_learn")
