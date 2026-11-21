# -*- coding: utf-8 -*-
"""Vendored 引擎包：与研究工程 quant 的 v2.0 单引擎逐行一致（parity=0）。

仅改动 research20.py 的数据路径解析，使其从子工程 data/ 读取，
从而做到自包含、可独立部署，不依赖 /workspace 外部绝对路径。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
