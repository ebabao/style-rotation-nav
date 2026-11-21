# -*- coding: utf-8 -*-
"""第29章29.2 · 零前视 walk-forward 验证因子预警力（诚实）

方法（对齐文档）：
  标签 = 未来 10 日市场组合收益 ≤ thr（默认 -8%），仅训练/评估用。
  扩张窗口 walk-forward：用 [0,k) 训练逻辑回归预测 k 日起的样本（严格因果）。
  OOF AUC：所有时间外预测拼在一起算 AUC。
  干净时间外：训练 2013-2021 → 测试 2022-2026（一次性切分）。
  跨阈值稳健：thr ∈ {-10%, -6%, -8%}，看因子是否全部 AUC>0.55。
  多因子模型：现有因子 vs GitHub 新因子 vs 稳健新因子6，OOF AUC 对比。

零前视：因子值 f[t] 在 t 收盘可得；标签用 t+1~t+10 收益；训练只用 k 之前数据。
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from timing_data import load_config, dump_json, OUT_DATA  # noqa: E402
from git_factors import build_factors, forward_drop_label, SOURCE, load_market_panel  # noqa: E402

CFG = load_config()
REG = CFG["regulator"]
MIN_TRAIN = 900
STEP = 63
EXISTING = ["mkt_er", "lowvol_state"]   # 现有因子（regime_signals 里的），作为对照组


def _existing_factors():
    """现有 regime 因子（文档中的 stress/mkt_er 对照），从市场 close 构造。"""
    from timing_regulator import regime_signals
    sigs = regime_signals(load_market_panel()["asset_close"])
    return pd.DataFrame({k: v for k, v in sigs.items()})


def oof_auc(F: pd.DataFrame, lab: pd.Series, cols: list,
            start=MIN_TRAIN, step=STEP, clean_split=None) -> dict:
    """扩张窗口 walk-forward OOF AUC。clean_split 给定则只在断点训练一次。"""
    X = F[cols]
    y = lab
    preds = []
    if clean_split is not None:
        cut = clean_split
        tr = (X.index < cut) & y.notna() & X.notna().all(axis=1)
        te = (X.index >= cut) & y.notna() & X.notna().all(axis=1)
        if tr.sum() < 100 or te.sum() < 20:
            return dict(auc=np.nan, n_test=int(te.sum()))
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        return dict(auc=float(roc_auc_score(y[te], p)), n_train=int(tr.sum()), n_test=int(te.sum()))
    idx = X.index
    for k in range(start, len(X) - 1, step):
        k2 = min(k + step, len(X))
        tr_pos = np.arange(0, k)
        te_pos = np.arange(k, k2)
        tr_idx, te_idx = idx[tr_pos], idx[te_pos]
        tr = X.loc[tr_idx].notna().all(axis=1) & y.loc[tr_idx].notna()
        te = X.loc[te_idx].notna().all(axis=1) & y.loc[te_idx].notna()
        if tr.sum() < 100 or te.sum() < 5 or y.loc[te_idx[te]].nunique() < 2:
            continue
        sc = StandardScaler().fit(X.loc[tr_idx[tr]])
        clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(
            sc.transform(X.loc[tr_idx[tr]]), y.loc[tr_idx[tr]])
        p = clf.predict_proba(sc.transform(X.loc[te_idx[te]]))[:, 1]
        preds.append(pd.Series(p, index=te_idx[te]))
    if not preds:
        return dict(auc=np.nan, n_test=0)
    P = pd.concat(preds)
    Y = y.reindex(P.index)
    ok = Y.notna()
    return dict(auc=float(roc_auc_score(Y[ok], P[ok])), n_test=int(ok.sum()))


def single_factor_auc(F: pd.DataFrame, lab: pd.Series, col: str,
                      clean_split=None) -> dict:
    """单因子方向 AUC（用因子值本身作打分，统一方向：AUC<0.5 取 1-AUC 后为正贡献）。"""
    x = F[col]
    ok = x.notna() & lab.notna()
    if clean_split is not None:
        ok = ok & (x.index >= clean_split)
    if ok.sum() < 20 or lab[ok].nunique() < 2:
        return dict(auc=np.nan, dir_auc=np.nan, n=int(ok.sum()))
    a = roc_auc_score(lab[ok], x[ok])
    return dict(auc=float(a), dir_auc=float(max(a, 1 - a)), n=int(ok.sum()))


def run_validation():
    panel = load_market_panel()
    F = build_factors(panel)
    ext = _existing_factors()
    # 对齐到 F 的日历
    ext = ext.reindex(F.index)
    All = pd.concat([F, ext], axis=1)
    lab8 = forward_drop_label(panel["close"], h=10, thr=-0.08)
    clean = pd.Timestamp("2022-01-01")

    print("=== 单因子 OOF AUC（扩张窗口方向AUC，标签未来10日≤-8%）===")
    singles = {}
    for c in F.columns:
        if c in SOURCE:
            r = single_factor_auc(F, lab8, c)
            singles[c] = dict(**r, source=SOURCE[c])
            print(f"  {c:16s} dirAUC {r['dir_auc']:.3f}  原始AUC {r['auc']:.3f}  n={r['n']}")
    # 现有因子对照
    for c in EXISTING:
        if c in ext.columns:
            r = single_factor_auc(ext, lab8, c)
            singles[c] = dict(**r, source="现有因子对照")
            print(f"  {c:16s} dirAUC {r['dir_auc']:.3f}  (现有对照)  n={r['n']}")

    print("\n=== 多因子模型 OOF AUC ===")
    gh_new = [c for c in ("fip252", "kurt20", "m5_z", "tsharpe", "diffusion20", "vol_contraction") if c in F.columns]
    robust6 = [c for c in ("fip252", "kurt20", "m5_z", "tsharpe", "diffusion20", "autocorr21") if c in F.columns]
    m_exist = oof_auc(All, lab8, [c for c in EXISTING if c in ext.columns])
    m_gh = oof_auc(All, lab8, gh_new)
    m_rb = oof_auc(All, lab8, robust6)
    print(f"  仅现有因子      OOF AUC {m_exist['auc']:.3f} (n={m_exist['n_test']})")
    print(f"  GitHub新因子6   OOF AUC {m_gh['auc']:.3f} (n={m_gh['n_test']})")
    print(f"  稳健新因子6     OOF AUC {m_rb['auc']:.3f} (n={m_rb['n_test']})")

    print("\n=== 干净时间外检验（训练2013-2021 → 测试2022-2026）===")
    hold = {}
    for c in ["fip252", "kurt20", "m5_z", "tsharpe", "diffusion20"]:
        if c in F.columns:
            r = single_factor_auc(F, lab8, c, clean_split=clean)
            hold[c] = r
            print(f"  {c:16s} 时间外 dirAUC {r['dir_auc']:.3f} (n={r['n']})")
    hold_gh = oof_auc(All, lab8, gh_new, clean_split=clean)
    hold_exist = oof_auc(All, lab8, [c for c in EXISTING if c in ext.columns], clean_split=clean)
    print(f"  GitHub新因子6   时间外 AUC {hold_gh['auc']:.3f}")
    print(f"  仅现有因子      时间外 AUC {hold_exist['auc']:.3f}")

    print("\n=== 跨阈值稳健性（三档 -10/-6/-8%，单因子是否全部>0.55）===")
    robust = {}
    for c in ["fip252", "kurt20", "m5_z", "tsharpe", "diffusion20"]:
        aucs = []
        for thr in (-0.10, -0.06, -0.08):
            lab = forward_drop_label(panel["close"], h=10, thr=thr)
            aucs.append(single_factor_auc(F, lab, c)["dir_auc"])
        robust[c] = dict(aucs=[round(float(a), 3) for a in aucs],
                         all_above_55=bool(all(a > 0.55 for a in aucs if np.isfinite(a))))
        print(f"  {c:16s} {[round(a,3) for a in aucs]}  全部>0.55: {robust[c]['all_above_55']}")

    out = dict(
        method=dict(label="未来10日组合收益≤-8%", oof="扩张窗口walk-forward",
                    clean_split="2013-2021训练→2022-2026测试", min_train=MIN_TRAIN, step=STEP),
        n_factors=len(F.columns), n_events=int(lab8.sum()), n_samples=int(lab8.notna().sum()),
        event_rate=round(float(lab8.sum() / lab8.notna().sum()), 4),
        singles=singles,
        multi_model=dict(existing=m_exist, github_new=m_gh, robust6=m_rb),
        holdout=dict(singles=hold, github_new=hold_gh, existing=hold_exist),
        robustness=robust,
        note="事件样本含黄金/纳指组合，波动大于文档A股宽基面板，事件率高于文档23例；结果诚实披露。",
    )
    dump_json(out, OUT_DATA / "git_factor_validation.json")
    # 看板接口（因子表 + 模型对比 + 稳健性）
    dump_json(dict(validation=out), OUT_DATA / "git_factor_report.json")
    print("\nSAVED git_factor_validation.json / git_factor_report.json")
    return out


if __name__ == "__main__":
    run_validation()
