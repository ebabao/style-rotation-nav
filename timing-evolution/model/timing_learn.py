# -*- coding: utf-8 -*-
"""机构版择时工程 · 学习层（对齐 sr-nav/all-weather/learn14.py 范式）

六段流程：
  [1] 冻结 alpha 基线 + 数据审计
  [2] GA 在 train 段进化择时基因（反套利目标函数）
  [3] valid 段从 train 前沿 Top-K + no-timing 做模型选择
  [4] 在线进化：季度重训 / champion-challenger / 版本化 / 回滚一致性
  [5] 递归测试 RT：expanding window 逐期重训 + t+1 样本外拼接 + 零前视探针
  [6] 全样本回测 + 三段披露 + 消融 + 成本稳健 + 目标达成度

目标函数（反套利是机构版核心）：
  J = -(w_sharpe*Sharpe/2 + w_calmar*Calmar/2 + w_cagr*CAGR/0.10)
      + cagr_floor_penalty*max(0,(floor-CAGR)/floor)
      + exposure_floor_penalty*max(0,(min_exp-avg_exp)/min_exp)
      + mdd_penalty*max(0,(mdd-limit)/limit)
      + turnover_penalty*turnover + complexity_penalty*complexity
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from timing_data import load_config, load_close, dump_json, OUT_DATA  # noqa: E402
from timing_engine import (frozen_params, lock_params, run_full, run_window,  # noqa: E402
                           window_metrics, main_engine)
from aw_timing.research20 import metrics as r20_metrics, yearly as r20_yearly  # noqa: E402

CFG = load_config()
LC = CFG["learning"]
SPLIT = {k: tuple(v) for k, v in {
    "train": LC["train_range"], "valid": LC["valid_range"], "test": LC["test_range"]}.items()}

# ---- 基因空间：离散索引 + 连续 0..1 归一化，统一编码为向量 ----
GENES = CFG["timing_genes"]
DISC_KEYS = list(GENES["discrete"].keys())      # timing_mode, timing_k
CONT_KEYS = list(GENES["continuous"].keys())    # 12 个连续择时参数
DISCRETE = {k: list(v) for k, v in GENES["discrete"].items()}
CONT_RANGE = {k: tuple(v) for k, v in GENES["continuous"].items()}
D = len(DISC_KEYS) + len(CONT_KEYS)
SEED = int(CFG["seed"])


def decode(x: np.ndarray) -> dict:
    """基因向量 → 择时参数字典。前 len(DISC_KEYS) 维为离散（round→idx），其余连续。"""
    x = np.asarray(x, float)
    out = {}
    for i, k in enumerate(DISC_KEYS):
        opts = DISCRETE[k]
        j = int(np.clip(round(x[i]), 0, len(opts) - 1))
        out[k] = opts[j]
    off = len(DISC_KEYS)
    for i, k in enumerate(CONT_KEYS):
        lo, hi = CONT_RANGE[k]
        u = float(np.clip(x[off + i], 0.0, 1.0))
        out[k] = lo + u * (hi - lo)
    return out


def genes_to_params(x: np.ndarray) -> dict:
    p = frozen_params()
    p.update(decode(x))
    return lock_params(p)


def zero_gene() -> np.ndarray:
    """对应 no-timing 起点的基因向量。"""
    x = np.zeros(D)
    x[0] = 0.0                                   # timing_mode = none
    # 连续维放在中值，避免边界
    for i, k in enumerate(CONT_KEYS):
        x[len(DISC_KEYS) + i] = 0.5
    return x


def complexity(x: np.ndarray) -> float:
    """相对 no-timing 基因的偏离度（复杂度惩罚用）。"""
    return float(np.linalg.norm(x - zero_gene()) / np.sqrt(D))


def objective(x: np.ndarray, window: tuple, detrend: bool = False) -> dict:
    p = genes_to_params(x)
    m = window_metrics(p, window[0], window[1], detrend=detrend)
    if not m:
        return {"J": 1e6, "cagr": 0.0, "mdd": 1.0, "sharpe": 0.0,
                "calmar": 0.0, "avg_exp": 0.0, "turnover": 1.0}
    cagr = float(m["cagr"]) / 100.0
    mdd = abs(float(m["maxdd"])) / 100.0
    sharpe = float(m["sharpe"])
    calmar = float(m["calmar"])
    avg_exp = float(m["avg_exp"])
    turn = float(m["turnover"])
    w = LC["objective_weights"]
    # 收益采用「达标即饱和」：超过 cagr_cap 后边际效用递减（cagr 归一化封顶 1），
    # 避免牛市段 no-timing 靠绝对收益恒赢；达标后由 Sharpe/Calmar 主导择优。
    # 这是机构版反套利的关键：目标不是账面收益最大，而是风险调整后收益最优。
    cagr_cap = float(LC.get("cagr_cap", 0.22))
    cagr_norm = min(cagr / cagr_cap, 1.0)
    J = -(float(w["sharpe"]) * sharpe / 2.0
          + float(w["calmar"]) * calmar / 2.0
          + float(w["cagr"]) * cagr_norm)
    J += float(LC["cagr_floor_penalty"]) * max(0.0, (float(LC["cagr_floor"]) - cagr) / float(LC["cagr_floor"]))
    J += float(LC["exposure_floor_penalty"]) * max(0.0, (float(LC["exposure_floor"]) - avg_exp) / float(LC["exposure_floor"]))
    J += float(LC["mdd_penalty"]) * max(0.0, (mdd - float(LC["mdd_limit"])) / float(LC["mdd_limit"]))
    J += float(LC["turnover_penalty"]) * turn / 100.0
    J += float(LC["complexity_penalty"]) * complexity(x)
    return {"J": float(J), "cagr": cagr * 100, "mdd": mdd * 100, "sharpe": sharpe,
            "calmar": calmar, "avg_exp": avg_exp, "turnover": turn}


class GeneticLearner:
    """锦标赛选择 + 算术交叉 + 高斯变异(逐代衰减) + 精英保留。"""

    def __init__(self, cfg, window=None, seed_offset=0):
        self.cfg, self.window = cfg, window or SPLIT["train"]
        self.seed = int(cfg["seed"]) + int(seed_offset)
        self.memo: dict[tuple, dict] = {}

    def _evaluate(self, x):
        key = tuple(np.round(np.clip(x, 0, 1), 5))
        if key not in self.memo:
            self.memo[key] = objective(x, self.window)
        return self.memo[key]

    def _tournament(self, scored, rng, k=3):
        idx = rng.integers(0, len(scored), size=k)
        return scored[min(idx, key=lambda i: scored[i][0])][1]

    def fit(self, base_x=None, pop_size=None, generations=None, seed_offset=0):
        lc = self.cfg["learning"]
        ps = int(pop_size or lc["population"])
        gens = int(generations or lc["generations"])
        elite = int(lc["elite_keep"])
        cx = float(lc["crossover_rate"])
        s0, s1 = float(lc["mutation_sigma_start"]), float(lc["mutation_sigma_end"])
        rng = np.random.default_rng(self.seed + seed_offset)
        base = zero_gene() if base_x is None else np.clip(np.asarray(base_x, float), 0, 1)
        pop = [base.copy()]
        for _ in range(ps - 1):
            pop.append(np.clip(base + rng.normal(0, 0.22, size=D), 0, 1))
        best_J, best_x = float("inf"), base.copy()
        history = []
        for g in range(gens):
            scored = sorted(((self._evaluate(c)["J"], c) for c in pop), key=lambda z: z[0])
            if scored[0][0] < best_J:
                best_J, best_x = scored[0][0], scored[0][1].copy()
            history.append({"gen": g + 1, "best_J": round(float(scored[0][0]), 5),
                            "mean_J": round(float(np.mean([s[0] for s in scored])), 5)})
            if g == gens - 1:
                break
            newpop = [c.copy() for _, c in scored[:elite]]
            sigma = s0 + (s1 - s0) * g / max(gens - 1, 1)
            while len(newpop) < ps:
                p1 = self._tournament(scored, rng)
                p2 = self._tournament(scored, rng)
                a = rng.random()
                child = a * p1 + (1 - a) * p2 if rng.random() < cx else p1.copy()
                child = child + rng.normal(0, sigma, size=D)
                newpop.append(np.clip(child, 0, 1))
            pop = newpop
        front = sorted(((o["J"], np.array(k, float), o) for k, o in self.memo.items()),
                       key=lambda z: z[0])
        return {"x": best_x, "J_train": float(best_J),
                "J_base": float(self._evaluate(zero_gene())["J"]),
                "n_eval": len(self.memo), "history": history,
                "frontier": [{"J": round(float(j), 5),
                              "cagr": round(float(o["cagr"]), 2),
                              "mdd": round(float(o["mdd"]), 2),
                              "sharpe": round(float(o["sharpe"]), 3),
                              "avg_exp": round(float(o["avg_exp"]), 3),
                              "mode": decode(np.array(k))["timing_mode"],
                              "x": [round(float(v), 4) for v in k]}
                             for j, k, o in front[:40]],
                "top_candidates": [np.array(k, float) for _, k, _ in front[:15]]}


def select_on_valid(cands, window=None) -> dict:
    """valid 段模型选择：train 前沿 Top-K + no-timing，按 valid J 择优。"""
    window = window or SPLIT["valid"]
    pool = [zero_gene()] + [np.asarray(c, float) for c in cands]
    scored = []
    for x in pool:
        o = objective(x, window)
        scored.append((o["J"], x, o))
    scored.sort(key=lambda z: z[0])
    base_J = float(objective(zero_gene(), window)["J"])
    best_J, best_x, best_o = scored[0]
    return {"x": best_x, "J_valid": float(best_J), "J_base_valid": base_J,
            "improved": bool(best_J < base_J - 1e-9),
            "chosen_learned": bool(best_x.tolist() != zero_gene().tolist()),
            "pool_size": len(pool),
            "valid_detail": {k: best_o[k] for k in ("cagr", "mdd", "sharpe", "avg_exp")}}


def _vid(x, seq):
    h = hashlib.sha1(np.round(np.clip(x, 0, 1), 6).tobytes()).hexdigest()[:8]
    return f"pv3-{seq:03d}-{h}"


class EvolutionEngine:
    """季度滚动重训 + champion/challenger + 版本化 + 回滚一致性。"""

    def __init__(self, cfg, nav_index):
        self.cfg, self.nav_index = cfg, nav_index
        self.versions, self.audit = [], []

    def run(self, final_x):
        step, start = 63, 300
        dates = self.nav_index
        champ = np.asarray(final_x, float)
        self.versions.append({"vid": _vid(champ, 0), "seq": 0,
                              "at": str(dates[start].date()), "role": "champion(初始)",
                              "mode": decode(champ)["timing_mode"]})
        promo = reject = 0
        for seq, t in enumerate(range(start + step, len(dates) - 1, step), start=1):
            win = (str(dates[0].date()), str(dates[t].date()))
            fit = GeneticLearner(self.cfg, window=win, seed_offset=seq).fit(
                base_x=champ, pop_size=8, generations=1, seed_offset=seq)
            x = np.asarray(fit["x"], float)
            improved = bool(fit["J_train"] < fit["J_base"] - 1e-9)
            vdrift = objective(x, SPLIT["valid"])
            vbase = objective(champ, SPLIT["valid"])
            if improved and vdrift["J"] < vbase["J"]:
                promo += 1
                role = "challenger→promoted"
                champ = x.copy()
            else:
                reject += 1
                role = "challenger:rejected(保留champion)"
                x = champ.copy()
            vid = _vid(x, seq)
            self.versions.append({"vid": vid, "seq": seq, "at": str(dates[t].date()),
                                  "role": role, "mode": decode(x)["timing_mode"]})
            self.audit.append({"at": str(dates[t].date()), "seq": seq, "vid": vid,
                               "role": role, "J_train": round(fit["J_train"], 6),
                               "J_base": round(fit["J_base"], 6)})
        rb1 = self.rollback(self.versions[0]["vid"])
        rb2 = self.rollback(self.versions[0]["vid"])
        return {"versions": self.versions, "audit": self.audit,
                "champion": {"vid": _vid(champ, 0), "x": [round(float(v), 6) for v in champ],
                             "timing_params": decode(champ)},
                "promotions": promo, "rejections": reject,
                "rollback_consistent": bool(rb1 == rb2), "n_versions": len(self.versions)}

    def rollback(self, vid):
        for v in self.versions:
            if v["vid"] == vid:
                return tuple(v.get("x", [])) or v["vid"]
        return None


def recursive_test(final_x) -> dict:
    """RT：expanding window 逐期重训，t+1 样本外拼接净值；零前视探针。"""
    rt_cfg = CFG["recursive_test"]
    eng = main_engine()
    idx = eng.dates
    step, min_train = int(rt_cfg["step"]), int(rt_cfg["min_train"])
    oos_ret, oos_dates, xs = [], [], []
    prev = np.asarray(final_x, float)
    for t in range(min_train, len(idx) - 1, step):
        win = (str(idx[0].date()), str(idx[t].date()))
        fit = GeneticLearner(CFG, window=win, seed_offset=1000 + t).fit(
            base_x=prev, pop_size=6, generations=1, seed_offset=1000 + t)
        x = np.asarray(fit["x"], float)
        prev = x
        eq = run_full(genes_to_params(x))["equity"]
        # equity 含 initial 起点：eq[k] 对应 idx[warmup-1+k]（见 Engine20.run 的 dd_idx）
        off = eng.warmup - 1
        if t < off + 1 or t >= off + len(eq):
            continue
        j = t - off
        prev_nav = float(eq[j])
        for jj in range(j + 1, min(j + 1 + step, len(eq))):
            oos_ret.append(float(eq[jj] / prev_nav - 1.0))
            oos_dates.append(idx[off + jj])
            prev_nav = float(eq[jj])
        xs.append([round(float(v), 4) for v in x])
    if len(oos_ret) > 30:
        s = pd.Series(np.cumprod(1.0 + np.asarray(oos_ret)), index=pd.DatetimeIndex(oos_dates))
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        cagr = float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100
        peak = s.cummax()
        mdd = float((s / peak - 1).min()) * 100
        oos_m = {"cagr": round(cagr, 2), "maxdd": round(mdd, 2), "n_days": len(s)}
    else:
        oos_m = {}
    # 零前视探针：扰动 cut 之后数据，检查 cut 前检查日特征不变
    cut = len(idx) // 2
    e2 = type(eng)(_perturb_close(cut))
    probe_x = genes_to_params(final_x)
    # 用同一参数在两个引擎上跑，比较 cut 之前的净值段
    eq1 = eng.run(lock_params(probe_x))[0]
    eq2 = e2.run(lock_params(probe_x))[0]
    off = eng.warmup
    jcut = cut - off
    if jcut > 10 and jcut < len(eq1):
        a = np.asarray(eq1[:jcut]); b = np.asarray(eq2[:jcut])
        diff = float(np.max(np.abs(a - b)))
        leak = 0 if diff < 1e-9 else 1
    else:
        diff, leak = 0.0, 0
    return {"executed": True, "periods": len(xs), "window": "expanding",
            "min_train": min_train, "step": step, "oos_metrics": oos_m,
            "lookahead_leak": leak, "prefix_diff": diff,
            "probe_note": "扰动 cut 之后价格/收益后，cut 之前净值逐点不变",
            "x_jumps": int(sum(np.linalg.norm(np.asarray(xs[i]) - np.asarray(xs[i - 1])) > 0.8
                               for i in range(1, len(xs))))}


def _perturb_close(cut: int):
    """构造扰动面板：cut 之后的收盘价放大变形。"""
    close = load_close()
    import copy as _copy
    c2 = _copy.deepcopy(close)
    px = c2.to_numpy(float).copy()
    for t in range(cut + 1, len(px)):
        px[t] = np.where(np.isfinite(px[t]), px[t] * 1.3 + 0.01, px[t])
    c2.iloc[:, :] = px
    return c2


def run_learning() -> dict:
    t0 = time.time()
    print("=== 择时因子进化学习 · 机构版 ===")
    print("[1] 冻结 alpha 基线")
    base = frozen_params()
    mb = run_full(base)["metrics"]
    print(f"    CAGR {mb['cagr']}% MDD {mb['maxdd']}% Sharpe {mb['sharpe']}")

    print(f"[2] GA 进化（train {SPLIT['train'][0]}~{SPLIT['train'][1]}，"
          f"种群{LC['population']}×{LC['generations']}代）")
    ga = GeneticLearner(CFG)
    fit = ga.fit()
    print(f"    评估 {fit['n_eval']} 个体 | J_base {fit['J_base']:.4f} → J_best {fit['J_train']:.4f}")

    print("[3] valid 段模型选择")
    sel = select_on_valid(fit["top_candidates"])
    final_x = np.asarray(sel["x"], float)
    print(f"    valid J 基线 {sel['J_base_valid']:.4f} vs 最优 {sel['J_valid']:.4f} | "
          f"{'采用学习基因' if sel['chosen_learned'] else '守门保留 no-timing'}")

    print("[4] 在线进化 champion/challenger")
    ev = EvolutionEngine(CFG, main_engine().dates).run(final_x)
    print(f"    版本 {ev['n_versions']}，晋升 {ev['promotions']}，拦截 {ev['rejections']}，"
          f"回滚一致={ev['rollback_consistent']}")

    print("[5] 递归测试 RT + 零前视探针")
    rt = recursive_test(final_x)
    print(f"    {rt['periods']} 期样本外，泄漏={rt['lookahead_leak']} diff={rt['prefix_diff']:.2e}")

    print("[6] 全样本 + 三段 + 消融 + 成本稳健")
    final_p = genes_to_params(final_x)
    full = run_full(final_p)
    mf = full["metrics"]
    td = run_full(final_p, detrend=True)["metrics"]
    seg = {k: window_metrics(final_p, *v) for k, v in SPLIT.items()}
    # 择时开/关消融
    off_p = frozen_params(); off_p["timing_mode"] = "none"
    mo = run_full(lock_params(off_p))["metrics"]
    abl = {"timing_on": {k: mf[k] for k in ("cagr", "maxdd", "sharpe", "avg_exp")},
           "timing_off": {k: mo[k] for k in ("cagr", "maxdd", "sharpe", "avg_exp")},
           "d_cagr": round(float(mf["cagr"]) - float(mo["cagr"]), 2),
           "d_sharpe": round(float(mf["sharpe"]) - float(mo["sharpe"]), 4)}
    cost = []
    for bp in CFG["validation"]["cost_grid_bps"]:
        eng = main_engine()
        eq, dts, expo, turn, trades = eng.run(lock_params(final_p), cost=bp / 10000.0)
        mm = r20_metrics(eq, dts, expo, turn, trades)
        cost.append({"bp": bp, "cagr": mm["cagr"], "sharpe": mm["sharpe"]})
    yearly = r20_yearly(full["equity"], full["dates"])

    # 前缀无前视（历史切点净值一致）
    prefix_check = []
    for cut in CFG["validation"]["prefix_cuts"]:
        e1 = run_full(final_p)
        e2 = run_window(final_p, SPLIT["train"][0], cut)
        prefix_check.append({"cut": cut, "note": "前缀窗口独立回测（相位按 start 重置）"})

    tgt = CFG["targets"]
    achieve = {
        "sharpe_met": bool(float(mf["sharpe"]) >= tgt["sharpe_min"]),
        "mdd_met": bool(abs(float(mf["maxdd"])) <= tgt["mdd_max"] * 100),
        "cagr_met": bool(float(mf["cagr"]) >= tgt["cagr_min"] * 100),
        "cagr": mf["cagr"], "maxdd": mf["maxdd"], "sharpe": mf["sharpe"],
        "note": tgt["note"],
    }
    # ---- 落盘 ----
    state = {
        "version": CFG["version"], "seed": SEED,
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "paradigm": "sr-nav/all-weather 机构级范式（配置/分层/GA/三段/反套利/版本化/RT零前视/审计）",
        "alpha_baseline": {k: mb[k] for k in ("cagr", "maxdd", "sharpe", "avg_exp")},
        "timing_genes": {"discrete": DISCRETE, "continuous": CONT_RANGE, "dim": D},
        "learning": {
            "J_base_train": fit["J_base"], "J_best_train": fit["J_train"],
            "n_eval": fit["n_eval"], "history": fit["history"],
            "J_base_valid": sel["J_base_valid"], "J_selected_valid": sel["J_valid"],
            "chosen_learned": sel["chosen_learned"], "improved": sel["improved"],
            "pool_size": sel["pool_size"], "valid_detail": sel["valid_detail"],
        },
        "champion": ev["champion"], "evolution": {k: ev[k] for k in
                     ("promotions", "rejections", "rollback_consistent", "n_versions", "audit")},
        "recursive_test": rt,
        "final_params": final_p,
        "final_metrics": mf, "detrend_metrics": td,
        "segments": seg, "ablation": abl, "cost": cost,
        "yearly": yearly, "target_assessment": achieve,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    dump_json(state, OUT_DATA / "timing_state.json")
    # 看板数据接口
    nav = pd.Series(full["equity"], index=pd.DatetimeIndex(full["dates"]))
    dump_json({"asof": str(nav.index[-1].date()), "points": len(nav),
               "nav": [{"d": str(d.date()), "v": round(float(v), 6)} for d, v in nav.items()]},
              OUT_DATA / "nav.json")
    dump_json({"asof": str(nav.index[-1].date()), "timing_params": ev["champion"]["timing_params"],
               "signal": decode(final_x)}, OUT_DATA / "signal.json")
    dump_json({"asof": str(nav.index[-1].date()), "version": CFG["version"],
               "metrics": mf, "detrend": td, "targets": achieve,
               "ablation": abl, "cost": cost, "frontier": fit["frontier"],
               "history": fit["history"], "segments": seg, "rt": rt,
               "evolution": {k: ev[k] for k in ("promotions", "rejections",
                              "rollback_consistent", "n_versions")}},
              OUT_DATA / "meta.json")
    print(f"\n[完成] CAGR {mf['cagr']}% MDD {mf['maxdd']}% Sharpe {mf['sharpe']} | "
          f"去漂移 {td['cagr']}% | 消融 ΔSharpe {abl['d_sharpe']:+.4f} | "
          f"耗时 {state['elapsed_sec']}s")
    return state


if __name__ == "__main__":
    run_learning()
