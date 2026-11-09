"""v1.4 七只真实 ETF · 全天候引擎（进化学习版）

相对 v1.3 的关键改动：
  * universe 仅 7 只真实 ETF，四宫格 EQ(4)/BD(1)/GD(1)/CM(1)；无指数代理、无拼接
  * 参数从「大类预算」下沉到「逐资产仓位倾斜 tilt」：类内权重 = 逆波动率 × tilt 后归一
  * 国债(TB5)大类专属杠杆 bond_lev；融资腿按年化 2% 计息（保守，不免费加杠杆）
  * 回撤刹车：滚动 1 年高点回撤超阈值按比例降杠杆（避免历史最高净值锁死）
  * 全历史(2013-)供协方差预热，业绩只统计 7 只齐全的评估期(2019-12-06 起)

严格无前视：t 日收盘后用截至 t 的数据估计协方差，按 t 收盘价建仓，t+1 起由份额承担收益。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from all_weather_prototype import blended_cov, metrics, solve_risk_budget  # noqa: E402
from aw14.data14 import load_config  # noqa: E402

OUT = ROOT / "aw14_out"
DATA = OUT / "data"
REPORTS = OUT / "reports"

QUADRANTS = {
    "EQ": ["DIVLV", "SP500", "NDX", "GEM50"],
    "BD": ["TB5"],
    "GD": ["GOLD"],
    "CM": ["SOY"],
}
QLIST = list(QUADRANTS)
QN_CN = {"EQ": "股票", "BD": "债券(国债)", "GD": "黄金", "CM": "商品"}
ASSET_CN = {"TB5": "国债ETF", "GOLD": "黄金ETF", "SOY": "豆粕ETF",
            "DIVLV": "红利低波", "SP500": "标普500", "NDX": "纳指100", "GEM50": "创业板50"}
TILT_KEYS = ["TB5", "GOLD", "SOY", "DIVLV", "SP500", "NDX", "GEM50"]

# 15 维可学习参数：债券杠杆 / 四大类风险预算 / 7 只逐资产仓位倾斜
# / 整体风险缩放 / 再平衡带 / 回撤刹车
PARAM_NAMES = (["bond_lev", "b_eq", "b_bd", "b_gd", "b_cm"]
               + [f"tilt_{k}" for k in TILT_KEYS]
               + ["risk_scale", "band", "dd_brake"])
THETA0 = np.array([1.40, 0.30, 0.30, 0.20, 0.20,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.00, 0.03, 0.06], dtype=float)
LO = np.array([1.00, 0.15, 0.10, 0.08, 0.05,
               0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30,
               1.00, 0.01, 0.03], dtype=float)
HI = np.array([2.50, 0.50, 0.45, 0.35, 0.30,
               2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50,
               2.50, 0.06, 0.12], dtype=float)
SPAN = HI - LO

MIN_HISTORY = 120
COV_WINDOWS = (60, 120, 250)
COV_ALPHAS = (0.2, 0.3, 0.5)
SHRINKAGE = 0.20
MAX_EXPOSURE = 2.8
QUARTER_MONTHS = (1, 4, 7, 10)
FINANCE_ANNUAL = 0.02
FIN_DAILY = FINANCE_ANNUAL / 252.0
BRAKE_FLOOR = 0.60
EVAL_START = "2019-12-06"


def dump_json(obj, path: Path) -> None:
    import json

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


def unpack(theta: np.ndarray) -> dict:
    return dict(zip(PARAM_NAMES, np.clip(np.asarray(theta, float), LO, HI)))


class AllWeatherEngine:
    def __init__(self, cfg, R, P, A, theta, cache=None):
        self.cfg = cfg
        self.R, self.P, self.A = R, P, A
        self.cols = list(R.columns)
        self.n = len(self.cols)
        self.ci = {c: i for i, c in enumerate(self.cols)}
        self.dates = R.index
        self.T = len(R)
        self.Rv, self.Pv, self.Av = R.to_numpy(float), P.to_numpy(float), A.to_numpy(bool)
        self.theta = np.clip(np.asarray(theta, float), LO, HI)
        self.p = unpack(self.theta)
        self.cost_bps = float(cfg["data"]["cost_bps"])
        self.cache = cache if cache is not None else {}
        self.eval_i = int(np.searchsorted(self.dates, pd.Timestamp(EVAL_START)))
        self.q_start = np.array([
            i == 0 or (self.dates[i].month in QUARTER_MONTHS
                       and self.dates[i].month != self.dates[i - 1].month)
            for i in range(self.T)])
        self.budget_assets = self.cols[:]
        self.q_of = {c: q for q, mem in QUADRANTS.items() for c in mem}
        # 向量化预计算每日可用大类签名（与 theta 无关）
        W = MIN_HISTORY
        fin = np.isfinite(self.Rv).astype(np.int32)
        csum = np.vstack([np.zeros((1, self.n), np.int32), np.cumsum(fin, axis=0)])
        tt = np.arange(self.T)
        lower = np.maximum(0, tt - W + 1)
        win_sum = csum[tt + 1] - csum[lower]
        avail = self.Av & (win_sum == W) & ((tt >= W - 1)[:, None])
        self.uni_sig = [tuple(q for q in QLIST
                              if any(avail[t, self.ci[c]] for c in QUADRANTS[q]))
                        for t in range(self.T)]

    def available_cols(self, t: int) -> list[str]:
        h0 = max(0, t - MIN_HISTORY + 1)
        hist = self.Rv[h0: t + 1]
        ok = self.Av[t] & np.isfinite(hist).all(axis=0) & (len(hist) >= MIN_HISTORY)
        return [c for c in self.budget_assets if ok[self.ci[c]]]

    def _prepare_day(self, t: int) -> dict | None:
        """只用截至 t 的最近 MIN_HISTORY 天全 finite 数据，杜绝窗口内 NaN。"""
        if t in self.cache:
            return self.cache[t]
        cols = self.available_cols(t)
        by_q = {q: [c for c in cols if self.q_of.get(c) == q] for q in QLIST}
        present = [q for q in QLIST if by_q[q]]
        if len(present) < 3:                      # 至少三大类才构成全天候
            self.cache[t] = None
            return None
        h0 = max(0, t - MIN_HISTORY + 1)
        js = [self.ci[c] for c in cols]
        sub_all = pd.DataFrame(self.Rv[h0: t + 1][:, js], columns=cols)
        cov_all = blended_cov(sub_all, windows=COV_WINDOWS, alphas=COV_ALPHAS,
                              shrinkage=SHRINKAGE)
        vol_all = np.sqrt(np.maximum(np.diag(cov_all), 1e-16))
        cls_ret = {q: sub_all[by_q[q]].mean(axis=1).to_numpy() for q in present}
        cov_cls = blended_cov(pd.DataFrame(cls_ret), windows=COV_WINDOWS,
                              alphas=COV_ALPHAS, shrinkage=SHRINKAGE)
        invvol = {}
        for q in present:
            v = np.array([vol_all[cols.index(c)] for c in by_q[q]])
            iv = 1.0 / np.maximum(v, 1e-8)
            invvol[q] = iv / iv.sum()
        out = {"cols": cols, "js": js, "by_q": by_q, "present": present,
               "cov_all": cov_all, "cov_cls": cov_cls, "invvol": invvol}
        self.cache[t] = out
        return out

    def target_weights(self, prep, t, w_actual, dd) -> dict:
        p = self.p
        cols, js = prep["cols"], prep["js"]
        present, by_q = prep["present"], prep["by_q"]
        raw = {"EQ": p["b_eq"], "BD": p["b_bd"], "GD": p["b_gd"], "CM": p["b_cm"]}
        b = np.array([raw[q] for q in present])
        b = b / b.sum()                            # 缺席大类剔除后重归一，不编造
        W_cls, _, _, erc_info = solve_risk_budget(prep["cov_cls"], b)
        w_cls_map = dict(zip(present, W_cls))

        w = np.zeros(self.n)
        for q in present:
            # 逐资产仓位倾斜：逆波动率 × tilt 后类内归一
            tilt = np.array([p[f"tilt_{c}"] for c in by_q[q]])
            wq = prep["invvol"][q] * tilt
            wq = wq / max(wq.sum(), 1e-12)
            for k, c in enumerate(by_q[q]):
                w[self.ci[c]] = w_cls_map[q] * wq[k]

        # 国债大类专属杠杆（只放大债券，其余不变）
        for c in by_q.get("BD", []):
            w[self.ci[c]] *= float(p["bond_lev"])
        # 整体风险缩放：在控制回撤的前提下放大/收缩名义敞口以提升收益
        w *= float(p["risk_scale"])

        # 回撤刹车：滚动 1 年高点回撤超阈值 → 按比例降杠杆，最低 BRAKE_FLOOR
        brake = 1.0
        if dd < -float(p["dd_brake"]):
            brake = max(BRAKE_FLOOR, 1.0 - 1.5 * (abs(dd) - float(p["dd_brake"])))
            w *= brake

        L = float(w.sum())
        if L > MAX_EXPOSURE:
            w *= MAX_EXPOSURE / L
            L = MAX_EXPOSURE
        changed = bool(np.max(np.abs(w - w_actual)) > p["band"])
        # 大类风险贡献（Euler）
        w_sub = w[js]
        cov = prep["cov_all"]
        sigma = float(np.sqrt(max(w_sub @ cov @ w_sub, 1e-16)))
        rc = w_sub * (cov @ w_sub) / sigma
        rc_q = {q: float(sum(rc[cols.index(c)] for c in by_q[q]) / sigma) for q in present}
        W_actual = {q: float(sum(w[self.ci[c]] for c in by_q.get(q, []))) for q in QLIST}
        return {"w": w, "L": L, "changed": changed, "brake": brake,
                "W_actual": W_actual, "budget": dict(zip(present, b.tolist())),
                "rc_class": rc_q, "erc_dev": erc_info["max_deviation"],
                "present": present}

    def run(self) -> dict:
        nav_arr = np.full(self.T, np.nan)
        shares = np.zeros(self.n)
        cash, cash_growth = 1.0, 1.0
        cur_w = np.zeros(self.n)
        nav_arr[0] = 1.0
        logs, weight_rows, rc_rows = [], [], []
        last_universe, warm = None, False

        for t in range(1, self.T):
            pt = self.Pv[t]
            valid = np.isfinite(pt)
            port = float(np.dot(shares, np.where(valid, pt, 0.0)))
            cash_growth *= (1.0 + FIN_DAILY)       # 融资腿计息（cash<0 时等价于付息）
            nav_arr[t] = port + cash * cash_growth
            w_actual = (shares * np.where(valid, pt, 0.0) / nav_arr[t]
                        if nav_arr[t] > 0 else np.zeros(self.n))
            # 滚动 1 年高点回撤
            lo = max(0, t - 252)
            peak = float(np.nanmax(nav_arr[lo: t + 1]))
            dd = nav_arr[t] / peak - 1.0

            universe = self.uni_sig[t]
            force = (not warm) or (last_universe is not None and universe != last_universe)
            if not (self.q_start[t] or force):
                last_universe = universe
                continue
            prep = self._prepare_day(t)
            if prep is None:
                last_universe = universe
                continue
            sig = self.target_weights(prep, t, w_actual, dd)
            if not sig["changed"] and not force:
                continue
            w_full = sig["w"]
            turnover = float(np.abs(w_full - cur_w).sum() / 2.0)
            cost = turnover * self.cost_bps / 10000.0 * 2.0
            nav_arr[t] -= cost
            new_shares = np.zeros(self.n)
            for j in prep["js"]:
                px = float(self.Pv[t, j])
                if np.isfinite(px) and px > 0:
                    new_shares[j] = nav_arr[t] * w_full[j] / px
            cash = nav_arr[t] * (1.0 - sig["L"])
            cash_growth = 1.0
            shares, cur_w = new_shares, w_full.copy()
            warm, last_universe = True, universe

            if t < self.eval_i:                   # 预热段建仓但不计入评估期日志
                continue
            row = {"date": str(self.dates[t].date()), "nav": round(float(nav_arr[t]), 6),
                   "L": round(sig["L"], 4), "bond_lev": round(float(self.p["bond_lev"]), 3),
                   "brake": round(sig["brake"], 3),
                   "turnover": round(turnover, 4), "cost": round(cost, 6),
                   **{f"w_{c}": round(float(w_full[self.ci[c]]), 5) for c in self.cols}}
            logs.append(row)
            weight_rows.append({"date": self.dates[t],
                                **{c: float(w_full[self.ci[c]]) for c in self.cols},
                                "L": sig["L"]})
            rc_rows.append({"date": self.dates[t],
                            **{f"rc_{q}": sig["rc_class"].get(q, 0.0) for q in QLIST},
                            **{f"W_{q}": sig["W_actual"].get(q, 0.0) for q in QLIST}})

        nav = pd.Series(nav_arr, index=self.dates, name="nav").ffill()
        nav_eval = nav.iloc[self.eval_i:].copy()
        nav_eval = nav_eval / nav_eval.iloc[0]
        logs_df = pd.DataFrame(logs)
        weights_df = pd.DataFrame(weight_rows).set_index("date") if weight_rows else pd.DataFrame()
        rc_df = pd.DataFrame(rc_rows).set_index("date") if rc_rows else pd.DataFrame()
        rc_mean = {q: float(rc_df[f"rc_{q}"].mean()) for q in QLIST} if len(rc_df) else {}
        W_mean = {q: float(rc_df[f"W_{q}"].mean()) for q in QLIST} if len(rc_df) else {}
        wmean_asset = ({c: float(weights_df[c].mean()) for c in self.cols}
                       if len(weights_df) else {})
        return {"nav": nav_eval, "nav_full": nav,
                "metrics": metrics(nav_eval),
                "annual": self._annual_returns(nav_eval), "monthly": self._monthly_returns(nav_eval),
                "logs": logs_df, "weights": weights_df, "rc": rc_df,
                "rc_mean": rc_mean, "W_mean": W_mean, "wmean_asset": wmean_asset,
                "L_mean": float(logs_df["L"].mean()) if len(logs_df) else 0.0,
                "theta": self.theta.copy(), "params": self.p,
                "n_rebalance": len(logs),
                "avg_turnover": float(logs_df["turnover"].mean()) if len(logs_df) else 0.0}

    @staticmethod
    def _annual_returns(nav):
        y = nav.resample("YE").last()
        prev = float(nav.iloc[0])
        out = {}
        for dt, v in y.items():
            out[dt.year] = float(v / prev - 1.0)
            prev = float(v)
        return pd.Series(out)

    @staticmethod
    def _monthly_returns(nav):
        return nav.resample("ME").last().pct_change().dropna()


def load_inputs(data_dir: Path | None = None):
    d = data_dir or DATA
    R = pd.read_parquet(d / "clean_returns.parquet")
    P = pd.read_parquet(d / "clean_prices.parquet")
    A = pd.read_parquet(d / "available.parquet")
    return R, P, A


def make_engine(theta=THETA0, cache=None, cfg=None) -> AllWeatherEngine:
    R, P, A = load_inputs()
    return AllWeatherEngine(cfg or load_config(), R, P, A,
                            np.asarray(theta, float), cache=cache)


def quick_run(theta=THETA0, cache=None) -> dict:
    return make_engine(theta, cache=cache).run()


def window_metrics(nav: pd.Series, start: str, end: str) -> dict:
    s = nav.loc[start: end]
    if len(s) < 30:
        return {}
    return metrics(s / s.iloc[0])


if __name__ == "__main__":
    import time
    t0 = time.time()
    cache = {}
    res = quick_run(THETA0, cache=cache)
    print(f"=== v1.4 基线 {time.time()-t0:.1f}s | 缓存检查日 {len(cache)} | 评估期 {len(res['nav'])} 天 ===")
    for k, v in res["metrics"].items():
        print(f"  {k:14s}: {v*100:.2f}%" if isinstance(v, float) and k not in
              ("夏普比率", "Calmar", "期末净值", "回撤修复交易日") else f"  {k:14s}: {v}")
    print("  平均总敞口 L =", round(res["L_mean"], 3), "| 调仓", res["n_rebalance"],
          "| 平均换手", round(res["avg_turnover"] * 100, 2), "%")
    print("  7只平均仓位:", {ASSET_CN[c]: round(v * 100, 1) for c, v in res["wmean_asset"].items()})
    print("  大类风险贡献:", {QN_CN[q]: round(v * 100, 1) for q, v in res["rc_mean"].items()})
