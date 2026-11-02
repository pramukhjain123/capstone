"""Run every experiment in the paper and save raw results to results/raw/.

Usage:  python scripts/run_experiments.py [main|ablation|lookback|stability|all]
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cahrp.backtest import load_universe, run_backtest, universe_constraints, universe_names  # noqa
from cahrp.constraints import violations  # noqa
from cahrp.estimators import estimate  # noqa
from cahrp.hrp import hrp  # noqa
from cahrp import optim  # noqa
from cahrp.strategies import build_ablations, build_strategies, ca_hrp  # noqa

RAW = ROOT / "results" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def save(obj, name):
    with open(RAW / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def run_set(strategies, tag, lookback=252, universes=None):
    for u in universes or universe_names():
        rets, groups = load_universe(u)
        c = universe_constraints(u, groups)
        out = {}
        for name, fn in strategies.items():
            t = time.time()
            out[name] = run_backtest(rets, fn, c, lookback)
            print(f"[{tag}] {u:5s} {name:22s} {time.time() - t:6.1f}s", flush=True)
        save(out, f"{tag}_{u}")


def stability(n_boot=30, seed=7):
    """Weight sensitivity to estimation noise: re-estimate on block-bootstrapped
    windows at several dates and measure the dispersion of the resulting weights."""
    rng = np.random.default_rng(seed)
    S = build_strategies()
    names = ["HRP", "HRP-Clip", "HRP-Proj", "MinVar-C", "MVO-C", "CA-HRP"]
    rows = []
    for u in universe_names():
        rets, groups = load_universe(u)
        c = universe_constraints(u, groups, turnover_max=None)
        R = rets.values
        dates = np.linspace(400, len(R) - 1, 8).astype(int)
        for t in dates:
            win = R[t - 251: t + 1]
            for name in names:
                ws = []
                for b in range(n_boot):
                    starts = rng.integers(0, 252 - 21, 12)
                    ids = (starts[:, None] + np.arange(21)[None, :]).ravel()
                    ws.append(S[name](win[ids], c, None))
                ws = np.array(ws)
                rows.append({"universe": u, "date": rets.index[t], "method": name,
                             "mean_abs_dev": float(np.abs(ws - ws.mean(0)).sum(1).mean()),
                             "std_sum": float(ws.std(0).sum())})
            print(f"[stability] {u} {rets.index[t].date()}", flush=True)
    pd.DataFrame(rows).to_csv(RAW / "stability.csv", index=False)


def tightness():
    """Feasibility and distortion as constraints tighten: compare how far each
    constrained-HRP method moves from the unconstrained HRP risk allocation."""
    rows = []
    for u in universe_names():
        rets, groups = load_universe(u)
        R = rets.values
        n = R.shape[1]
        dates = np.linspace(400, len(R) - 1, 12).astype(int)
        for mult in [4.0, 3.0, 2.5, 2.0, 1.5, 1.25]:
            gm = {"etf": 0.40, "nifty": 0.25, "us": 0.20}[u]
            gm = gm * mult / 2.5 if mult < 2.5 else gm
            c = universe_constraints(u, groups, upper=mult / n, group_max=max(gm, 1.0 / len(set(groups)) + 0.02),
                                     turnover_max=None)
            for t in dates:
                win = R[t - 251: t + 1]
                cov_s = estimate(win, "sample")
                cov_l = estimate(win, "lw")
                base = hrp(cov_s)
                ws = {
                    "HRP-Clip": optim.clip_and_renormalise(base.weights, c),
                    "HRP-Proj": optim.euclidean_projection(base.weights, c),
                    "CA-HRP": ca_hrp(win, c, None, turnover=False),
                }
                for k, w in ws.items():
                    v = violations(w, c)
                    rc = w * (cov_l @ w)
                    rc = rc / rc.sum()
                    rows.append({"universe": u, "upper_x_n": mult, "date": rets.index[t], "method": k,
                                 "violation": v["box"] + v["group"],
                                 "risk_conc": float((rc ** 2).sum() * n),
                                 "port_vol": float(np.sqrt(w @ cov_l @ w * 252))})
        print(f"[tightness] {u}", flush=True)
    pd.DataFrame(rows).to_csv(RAW / "tightness.csv", index=False)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("main", "all"):
        run_set(build_strategies(), "main")
    if what in ("ablation", "all"):
        run_set(build_ablations(), "ablation")
    if what in ("lookback", "all"):
        S = build_strategies()
        sub = {k: S[k] for k in ["EW", "HRP", "HRP-Clip", "HRP-Proj", "MinVar-C", "MVO-C", "CA-HRP"]}
        run_set(sub, "lookback504", lookback=504)
        run_set(sub, "lookback126", lookback=126)
    if what in ("stability", "all"):
        stability()
    if what in ("tightness", "all"):
        tightness()
