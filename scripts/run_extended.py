"""Extended experiments (second phase of the study).

  python scripts/run_extended.py main6        # all strategies (incl. CA-HRP-R) on all six universes
  python scripts/run_extended.py frequency    # weekly / quarterly rebalancing
  python scripts/run_extended.py decompose    # which constraint family drives the differences
  python scripts/run_extended.py montecarlo   # simulation with a known true covariance
  python scripts/run_extended.py runtime      # computation time vs number of assets
  python scripts/run_extended.py all
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cahrp import optim  # noqa: E402
from cahrp.backtest import (all_universe_names, load_universe, run_backtest,  # noqa: E402
                            universe_constraints)
from cahrp.constraints import Constraints, make_constraints, violations  # noqa: E402
from cahrp.estimators import estimate  # noqa: E402
from cahrp.hrp import hrp  # noqa: E402
from cahrp.strategies import build_strategies, ca_hrp, ca_hrp_robust  # noqa: E402

RAW = ROOT / "results" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
CORE = ["EW", "HRP", "MinVar-C", "MVO-C", "HRP-Clip", "HRP-Proj", "CA-HRP", "CA-HRP-R"]


def save(obj, name):
    with open(RAW / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def main6():
    S = build_strategies()
    for u in all_universe_names():
        rets, groups = load_universe(u)
        c = universe_constraints(u, groups)
        out = {}
        for name, fn in S.items():
            t = time.time()
            out[name] = run_backtest(rets, fn, c)
            print(f"[main6] {u:9s} {name:10s} {time.time() - t:6.1f}s", flush=True)
        save(out, f"main6_{u}")


def frequency():
    S = build_strategies()
    for freq in ["W", "Q"]:
        for u in all_universe_names():
            rets, groups = load_universe(u)
            c = universe_constraints(u, groups)
            out = {}
            for name in CORE:
                out[name] = run_backtest(rets, S[name], c, freq=freq)
            save(out, f"freq{freq}_{u}")
            print(f"[frequency] {freq} {u}", flush=True)


def decompose():
    """Run the HRP-family methods under subsets of the constraint families."""
    S = build_strategies()
    names = ["HRP-Clip", "HRP-Proj", "CA-HRP", "CA-HRP-R"]
    for u in all_universe_names():
        rets, groups = load_universe(u)
        full = universe_constraints(u, groups)
        n = full.n
        free_groups = {g: 1.0 for g in set(groups)}
        subsets = {
            "box only": Constraints(full.lower, full.upper, groups, free_groups, {}, None),
            "groups only": Constraints(full.lower, np.ones(n), groups, full.group_max, full.group_min, None),
            "box + groups": Constraints(full.lower, full.upper, groups, full.group_max, full.group_min, None),
            "all (+ turnover)": full,
        }
        out = {}
        for sname, c in subsets.items():
            for m in names:
                out[(sname, m)] = run_backtest(rets, S[m], c)
        save(out, f"decompose_{u}")
        print(f"[decompose] {u}", flush=True)


# ---------------------------------------------------------------- Monte Carlo
def true_model(N=40, K=5, seed=0):
    """Block-correlated assets with one low-volatility, negatively correlated block
    (bond-like), so that HRP concentrates there and constraints bind."""
    rng = np.random.default_rng(seed)
    block = np.repeat(np.arange(K), N // K)
    rho_in = np.array([0.7, 0.6, 0.55, 0.5, 0.65])[:K]
    corr = np.full((N, N), 0.25)
    for k in range(K):
        idx = block == k
        corr[np.ix_(idx, idx)] = rho_in[k]
    b0 = block == 0
    corr[np.ix_(b0, ~b0)] = -0.15
    corr[np.ix_(~b0, b0)] = -0.15
    np.fill_diagonal(corr, 1.0)
    vol = np.where(block == 0, rng.uniform(0.03, 0.07, N), rng.uniform(0.15, 0.40, N)) / np.sqrt(252)
    cov = corr * np.outer(vol, vol)
    # mild expected returns proportional to volatility (Sharpe ~0.4 per asset)
    mu = 0.4 * vol
    # groups = blocks, except two assets per block are relabelled to cut across the tree
    groups = [f"G{k}" for k in block]
    for k in range(K):
        idx = np.where(block == k)[0][:2]
        for i in idx:
            groups[i] = f"G{(k + 1) % K}"
    return mu, cov, groups


def simulate(mu, cov, T, dist, rng):
    N = len(mu)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((T, N))
    if dist == "t5":
        g = rng.chisquare(5, size=(T, 1)) / 5
        z = z / np.sqrt(g) * np.sqrt(3 / 5)      # unit-variance Student-t(5)
    return mu + z @ L.T


def montecarlo(reps=200, seed=11):
    mu, cov, groups = true_model()
    N = len(mu)
    c = make_constraints(groups, upper=2.5 / N, group_max=0.30)
    S = build_strategies()
    methods = ["EW", "HRP-Clip", "HRP-Proj", "MinVar-C", "MVO-C", "CA-HRP", "CA-HRP-R"]
    # oracle portfolios computed on the true covariance
    w_hrp_true = hrp(cov).weights          # infeasible oracle target on the true covariance
    rows = []
    rng = np.random.default_rng(seed)
    for dist in ["normal", "t5"]:
        for T in [126, 252, 504]:
            for rep in range(reps):
                R = simulate(mu, cov, T, dist, rng)
                for m in methods:
                    w = S[m](R, c, None)
                    d = w - w_hrp_true
                    v = violations(w, c)
                    rows.append({"dist": dist, "T": T, "rep": rep, "method": m,
                                 "true_vol": float(np.sqrt(w @ cov @ w * 252)),
                                 "true_te_hrp": float(np.sqrt(d @ cov @ d * 252)),
                                 "true_sharpe": float(w @ mu / np.sqrt(w @ cov @ w) * np.sqrt(252)),
                                 "violation": v["box"] + v["group"],
                                 "eff_n": float(1 / np.sum(w ** 2))})
            print(f"[montecarlo] {dist} T={T}", flush=True)
    pd.DataFrame(rows).to_csv(RAW / "montecarlo.csv", index=False)


def runtime(reps=5, seed=3):
    rng = np.random.default_rng(seed)
    S = build_strategies()
    methods = ["HRP", "HRP-Proj", "CA-HRP-B", "CA-HRP", "CA-HRP-R", "MinVar-C", "MVO-C"]
    rows = []
    for N in [10, 25, 50, 100, 200, 400]:
        K = 10
        groups = [f"G{i % K}" for i in range(N)]
        c = make_constraints(groups, upper=min(1.0, 3.0 / N), group_max=0.2, turnover_max=0.3)
        w_prev = np.full(N, 1.0 / N)
        F = rng.standard_normal((252, 5))
        for m in methods:
            ts = []
            for r in range(reps):
                R = (F @ rng.standard_normal((5, N)) + rng.standard_normal((252, N)) * 2) / 100
                t0 = time.perf_counter()
                S[m](R, c, w_prev)
                ts.append(time.perf_counter() - t0)
            rows.append({"N": N, "method": m, "seconds": float(np.median(ts))})
        print(f"[runtime] N={N}", flush=True)
    pd.DataFrame(rows).to_csv(RAW / "runtime.csv", index=False)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for stage, fn in [("main6", main6), ("frequency", frequency), ("decompose", decompose),
                      ("montecarlo", montecarlo), ("runtime", runtime)]:
        if what in (stage, "all"):
            fn()
