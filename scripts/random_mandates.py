"""Random-mandate robustness and placebo study.

Guards against the objection that the constraint sets in the paper were hand-picked.
For each draw: a random subset of assets, a random start date, random per-asset and group
caps, and either the TRUE group labels (sectors / asset classes / regions) or SCRAMBLED
labels (placebo: groups unrelated to the correlation structure). Infeasible draws are
rejected with a linear feasibility check. The design below was fixed before running.

Usage: python scripts/random_mandates.py [draws_per_universe]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cahrp import optim  # noqa: E402
from cahrp.backtest import load_universe, run_backtest  # noqa: E402
from cahrp.constraints import make_constraints  # noqa: E402
from cahrp.metrics import perf  # noqa: E402
from cahrp.strategies import build_strategies  # noqa: E402

UNIVERSES = ["etf", "nifty", "us", "country"]
METHODS = ["EW", "HRP-Clip", "HRP-Proj", "MinVar-C", "CA-HRP"]
COST = 10


def feasible(c) -> bool:
    import cvxpy as cp
    w = cp.Variable(c.n)
    prob = cp.Problem(cp.Minimize(0), optim._constraint_list(w, c, None))
    prob.solve(solver="CLARABEL")
    return prob.status == "optimal"


def draw_mandate(u, k, rets, groups):
    rng = np.random.default_rng(1000 * UNIVERSES.index(u) + k)
    N = rets.shape[1]
    while True:
        n = int(rng.integers(max(8, N // 2), N + 1))
        idx = np.sort(rng.choice(N, n, replace=False))
        placebo = bool(k % 2)                       # alternate true / scrambled labels
        g = [groups[i] for i in idx]
        if placebo:
            G = int(rng.integers(3, 7))
            g = [f"R{j}" for j in rng.integers(0, G, n)]
        G = len(set(g))
        upper = float(rng.uniform(1.5, 4.0)) / n
        gcap = float(rng.uniform(1.2, 2.5)) / G
        c = make_constraints(g, upper=upper, group_max=gcap, turnover_max=0.30)
        if not feasible(c):
            continue
        start = int(rng.integers(0, len(rets) // 3))
        return dict(idx=idx, groups=g, placebo=placebo, upper_x_n=upper * n, gcap_x_G=gcap * G,
                    n=n, G=G, start=start, c=c)


def run_one(args):
    u, k = args
    rets, groups = load_universe(u)
    m = draw_mandate(u, k, rets, groups)
    sub = rets.iloc[m["start"]:, m["idx"]]
    S = build_strategies()
    out = []
    t0 = time.time()
    hrp_bt = run_backtest(sub, S["HRP"], m["c"])
    hrp_ret = hrp_bt.net(COST)
    for name in METHODS:
        r = run_backtest(sub, S[name], m["c"])
        net = r.net(COST)
        p = perf(net)
        out.append({"universe": u, "draw": k, "method": name, "placebo": m["placebo"],
                    "n": m["n"], "G": m["G"], "upper_x_n": m["upper_x_n"], "gcap_x_G": m["gcap_x_G"],
                    "start": sub.index[0].date(), "Sharpe": p["Sharpe"], "Vol": p["Vol"],
                    "MaxDD": p["MaxDD"], "ViolRate": r.viol["any"].mean(),
                    "TE_HRP": float((net - hrp_ret.reindex(net.index)).std() * np.sqrt(252)),
                    "HRP_viol": float(hrp_bt.viol["any"].mean())})
    print(f"[random] {u} draw {k} placebo={m['placebo']} n={m['n']} {time.time() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    jobs = [(u, k) for u in UNIVERSES for k in range(draws)]
    with Pool(2) as pool:
        res = pool.map(run_one, jobs, chunksize=1)
    rows = [r for chunk in res for r in chunk]
    out = ROOT / "results" / "raw" / "random_mandates.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out)
