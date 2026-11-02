"""Verification suite: independent checks that the implementation and results are correct.

Writes results/tables/verification.csv (check, detail, value, threshold, passed) and
data/MANIFEST.csv (SHA-256 of every data file).
"""
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cahrp import optim  # noqa: E402
from cahrp.backtest import all_universe_names, load_universe, run_backtest, universe_constraints  # noqa
from cahrp.constraints import make_constraints  # noqa: E402
from cahrp.estimators import estimate  # noqa: E402
from cahrp.hrp import hrp  # noqa: E402
from cahrp.metrics import perf  # noqa: E402
from cahrp.strategies import build_strategies  # noqa: E402

RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"
CHECKS = []


def record(check, detail, value, threshold, passed=None):
    if passed is None:
        passed = bool(value <= threshold)
    CHECKS.append({"check": check, "detail": detail, "value": value, "threshold": threshold,
                   "passed": passed})
    print(f"[{'PASS' if passed else 'FAIL'}] {check} | {detail} | {value:.3g} (<= {threshold})", flush=True)


def load6(u):
    with open(RAW / f"main6_{u}.pkl", "rb") as f:
        return pickle.load(f)


def check_pypfopt():
    import scipy.cluster.hierarchy as sch
    if not hasattr(sch, "_LINKAGE_METHODS"):     # PyPortfolioOpt 1.6 expects this private name
        sch._LINKAGE_METHODS = {m: i for i, m in enumerate(
            ["single", "complete", "average", "weighted", "centroid", "median", "ward"])}
    from pypfopt import EfficientFrontier
    from pypfopt.hierarchical_portfolio import HRPOpt
    worst_hrp, worst_mv = 0.0, 0.0
    for u in all_universe_names():
        rets, groups = load_universe(u)
        n = rets.shape[1]
        for t in np.linspace(300, len(rets) - 1, 6).astype(int):
            win = rets.iloc[t - 251: t + 1]
            ours = hrp(estimate(win.values, "sample"), "single", "list").weights
            ref = HRPOpt(win).optimize(linkage_method="single")
            ref = np.array([ref[c] for c in win.columns])
            worst_hrp = max(worst_hrp, np.abs(ours - ref).max())
            cov = estimate(win.values, "lw")
            u_cap = min(1.0, 3.0 / n)
            c = make_constraints(groups, upper=u_cap, group_max=1.0)
            mine = optim.min_variance(cov, c)
            ef = EfficientFrontier(None, pd.DataFrame(cov, index=win.columns, columns=win.columns),
                                   weight_bounds=(0, u_cap))
            refw = ef.min_volatility()
            refw = np.array([refw[c_] for c_ in win.columns])
            # compare portfolio variance (weights can differ when the optimum is flat)
            v1, v2 = mine @ cov @ mine, refw @ cov @ refw
            worst_mv = max(worst_mv, (v1 - v2) / v2)          # > 0: our solution is worse
    record("HRP matches PyPortfolioOpt", "max |w_ours - w_ref| over 36 windows, 6 universes", worst_hrp, 1e-8)
    record("Min-variance matches PyPortfolioOpt",
           "max relative excess variance of our solution over the reference, 36 windows", worst_mv, 1e-4)


def check_minvar_impact():
    """Follow-up to the precision check: re-solve MinVar-C with a trace-normalised objective
    (better conditioned for the solver) and measure the change in its backtest Sharpe ratio."""
    def mv_scaled(R, c, wp):
        cov = estimate(R, "lw")
        return optim.min_variance(cov / np.trace(cov) * cov.shape[0], c, wp)
    worst = 0.0
    from cahrp.metrics import perf as _perf
    for u in all_universe_names():
        rets, groups = load_universe(u)
        c = universe_constraints(u, groups)
        new = run_backtest(rets, mv_scaled, c)
        old = load6(u)["MinVar-C"]
        worst = max(worst, abs(_perf(new.net(10))["Sharpe"] - _perf(old.net(10))["Sharpe"]))
    record("Min-variance precision: economic impact",
           "max change in MinVar-C Sharpe when re-solved with a rescaled objective, 6 universes", worst, 1e-3)


def check_lookahead():
    rets, groups = load_universe("etf")
    c = universe_constraints("etf", groups)
    S = build_strategies()
    cut = rets.index[2500]
    a = run_backtest(rets, S["CA-HRP"], c)
    noisy = rets.copy()
    rng = np.random.default_rng(0)
    noisy.loc[noisy.index > cut] = rng.normal(0, 0.05, noisy.loc[noisy.index > cut].shape)
    b = run_backtest(noisy, S["CA-HRP"], c)
    wa = a.weights.loc[:cut]
    wb = b.weights.loc[:cut]
    diff = float(np.abs(wa.values - wb.values).max())
    ra = a.gross.loc[:cut]
    rb = b.gross.loc[:cut]
    record("No look-ahead", "weights up to a date unchanged when all later returns are replaced by noise",
           max(diff, float(np.abs(ra.values - rb.values).max())), 0.0)


def check_determinism():
    S = build_strategies()
    worst = 0.0
    for u in ["etf", "india_etf"]:
        rets, groups = load_universe(u)
        c = universe_constraints(u, groups)
        for m in ["CA-HRP", "HRP-Proj", "MinVar-C"]:
            new = run_backtest(rets, S[m], c)
            old = load6(u)[m]
            worst = max(worst, float(np.abs(new.weights.values - old.weights.values).max()),
                        float(np.abs(new.gross.values - old.gross.values).max()))
    record("Deterministic re-run", "max diff vs stored results (2 universes x 3 methods)", worst, 1e-9)


def check_solvers():
    import cvxpy as cp
    worst = 0.0
    for u in all_universe_names():
        rets, groups = load_universe(u)
        c = universe_constraints(u, groups, turnover_max=None)
        n = rets.shape[1]
        for t in np.linspace(300, len(rets) - 1, 5).astype(int):
            cov = estimate(rets.values[t - 251: t + 1], "lw")
            tgt = hrp(cov).weights
            sols = []
            for solver, kw in [("CLARABEL", {}), ("OSQP", dict(eps_abs=1e-10, eps_rel=1e-10, max_iter=200000)),
                               ("SCS", dict(eps=1e-9, max_iters=200000))]:
                S = cov / np.trace(cov) * n + 1e-4 * np.eye(n)
                w = cp.Variable(n)
                prob = cp.Problem(cp.Minimize(cp.quad_form(w - tgt, cp.psd_wrap(S))),
                                  optim._constraint_list(w, c, None))
                prob.solve(solver=solver, **kw)
                sols.append(np.asarray(w.value).ravel())
            worst = max(worst, max(np.abs(sols[0] - s).max() for s in sols[1:]))
    record("Three solvers agree", "max weight gap CLARABEL vs OSQP/SCS, 30 projections", worst, 1e-4)


def check_constraints_raw():
    worst = 0.0
    for u in all_universe_names():
        rets, groups = load_universe(u)
        c = universe_constraints(u, groups)
        labels, G = c.group_matrix()
        res = load6(u)
        for m in ["MinVar-C", "MVO-C", "HRP-Proj", "CA-HRP-B", "CA-HRP", "CA-HRP-R"]:
            W = res[m].weights.values
            box = np.maximum(W - c.upper, 0).max()
            gw = W @ G.T
            gmax = np.array([c.group_max.get(l, 1) for l in labels])
            gmin = np.array([c.group_min.get(l, 0) for l in labels])
            grp = max(np.maximum(gw - gmax, 0).max(), np.maximum(gmin - gw, 0).max())
            to = (res[m].turnover.iloc[1:] - c.turnover_max).clip(lower=0).max()
            worst = max(worst, box, grp, to, -W.min(), abs(W.sum(1) - 1).max())
    record("Raw constraint audit", "largest breach of any constraint, 6 compliant methods x 6 universes, no tolerance",
           float(worst), 1e-6)


def check_costs_and_metrics():
    worst_cost, worst_metric = 0.0, 0.0
    for u in all_universe_names():
        res = load6(u)
        for m, r in res.items():
            gap = (r.gross.sum() - r.net(10).sum()) - r.turnover.iloc[1:].sum() * 10 / 1e4
            worst_cost = max(worst_cost, abs(gap))
            x = r.net(10).values
            # independent recomputation with plain numpy
            wealth = np.cumprod(1 + x)
            peak = np.maximum.accumulate(wealth)
            mdd = (wealth / peak - 1).min()
            sharpe = x.mean() / x.std(ddof=1) * np.sqrt(252)
            cagr = wealth[-1] ** (252 / len(x)) - 1
            p = perf(r.net(10))
            worst_metric = max(worst_metric, abs(mdd - p["MaxDD"]), abs(sharpe - p["Sharpe"]),
                               abs(cagr - p["CAGR"]))
    record("Cost accounting", "|(gross - net) - turnover x cost| summed per backtest, max", worst_cost, 1e-10)
    record("Independent metric recomputation", "max gap in Sharpe / CAGR / MaxDD, all backtests", worst_metric, 1e-9)


def check_tables_vs_raw():
    """The CSV tables used by the paper must equal a fresh recomputation from the raw backtests."""
    t = pd.read_csv(TAB / "main6_results.csv")
    worst = 0.0
    for u in all_universe_names():
        res = load6(u)
        for m, r in res.items():
            p = perf(r.net(10))
            row = t[(t.universe == u) & (t.method == m)].iloc[0]
            worst = max(worst, abs(row.Sharpe - p["Sharpe"]), abs(row.Vol - p["Vol"]))
    record("Paper tables match raw results", "max gap Sharpe/Vol between table CSV and recomputation", worst, 1e-12)


def manifest():
    rows = []
    for f in sorted((ROOT / "data" / "raw").glob("*.csv")):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        info = {"file": f.name, "sha256": h, "bytes": f.stat().st_size}
        if f.name.endswith("_prices.csv"):
            p = pd.read_csv(f, index_col=0, parse_dates=True)
            info.update({"rows": len(p), "assets": p.shape[1], "first": p.index.min().date(),
                         "last": p.index.max().date(), "missing_cells": int(p.isna().sum().sum())})
        rows.append(info)
    pd.DataFrame(rows).to_csv(ROOT / "data" / "MANIFEST.csv", index=False)
    record("Data manifest written", "SHA-256 of every raw data file", 0.0, 0.0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pypfopt":
        check_pypfopt()
        sys.exit()
    manifest()
    check_lookahead()
    check_determinism()
    check_costs_and_metrics()
    check_tables_vs_raw()
    check_constraints_raw()
    check_solvers()
    check_pypfopt()
    check_minvar_impact()
    df = pd.DataFrame(CHECKS)
    df.to_csv(TAB / "verification.csv", index=False)
    print(f"{df.passed.sum()}/{len(df)} checks passed")
