"""Allocation strategies. Each takes (returns window, constraints, previous weights) -> weights.

Final CA-HRP design was fixed by scripts/design_study.py on the validation period only.
"""
from typing import Callable, Dict, Optional

import numpy as np

from . import optim
from .constraints import Constraints
from .estimators import estimate
from .hrp import hrp

Strategy = Callable[[np.ndarray, Constraints, Optional[np.ndarray]], np.ndarray]


def ca_hrp(R, c, w_prev, cov_method="lw", linkage_method="single", split="list",
           capacity=False, metric="te", tree_weight=0.0, gamma=0.5, turnover=True):
    """Constraint-Aware HRP.

    1. robust covariance estimate (Ledoit-Wolf shrinkage)
    2. unconstrained HRP allocation (tree clustering + recursive bisection),
       optionally with capacity-aware bisection
    3. tracking-error projection onto the feasible set (box, group caps/floors,
       turnover): the feasible portfolio whose ex-ante returns stay closest to
       the HRP portfolio.

    metric: "te" (full covariance), "diag" (variances only), "euclid", "tree"
    """
    cov = estimate(R, cov_method)
    res = hrp(cov, linkage_method, split, constraints=c if capacity else None)
    cc = c if turnover else Constraints(c.lower, c.upper, c.groups, c.group_max, c.group_min, None)
    if metric == "te":
        return optim.risk_projection(res.weights, cov, cc, w_prev, res.nodes, gamma, tree_weight)
    if metric == "diag":
        return optim.risk_projection(res.weights, cov, cc, w_prev, diag_only=True)
    if metric == "euclid":
        return optim.euclidean_projection(res.weights, cc, w_prev)
    if metric == "tree":
        return optim.tree_projection(res.weights, res.nodes, cc, w_prev, gamma)
    raise ValueError(metric)


def build_strategies() -> Dict[str, Strategy]:
    s: Dict[str, Strategy] = {}
    # ---- unconstrained reference methods ----
    s["EW"] = lambda R, c, wp: np.full(R.shape[1], 1.0 / R.shape[1])
    s["IVP"] = lambda R, c, wp: (lambda v: (1 / v) / (1 / v).sum())(R.var(axis=0, ddof=1))
    s["MinVar"] = lambda R, c, wp: optim.min_variance(estimate(R, "lw"))
    s["ERC"] = lambda R, c, wp: optim.equal_risk_contribution(estimate(R, "lw"))
    s["HRP"] = lambda R, c, wp: hrp(estimate(R, "sample"), "single", "list").weights
    # ---- constrained competitors ----
    s["MinVar-C"] = lambda R, c, wp: optim.min_variance(estimate(R, "lw"), c, wp)
    s["MVO-C"] = lambda R, c, wp: optim.mean_variance(
        R.mean(axis=0) * 252, estimate(R, "lw") * 252, 5.0, c, wp)
    s["HRP-Clip"] = lambda R, c, wp: optim.clip_and_renormalise(
        hrp(estimate(R, "sample"), "single", "list").weights, c)
    s["HRP-Proj"] = lambda R, c, wp: optim.euclidean_projection(
        hrp(estimate(R, "sample"), "single", "list").weights, c, wp)
    # ---- proposed ----
    s["CA-HRP-B"] = lambda R, c, wp: ca_hrp(R, c, wp, capacity=True, metric="tree")
    s["CA-HRP"] = lambda R, c, wp: ca_hrp(R, c, wp)
    return s


def build_ablations() -> Dict[str, Strategy]:
    return {
        "CA-HRP (full)": lambda R, c, wp: ca_hrp(R, c, wp),
        "sample covariance": lambda R, c, wp: ca_hrp(R, c, wp, cov_method="sample"),
        "diagonal metric": lambda R, c, wp: ca_hrp(R, c, wp, metric="diag"),
        "Euclidean metric": lambda R, c, wp: ca_hrp(R, c, wp, metric="euclid"),
        "+ capacity bisection": lambda R, c, wp: ca_hrp(R, c, wp, capacity=True),
        "+ tree-consistency term": lambda R, c, wp: ca_hrp(R, c, wp, tree_weight=1.0),
        "no turnover cap": lambda R, c, wp: ca_hrp(R, c, wp, turnover=False),
        "Ward linkage": lambda R, c, wp: ca_hrp(R, c, wp, linkage_method="ward"),
        "dendrogram splits": lambda R, c, wp: ca_hrp(R, c, wp, split="tree"),
    }
