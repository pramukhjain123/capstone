"""Allocation strategies. Each takes (returns window, constraints, previous weights) -> weights."""
from typing import Callable, Dict, Optional

import numpy as np

from . import optim
from .constraints import Constraints
from .estimators import estimate
from .hrp import hrp

Strategy = Callable[[np.ndarray, Constraints, Optional[np.ndarray]], np.ndarray]


def ca_hrp(R, c, w_prev, cov_method="lw", linkage_method="single", split="list",
           capacity=True, projection="tree", gamma=0.5, turnover=True):
    """Constraint-Aware HRP.

    1. robust covariance (Ledoit-Wolf shrinkage)
    2. HRP tree + capacity-aware recursive bisection (box bounds and group caps
       enter the split rule, box bounds are met exactly)
    3. hierarchy-consistent projection onto the full feasible set
       (group floors/caps, turnover) that preserves the top-level risk budget.
    """
    cov = estimate(R, cov_method)
    res = hrp(cov, linkage_method, split, constraints=c if capacity else None)
    cc = c
    if not turnover:
        cc = Constraints(c.lower, c.upper, c.groups, c.group_max, c.group_min, None)
    if projection == "tree":
        return optim.tree_projection(res.weights, res.nodes, cc, w_prev, gamma)
    if projection == "euclid":
        return optim.euclidean_projection(res.weights, cc, w_prev)
    return res.weights


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
    s["CA-HRP"] = lambda R, c, wp: ca_hrp(R, c, wp)
    return s


def build_ablations() -> Dict[str, Strategy]:
    return {
        "CA-HRP": lambda R, c, wp: ca_hrp(R, c, wp),
        "no LW (sample cov)": lambda R, c, wp: ca_hrp(R, c, wp, cov_method="sample"),
        "no capacity bisection": lambda R, c, wp: ca_hrp(R, c, wp, capacity=False),
        "Euclidean projection": lambda R, c, wp: ca_hrp(R, c, wp, projection="euclid"),
        "no turnover cap": lambda R, c, wp: ca_hrp(R, c, wp, turnover=False),
        "dendrogram splits": lambda R, c, wp: ca_hrp(R, c, wp, split="tree"),
        "gamma=0.25": lambda R, c, wp: ca_hrp(R, c, wp, gamma=0.25),
        "gamma=1.0": lambda R, c, wp: ca_hrp(R, c, wp, gamma=1.0),
    }
