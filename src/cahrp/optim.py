"""Convex optimisation pieces: constrained projections and baseline portfolios."""
from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np

from .constraints import Constraints


def _constraint_list(w, c: Constraints, w_prev: Optional[np.ndarray], use_turnover: bool = True):
    cons = [cp.sum(w) == 1, w >= c.lower, w <= c.upper]
    labels, G = c.group_matrix()
    for k, lab in enumerate(labels):
        if lab in c.group_max:
            cons.append(G[k] @ w <= c.group_max[lab])
        if lab in c.group_min:
            cons.append(G[k] @ w >= c.group_min[lab])
    if use_turnover and c.turnover_max is not None and w_prev is not None:
        cons.append(cp.norm1(w - w_prev) <= c.turnover_max)
    return cons


def _solve(objective, w, c, w_prev, fallback: np.ndarray) -> np.ndarray:
    """Solve; if the turnover cap makes the problem infeasible, retry without it."""
    for use_to in (True, False):
        prob = cp.Problem(objective, _constraint_list(w, c, w_prev, use_to))
        for solver in ("CLARABEL", "OSQP", "SCS"):
            try:
                prob.solve(solver=solver)
            except Exception:
                continue
            if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                x = np.clip(np.asarray(w.value).ravel(), 0, None)
                return x / x.sum()
    return fallback


def node_matrix(nodes: List[Tuple[np.ndarray, int]], n: int, gamma: float):
    """Membership matrix A (one row per tree node) and depth weights gamma**depth.

    Leaves are always included (at depth max+1 if the tree does not list them),
    which makes the objective strictly convex.
    """
    rows, wts = [], []
    seen_leaves = set()
    for idx, depth in nodes:
        r = np.zeros(n)
        r[idx] = 1.0
        rows.append(r)
        wts.append(gamma ** depth)
        if len(idx) == 1:
            seen_leaves.add(int(idx[0]))
    max_depth = max((d for _, d in nodes), default=0)
    for i in range(n):
        if i not in seen_leaves:
            r = np.zeros(n)
            r[i] = 1.0
            rows.append(r)
            wts.append(gamma ** (max_depth + 1))
    return np.array(rows), np.array(wts)


def tree_projection(target: np.ndarray, nodes, c: Constraints,
                    w_prev: Optional[np.ndarray] = None, gamma: float = 0.5) -> np.ndarray:
    """Hierarchy-consistent projection (this project).

        min_w  sum_k gamma^{depth_k} * ( a_k' w - a_k' w_target )^2
        s.t.   w in feasible set (box, groups, turnover)

    Moving weight across high-level clusters is penalised more than moving it
    inside a cluster, so the risk budget of the top splits is preserved first.
    If the target is already feasible it is returned unchanged.
    """
    n = len(target)
    A, om = node_matrix(nodes, n, gamma)
    w = cp.Variable(n)
    resid = A @ w - A @ target
    obj = cp.Minimize(cp.sum(cp.multiply(om, cp.square(resid))))
    return _solve(obj, w, c, w_prev, fallback=target)


def risk_projection(target: np.ndarray, cov: np.ndarray, c: Constraints,
                    w_prev: Optional[np.ndarray] = None, nodes=None, gamma: float = 0.5,
                    tree_weight: float = 0.0, diag_only: bool = False,
                    ridge: float = 1e-4) -> np.ndarray:
    """Tracking-error projection (this project).

        min_w  (w - w*)' S (w - w*)  +  tree_weight * sum_k gamma^d_k (a_k'(w - w*))^2
        s.t.   w feasible

    S = Sigma (full covariance) or diag(Sigma). With S = Sigma the result is the
    feasible portfolio whose returns track the unconstrained HRP portfolio most
    closely; excess weight removed from capped assets is re-assigned to the
    assets that best replicate their risk, instead of being spread uniformly
    (Euclidean) or pushed down the tree (capacity-only bisection).
    """
    n = len(target)
    S = np.diag(np.diag(cov)) if diag_only else 0.5 * (cov + cov.T)
    S = S / np.trace(S) * n + ridge * np.eye(n)         # scale-free
    w = cp.Variable(n)
    d = w - target
    obj = cp.quad_form(d, cp.psd_wrap(S))
    if tree_weight > 0 and nodes is not None:
        A, om = node_matrix(nodes, n, gamma)
        obj = obj + tree_weight * cp.sum(cp.multiply(om, cp.square(A @ d)))
    return _solve(cp.Minimize(obj), w, c, w_prev, fallback=target)


def robust_projection(target: np.ndarray, covs: List[np.ndarray], c: Constraints,
                      w_prev: Optional[np.ndarray] = None, aggregate: str = "max",
                      ridge: float = 1e-4) -> np.ndarray:
    """Regime-robust tracking-error projection (this project).

        min_w  max_k (w - w*)' S_k (w - w*)      (aggregate="max")
        min_w  mean_k (w - w*)' S_k (w - w*)     (aggregate="mean")
        s.t.   w feasible

    Every scenario covariance is scaled by the same factor (that of the first one),
    so relative risk levels across regimes are preserved. Because the tracking-error
    variance is linear in S, the "max" solution bounds the tracking error under every
    covariance matrix in the convex hull of the scenarios.
    """
    n = len(target)
    scale = n / np.trace(covs[0])
    w = cp.Variable(n)
    d = w - target
    terms = [cp.quad_form(d, cp.psd_wrap(0.5 * (S + S.T) * scale + ridge * np.eye(n))) for S in covs]
    if aggregate == "max":
        t = cp.Variable()
        obj = cp.Minimize(t)
        # epigraph form is handled by adding the extra constraints through _solve_extra
        return _solve_extra(obj, w, c, w_prev, [tk <= t for tk in terms], fallback=target)
    obj = cp.Minimize(sum(terms) / len(terms))
    return _solve(obj, w, c, w_prev, fallback=target)


def _solve_extra(objective, w, c, w_prev, extra, fallback):
    for use_to in (True, False):
        prob = cp.Problem(objective, _constraint_list(w, c, w_prev, use_to) + extra)
        for solver in ("CLARABEL", "SCS"):
            try:
                prob.solve(solver=solver)
            except Exception:
                continue
            if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                x = np.clip(np.asarray(w.value).ravel(), 0, None)
                return x / x.sum()
    return fallback


def projection_attribution(target: np.ndarray, cov: np.ndarray, c: Constraints,
                           w_prev: Optional[np.ndarray] = None, ridge: float = 1e-4):
    """Solve the tracking-error projection and decompose its optimal value by constraint.

    For f(w) = d'Sd with d = w - t, KKT stationarity gives 2Sd = -(sum_k lambda_k a_k + nu 1)
    for linear constraints a_k'w <= b_k, hence the exact identity
        f* = d'Sd = 1/2 * sum_k lambda_k (a_k't - b_k)          (budget term vanishes)
    i.e. the cost of compliance splits additively over constraints, each term being the
    shadow price times the amount by which the HRP target violates that constraint.
    The l1 turnover constraint is non-linear; its contribution is the remainder f* - sum.

    Returns dict with weights, f*, per-family terms and annualised ex-ante TE.
    """
    n = len(target)
    S = 0.5 * (cov + cov.T)
    S = S / np.trace(S) * n + ridge * np.eye(n)
    w = cp.Variable(n)
    d = w - target
    labels, G = c.group_matrix()
    cap_rows = [k for k, lab in enumerate(labels) if lab in c.group_max]
    flo_rows = [k for k, lab in enumerate(labels) if lab in c.group_min]
    cons = {"budget": cp.sum(w) == 1, "long-only": w >= c.lower, "asset caps": w <= c.upper}
    if cap_rows:
        cmax = np.array([c.group_max[labels[k]] for k in cap_rows])
        cons["group caps"] = G[cap_rows] @ w <= cmax
    if flo_rows:
        cmin = np.array([c.group_min[labels[k]] for k in flo_rows])
        cons["group floors"] = G[flo_rows] @ w >= cmin
    use_to = c.turnover_max is not None and w_prev is not None
    if use_to:
        cons["turnover"] = cp.norm1(w - w_prev) <= c.turnover_max
    prob = cp.Problem(cp.Minimize(cp.quad_form(d, cp.psd_wrap(S))), list(cons.values()))
    prob.solve(solver="CLARABEL")
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    lam = {k: np.atleast_1d(np.asarray(v.dual_value, dtype=float)) for k, v in cons.items()}
    terms = {
        "long-only": 0.5 * float(lam["long-only"] @ (c.lower - target)),
        "asset caps": 0.5 * float(lam["asset caps"] @ (target - c.upper)),
    }
    if cap_rows:
        terms["group caps"] = 0.5 * float(lam["group caps"] @ (G[cap_rows] @ target - cmax))
    if flo_rows:
        terms["group floors"] = 0.5 * float(lam["group floors"] @ (cmin - G[flo_rows] @ target))
    fstar = float(prob.value)
    terms["turnover"] = fstar - sum(terms.values())
    x = np.asarray(w.value).ravel()
    dd = x - target
    return {"w": x, "fstar": fstar, "terms": terms,
            "te_ann": float(np.sqrt(max(dd @ cov @ dd, 0) * 252)),
            "turnover_dual": float(lam["turnover"][0]) if use_to else 0.0}


def euclidean_projection(target: np.ndarray, c: Constraints,
                         w_prev: Optional[np.ndarray] = None) -> np.ndarray:
    w = cp.Variable(len(target))
    return _solve(cp.Minimize(cp.sum_squares(w - target)), w, c, w_prev, fallback=target)


def min_variance(cov: np.ndarray, c: Optional[Constraints] = None,
                 w_prev: Optional[np.ndarray] = None) -> np.ndarray:
    n = cov.shape[0]
    w = cp.Variable(n)
    P = cp.psd_wrap(0.5 * (cov + cov.T))
    obj = cp.Minimize(cp.quad_form(w, P))
    if c is None:
        prob = cp.Problem(obj, [cp.sum(w) == 1, w >= 0])
        prob.solve(solver="CLARABEL")
        x = np.clip(w.value, 0, None)
        return x / x.sum()
    return _solve(obj, w, c, w_prev, fallback=np.full(n, 1 / n))


def mean_variance(mu: np.ndarray, cov: np.ndarray, risk_aversion: float,
                  c: Optional[Constraints] = None, w_prev: Optional[np.ndarray] = None) -> np.ndarray:
    n = cov.shape[0]
    w = cp.Variable(n)
    P = cp.psd_wrap(0.5 * (cov + cov.T))
    obj = cp.Maximize(mu @ w - 0.5 * risk_aversion * cp.quad_form(w, P))
    if c is None:
        prob = cp.Problem(obj, [cp.sum(w) == 1, w >= 0])
        prob.solve(solver="CLARABEL")
        x = np.clip(w.value, 0, None)
        return x / x.sum()
    return _solve(obj, w, c, w_prev, fallback=np.full(n, 1 / n))


def equal_risk_contribution(cov: np.ndarray) -> np.ndarray:
    """Long-only ERC via the convex log-barrier formulation (Spinu, 2013)."""
    n = cov.shape[0]
    y = cp.Variable(n)
    P = cp.psd_wrap(0.5 * (cov + cov.T))
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(y, P) - cp.sum(cp.log(y)) / n), [y >= 1e-10])
    prob.solve(solver="CLARABEL")
    x = np.asarray(y.value).ravel()
    return x / x.sum()


def clip_and_renormalise(w: np.ndarray, c: Constraints, iters: int = 500) -> np.ndarray:
    """Common practitioner heuristic: alternately clip to the box, scale down
    over-cap groups, and renormalise. No optimality or feasibility guarantee."""
    w = w.copy()
    labels, G = c.group_matrix()
    for _ in range(iters):
        w = np.clip(w, c.lower, c.upper)
        for k, lab in enumerate(labels):
            gw = G[k] @ w
            cap = c.group_max.get(lab, np.inf)
            if gw > cap:
                w[G[k] > 0] *= cap / gw
        w = w / w.sum()
        if (w <= c.upper + 1e-9).all() and all(
                G[k] @ w <= c.group_max.get(lab, np.inf) + 1e-9 for k, lab in enumerate(labels)):
            break
    return w
