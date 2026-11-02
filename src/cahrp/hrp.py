"""Hierarchical Risk Parity (HRP) and its capacity-aware recursive bisection.

Standard HRP (Lopez de Prado, 2016):
  1. tree clustering on the correlation distance d_ij = sqrt((1 - rho_ij) / 2)
  2. quasi-diagonalisation (leaf order of the dendrogram)
  3. recursive bisection: split weight between two sub-clusters in inverse
     proportion to their (inverse-variance-portfolio) variance.

Capacity-aware bisection (this project):
  every split fraction alpha is clipped into the interval that keeps both
  children inside their *weight capacity* [lo_S, hi_S], where
      lo_S = sum_{i in S} l_i
      hi_S = sum_g min( sum_{i in S, g(i)=g} u_i , c_g )
  Because lo is additive and hi is sub-additive the interval is never empty,
  so every leaf ends up inside its box bounds exactly (Proposition 1 in the paper).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, to_tree
from scipy.spatial.distance import squareform

from .constraints import Constraints
from .estimators import cov_to_corr


@dataclass
class HRPResult:
    weights: np.ndarray
    order: np.ndarray
    nodes: List[Tuple[np.ndarray, int]]       # (asset indices, depth) for every non-root node
    raw_alphas: List[float]
    clipped_alphas: List[float]


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(d, 0.0)
    return d


def cluster(cov: np.ndarray, method: str = "single"):
    dist = correlation_distance(cov_to_corr(cov))
    if method == "ward":
        # Ward on the "distance of distances" embedding, as is common for HRP variants
        n = len(dist)
        dd = np.sqrt(((dist[:, None, :] - dist[None, :, :]) ** 2).sum(-1))
        Z = linkage(squareform(dd, checks=False), method="ward")
    else:
        Z = linkage(squareform(dist, checks=False), method=method)
    return Z


def ivp_variance(cov: np.ndarray, idx: np.ndarray) -> float:
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def _capacity(idx: np.ndarray, c: Optional[Constraints]) -> Tuple[float, float]:
    if c is None:
        return 0.0, 1.0
    lo = float(c.lower[idx].sum())
    hi = 0.0
    groups = np.asarray(c.groups)[idx]
    for g in np.unique(groups):
        cap_assets = float(c.upper[idx][groups == g].sum())
        cap_group = c.group_max.get(g, np.inf)
        hi += min(cap_assets, cap_group)
    return lo, min(hi, 1.0)


def _children(Z, n, split: str, order: np.ndarray):
    """Return a function mapping a cluster (ordered index array) to its two children."""
    if split == "list":
        def kids(idx):
            h = len(idx) // 2
            return idx[:h], idx[h:]
        return kids

    root = to_tree(Z)
    lookup = {}

    def walk(node):
        if node.is_leaf():
            return np.array([node.id])
        left, right = walk(node.left), walk(node.right)
        lookup[tuple(sorted(np.concatenate([left, right])))] = (left, right)
        return np.concatenate([left, right])

    walk(root)

    def kids(idx):
        return lookup[tuple(sorted(idx))]
    return kids


def hrp(cov: np.ndarray, linkage_method: str = "single", split: str = "list",
        constraints: Optional[Constraints] = None) -> HRPResult:
    """HRP weights. If `constraints` is given, use capacity-aware bisection."""
    n = cov.shape[0]
    Z = cluster(cov, linkage_method)
    order = leaves_list(Z)
    kids = _children(Z, n, split, order)

    w = np.zeros(n)
    nodes, raw, clipped = [], [], []
    stack = [(np.asarray(order), 1.0, 0)]
    while stack:
        idx, W, depth = stack.pop()
        if len(idx) == 1:
            w[idx[0]] = W
            continue
        left, right = kids(idx)
        vl, vr = ivp_variance(cov, left), ivp_variance(cov, right)
        alpha = 1.0 - vl / (vl + vr)
        raw.append(alpha)
        if constraints is not None and W > 0:
            lo_l, hi_l = _capacity(left, constraints)
            lo_r, hi_r = _capacity(right, constraints)
            a_min = max(lo_l, W - hi_r) / W
            a_max = min(hi_l, W - lo_r) / W
            if a_min > a_max + 1e-12:     # cannot happen when the root is feasible
                a_min = a_max = 0.5 * (a_min + a_max)
            alpha = float(np.clip(alpha, a_min, a_max))
        clipped.append(alpha)
        for child, cw in ((left, W * alpha), (right, W * (1 - alpha))):
            nodes.append((np.asarray(child), depth + 1))
            stack.append((np.asarray(child), cw, depth + 1))
    return HRPResult(w, np.asarray(order), nodes, raw, clipped)
