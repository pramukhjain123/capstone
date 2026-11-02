"""Portfolio constraint specification and violation checks."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class Constraints:
    """Long-only, fully-invested portfolio constraints.

    lower, upper : per-asset bounds (arrays of length n)
    groups       : group label of each asset (length n)
    group_max    : {group: max total weight}
    group_min    : {group: min total weight}
    turnover_max : max sum |w - w_prev| per rebalance (None = unconstrained)
    """
    lower: np.ndarray
    upper: np.ndarray
    groups: Sequence[str]
    group_max: Dict[str, float] = field(default_factory=dict)
    group_min: Dict[str, float] = field(default_factory=dict)
    turnover_max: Optional[float] = None

    @property
    def n(self) -> int:
        return len(self.lower)

    def group_matrix(self):
        labels = sorted(set(self.groups))
        G = np.array([[1.0 if g == lab else 0.0 for g in self.groups] for lab in labels])
        return labels, G

    def box_feasible(self) -> bool:
        return self.lower.sum() <= 1 + 1e-12 <= self.upper.sum() + 2e-12


def make_constraints(groups: Sequence[str], upper: float, group_max: float,
                     group_min: Optional[Dict[str, float]] = None,
                     turnover_max: Optional[float] = None,
                     group_max_overrides: Optional[Dict[str, float]] = None) -> Constraints:
    n = len(groups)
    labels = sorted(set(groups))
    gmax = {g: group_max for g in labels}
    if group_max_overrides:
        gmax.update(group_max_overrides)
    return Constraints(lower=np.zeros(n), upper=np.full(n, upper), groups=list(groups),
                       group_max=gmax, group_min=dict(group_min or {}),
                       turnover_max=turnover_max)


def violations(w: np.ndarray, c: Constraints, w_prev: Optional[np.ndarray] = None,
               tol: float = 1e-6) -> Dict[str, float]:
    """Total magnitude by which w breaks each constraint family (0 = satisfied)."""
    box = np.maximum(w - c.upper, 0).sum() + np.maximum(c.lower - w, 0).sum()
    labels, G = c.group_matrix()
    gw = G @ w
    grp = 0.0
    for lab, val in zip(labels, gw):
        if lab in c.group_max:
            grp += max(val - c.group_max[lab], 0)
        if lab in c.group_min:
            grp += max(c.group_min[lab] - val, 0)
    to = 0.0
    if c.turnover_max is not None and w_prev is not None:
        to = max(np.abs(w - w_prev).sum() - c.turnover_max, 0)
    out = {"box": box, "group": grp, "turnover": to}
    out = {k: (v if v > tol else 0.0) for k, v in out.items()}
    out["any"] = float(any(v > 0 for v in out.values()))
    return out
