"""Walk-forward backtest with monthly rebalancing, weight drift and transaction costs."""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .constraints import Constraints, make_constraints, violations
from .universes import UNIVERSES

ROOT = Path(__file__).resolve().parents[2]

# Constraint sets per universe (fixed a priori, before looking at results)
CONSTRAINT_SPECS = {
    "etf": dict(upper=0.15, group_max=0.40, group_min={"Equity": 0.20}, turnover_max=0.30),
    "nifty": dict(upper=0.06, group_max=0.25, group_min=None, turnover_max=0.30),
    "us": dict(upper=0.06, group_max=0.20, group_min=None, turnover_max=0.30),
}


def load_universe(name: str):
    prices = pd.read_csv(ROOT / "data" / "raw" / f"{name}_prices.csv", index_col=0, parse_dates=True)
    groups = pd.read_csv(ROOT / "data" / "raw" / f"{name}_groups.csv", index_col=0)["group"]
    prices = prices[groups.index].ffill().dropna(how="any")
    rets = prices.pct_change().iloc[1:]
    return rets, list(groups.values)


def universe_constraints(name: str, groups, **overrides) -> Constraints:
    spec = dict(CONSTRAINT_SPECS[name])
    spec.update(overrides)
    return make_constraints(groups, **spec)


@dataclass
class BacktestResult:
    gross: pd.Series                   # daily gross returns
    turnover: pd.Series                # sum |w_new - w_drifted| at each rebalance
    weights: pd.DataFrame              # target weights at each rebalance
    viol: pd.DataFrame                 # constraint violations at each rebalance
    seconds: float = 0.0
    extra: Dict = field(default_factory=dict)

    def net(self, cost_bps: float) -> pd.Series:
        r = self.gross.copy()
        cost = (self.turnover * cost_bps / 1e4).reindex(r.index).fillna(0.0)
        return r - cost


def rebalance_dates(index: pd.DatetimeIndex, lookback: int) -> List[pd.Timestamp]:
    month_end = pd.Series(index, index=index).groupby([index.year, index.month]).max().values
    return [d for d in month_end if index.get_loc(d) >= lookback]


def run_backtest(rets: pd.DataFrame, strategy, c: Constraints, lookback: int = 252) -> BacktestResult:
    idx = rets.index
    R = rets.values
    n = R.shape[1]
    rebal = set(rebalance_dates(idx, lookback))
    first = min(rebal)
    start = idx.get_loc(first)

    w = None                     # current (drifted) weights
    gross, to_rec, w_rec, v_rec = [], {}, {}, {}
    t0 = time.time()
    for t in range(start, len(idx)):
        d = idx[t]
        if d in rebal:
            window = R[t - lookback + 1: t + 1]
            target = np.asarray(strategy(window, c, w), dtype=float)
            target = np.clip(target, 0, None)
            target /= target.sum()
            prev = w if w is not None else target
            to_rec[d] = float(np.abs(target - prev).sum()) if w is not None else 0.0
            w_rec[d] = target
            v_rec[d] = violations(target, c, w)
            w = target
            continue          # trade at close; returns accrue from next day
        r = R[t]
        port = float(w @ r)
        gross.append((d, port))
        w = w * (1 + r)
        w = w / w.sum()
    secs = time.time() - t0
    g = pd.Series(dict(gross)).sort_index()
    return BacktestResult(
        gross=g,
        turnover=pd.Series(to_rec).sort_index(),
        weights=pd.DataFrame(w_rec, index=rets.columns).T.sort_index(),
        viol=pd.DataFrame(v_rec).T.sort_index(),
        seconds=secs,
    )


def universe_names():
    return list(UNIVERSES.keys())
