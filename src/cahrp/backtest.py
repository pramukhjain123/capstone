"""Walk-forward backtest with monthly rebalancing, weight drift and transaction costs."""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .constraints import Constraints, make_constraints, violations
from .universes import EXTERNAL_UNIVERSES, UNIVERSES

ROOT = Path(__file__).resolve().parents[2]

# Constraint sets per universe (fixed a priori, before looking at results)
CONSTRAINT_SPECS = {
    "etf": dict(upper=0.15, group_max=0.40, group_min={"Equity": 0.20}, turnover_max=0.30),
    "nifty": dict(upper=0.06, group_max=0.25, group_min=None, turnover_max=0.30),
    "us": dict(upper=0.06, group_max=0.20, group_min=None, turnover_max=0.30),
    # external-validation universes (fixed before running any backtest on them)
    "india_etf": dict(upper=0.35, group_max=0.80, group_min={"IndiaEquity": 0.40}, turnover_max=0.30,
                      group_max_overrides={"Gold": 0.25, "GlobalEquity": 0.35}),
    "sector": dict(upper=0.20, group_max=0.60, group_min=None, turnover_max=0.30,
                   group_max_overrides={"Defensive": 0.50}),
    "country": dict(upper=0.10, group_max=0.40, group_min=None, turnover_max=0.30),
}


def repair_split_glitches(prices: pd.DataFrame, log: list = None) -> pd.DataFrame:
    """Undo temporary unadjusted stretches around unit splits.

    Yahoo occasionally records a split on day i (return ~ -90%) and reverses it a few
    days later (return ~ +900%). We detect a fall below -80% followed within 10 days by a
    rise that cancels it to within 30%, and rescale the prices in between.
    """
    p = prices.copy()
    for col in p.columns:
        x = p[col].values.astype(float)
        r = x[1:] / x[:-1] - 1
        for i in np.where(r < -0.8)[0]:
            for j in range(i + 1, min(i + 11, len(r))):
                if r[j] > 3 and abs((1 + r[i]) * (1 + r[j]) - 1) < 0.3:
                    f = 1.0 / (1 + r[i])
                    x[i + 1: j + 1] *= f
                    if log is not None:
                        log.append((col, p.index[i + 1].date(), p.index[j].date(), round(f, 2)))
                    r = x[1:] / x[:-1] - 1
                    break
        p[col] = x
    return p


def load_universe(name: str):
    prices = pd.read_csv(ROOT / "data" / "raw" / f"{name}_prices.csv", index_col=0, parse_dates=True)
    groups = pd.read_csv(ROOT / "data" / "raw" / f"{name}_groups.csv", index_col=0)["group"]
    prices = prices[groups.index].ffill().dropna(how="any")
    prices = repair_split_glitches(prices)
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


def rebalance_dates(index: pd.DatetimeIndex, lookback: int, freq: str = "M") -> List[pd.Timestamp]:
    s = pd.Series(index, index=index)
    if freq == "M":
        ends = s.groupby([index.year, index.month]).max().values
    elif freq == "Q":
        ends = s.groupby([index.year, (index.month - 1) // 3]).max().values
    elif freq == "W":
        iso = index.isocalendar()
        ends = s.groupby([iso.year.values, iso.week.values]).max().values
    else:
        raise ValueError(freq)
    return sorted(d for d in ends if index.get_loc(d) >= lookback)


def run_backtest(rets: pd.DataFrame, strategy, c: Constraints, lookback: int = 252,
                 freq: str = "M") -> BacktestResult:
    idx = rets.index
    R = rets.values
    n = R.shape[1]
    rebal = set(rebalance_dates(idx, lookback, freq))
    first = min(rebal)
    start = idx.get_loc(first)

    w = None                     # current (drifted) weights
    gross, to_rec, w_rec, v_rec = [], {}, {}, {}
    t0 = time.time()
    for t in range(start, len(idx)):
        d = idx[t]
        if w is not None:
            # day-t return earned by the weights held over day t, then drift
            r = R[t]
            gross.append((d, float(w @ r)))
            w = w * (1 + r)
            w = w / w.sum()
        if d in rebal:
            # trade at the close of day t; cost is charged to day t
            window = R[t - lookback + 1: t + 1]
            target = np.asarray(strategy(window, c, w), dtype=float)
            target = np.clip(target, 0, None)
            target /= target.sum()
            to_rec[d] = float(np.abs(target - w).sum()) if w is not None else 0.0
            w_rec[d] = target
            v_rec[d] = violations(target, c, w)
            w = target
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


def external_universe_names():
    return list(EXTERNAL_UNIVERSES.keys())


def all_universe_names():
    return universe_names() + external_universe_names()
