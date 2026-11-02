"""Validation-period design study for the regime-robust projection (CA-HRP-R).

Pre-declared rule: among the robust variants, adopt the one with the highest average
validation Sharpe ratio over the three original universes. Test-period data are not used.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cahrp.backtest import load_universe, run_backtest, universe_constraints, universe_names  # noqa
from cahrp.metrics import perf  # noqa: E402
from cahrp.strategies import ca_hrp, ca_hrp_robust  # noqa: E402

VALID_END = {"etf": "2016-12-31", "nifty": "2018-12-31", "us": "2015-12-31"}

VARIANTS = {
    "CA-HRP (reference)": lambda R, c, wp: ca_hrp(R, c, wp),
    "max{long,short}": lambda R, c, wp: ca_hrp_robust(R, c, wp, ("long", "short"), "max"),
    "max{long,stress}": lambda R, c, wp: ca_hrp_robust(R, c, wp, ("long", "stress"), "max"),
    "max{long,short,stress}": lambda R, c, wp: ca_hrp_robust(R, c, wp, ("long", "short", "stress"), "max"),
    "mean{long,short,stress}": lambda R, c, wp: ca_hrp_robust(R, c, wp, ("long", "short", "stress"), "mean"),
}

if __name__ == "__main__":
    rows = []
    for u in universe_names():
        rets, groups = load_universe(u)
        rets = rets.loc[:VALID_END[u]]
        c = universe_constraints(u, groups)
        for name, fn in VARIANTS.items():
            r = run_backtest(rets, fn, c)
            p = perf(r.net(10))
            p.update({"universe": u, "variant": name, "Turnover": r.turnover.iloc[1:].mean() * 12})
            rows.append(p)
            print(u, name, round(p["Sharpe"], 3), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "tables" / "design_study_robust_validation.csv", index=False)
    piv = df.pivot(index="variant", columns="universe", values="Sharpe")
    piv["mean"] = piv.mean(axis=1)
    print(piv.round(3))
    robust = piv.drop(index="CA-HRP (reference)")
    print("SELECTED:", robust["mean"].idxmax())
