"""Design study on the VALIDATION period only (first ~half of each sample).

Chooses the CA-HRP configuration before any test-period result is looked at.
The test period (after VALID_END) is never used here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cahrp import optim  # noqa: E402
from cahrp.backtest import load_universe, run_backtest, universe_constraints, universe_names  # noqa
from cahrp.estimators import estimate  # noqa: E402
from cahrp.hrp import hrp  # noqa: E402
from cahrp.metrics import perf, structure  # noqa: E402

VALID_END = {"etf": "2016-12-31", "nifty": "2018-12-31", "us": "2015-12-31"}


def variant(cov_m, capacity, proj, tree_w=0.0):
    def f(R, c, wp):
        cov = estimate(R, cov_m)
        res = hrp(cov, "single", "list", constraints=c if capacity else None)
        if proj == "tree":
            return optim.tree_projection(res.weights, res.nodes, c, wp)
        if proj == "euclid":
            return optim.euclidean_projection(res.weights, c, wp)
        if proj == "diag":
            return optim.risk_projection(res.weights, cov, c, wp, diag_only=True)
        if proj == "te":
            return optim.risk_projection(res.weights, cov, c, wp, res.nodes, tree_weight=tree_w)
        raise ValueError(proj)
    return f


VARIANTS = {
    "cap+tree (v0)": variant("lw", True, "tree"),
    "euclid (=HRP-Proj, LW)": variant("lw", False, "euclid"),
    "cap+euclid": variant("lw", True, "euclid"),
    "diag-var proj": variant("lw", False, "diag"),
    "TE proj": variant("lw", False, "te"),
    "cap+TE proj": variant("lw", True, "te"),
    "TE+tree proj": variant("lw", False, "te", tree_w=1.0),
    "TE proj (sample cov)": variant("sample", False, "te"),
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
            p.update(structure(r.weights, rets))
            p.update({"universe": u, "variant": name, "Turnover": r.turnover.iloc[1:].mean() * 12,
                      "ViolRate": r.viol["any"].mean()})
            rows.append(p)
            print(u, name, round(p["Sharpe"], 3), flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables" / "design_study_validation.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df[["universe", "variant", "Sharpe", "Vol", "MaxDD", "Turnover", "RiskConc", "EffN"]]
          .round(3).to_string(index=False))
