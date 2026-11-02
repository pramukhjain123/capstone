"""CA-HRP-IV: option-implied volatility inputs, on the two multi-asset universes.

Runs CA-HRP-IV and MinVar-C-IV (same IV covariance given to the strongest benchmark, so any
gain can be attributed to information rather than method), and checks that the IV code path
with historical volatilities reproduces CA-HRP exactly.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cahrp.backtest import load_universe, run_backtest, universe_constraints  # noqa: E402
from cahrp.estimators import estimate  # noqa: E402
from cahrp.implied import iv_ratios_frame  # noqa: E402
from cahrp.strategies import build_iv_strategies, ca_hrp_from_cov  # noqa: E402

RAW = ROOT / "results" / "raw"

if __name__ == "__main__":
    for u in ["etf", "india_etf"]:
        rets, groups = load_universe(u)
        c = universe_constraints(u, groups)
        S = build_iv_strategies(u, list(rets.columns))
        out = {name: run_backtest(rets, fn, c) for name, fn in S.items()}
        # identity check: IV code path with plain LW covariance == CA-HRP
        same = run_backtest(rets, lambda R, cc, wp: ca_hrp_from_cov(estimate(R, "lw"), cc, wp), c)
        with open(RAW / f"main6_{u}.pkl", "rb") as f:
            ref = pickle.load(f)["CA-HRP"]
        out["_identity_gap"] = float(np.abs(same.weights.values - ref.weights.values).max())
        out["_ratios"] = iv_ratios_frame(u, list(out["CA-HRP-IV"].weights.index))
        with open(RAW / f"iv_{u}.pkl", "wb") as f:
            pickle.dump(out, f)
        print(f"[iv] {u}: identity gap {out['_identity_gap']:.2e}", flush=True)
