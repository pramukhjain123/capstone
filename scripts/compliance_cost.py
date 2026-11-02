"""Price-of-compliance attribution and noise-damping (non-expansiveness) experiments.

1. For every CA-HRP rebalance in all six universes: the ex-ante tracking error to HRP that
   compliance costs, its exact Shapley split over three constraint families
   (asset caps, group caps/floors, turnover), and the KKT-identity check.
   Also verifies that the re-solved portfolio equals the stored backtest weights.
2. Noise damping: bootstrap the estimation window, and compare the dispersion of the HRP
   target with that of each projected portfolio, measured in the S-norm of the projection
   (Proposition 4: the S-projection can never expand it; clipping has no such guarantee).
"""
import itertools
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cahrp import optim  # noqa: E402
from cahrp.backtest import all_universe_names, load_universe, universe_constraints  # noqa: E402
from cahrp.constraints import Constraints  # noqa: E402
from cahrp.estimators import estimate  # noqa: E402
from cahrp.hrp import hrp  # noqa: E402

RAW = ROOT / "results" / "raw"
FAMILIES = ("asset caps", "group limits", "turnover")


def subset_constraints(c: Constraints, fams, w_prev):
    n = c.n
    upper = c.upper if "asset caps" in fams else np.ones(n)
    gmax = c.group_max if "group limits" in fams else {}
    gmin = c.group_min if "group limits" in fams else {}
    to = c.turnover_max if ("turnover" in fams and w_prev is not None) else None
    return Constraints(c.lower, upper, c.groups, gmax, gmin, to)


def shapley(values: dict):
    n = len(FAMILIES)
    phi = {}
    for f in FAMILIES:
        others = [x for x in FAMILIES if x != f]
        tot = 0.0
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                wgt = math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
                tot += wgt * (values[frozenset(S) | {f}] - values[frozenset(S)])
        phi[f] = tot
    return phi


def attribution(u):
    with open(RAW / f"main6_{u}.pkl", "rb") as f:
        bt = pickle.load(f)["CA-HRP"]
    rets, groups = load_universe(u)
    c = universe_constraints(u, groups)
    R = rets.values
    W = bt.weights
    rows = []
    prev_target, prev_date = None, None
    for d, w_stored in W.iterrows():
        t = rets.index.get_loc(d)
        if prev_target is None:
            w_prev = None
        else:
            # drift the previous target with realised returns up to and including day t
            t0 = rets.index.get_loc(prev_date)
            w_prev = prev_target.copy()
            for s in range(t0 + 1, t + 1):
                w_prev = w_prev * (1 + R[s])
                w_prev = w_prev / w_prev.sum()
        cov = estimate(R[t - 251: t + 1], "lw")
        target = hrp(cov).weights
        full = optim.projection_attribution(target, cov, c, w_prev)
        vals = {}
        for r in range(len(FAMILIES) + 1):
            for S in itertools.combinations(FAMILIES, r):
                if len(S) == 0:
                    vals[frozenset()] = 0.0
                    continue
                a = optim.projection_attribution(target, cov, subset_constraints(c, S, w_prev), w_prev)
                vals[frozenset(S)] = a["fstar"] if a is not None else np.nan
        phi = shapley(vals)
        row = {"universe": u, "date": d, "te_ann": full["te_ann"], "fstar": full["fstar"],
               "w_match": float(np.abs(full["w"] - w_stored.values).max()),
               "kkt_turnover_remainder": full["terms"]["turnover"],
               "turnover_dual": full["turnover_dual"]}
        for k, v in phi.items():
            row[f"shap_{k}"] = v
        for k, v in full["terms"].items():
            row[f"kkt_{k}"] = v
        rows.append(row)
        prev_target, prev_date = w_stored.values, d
    print(f"[attribution] {u}: {len(rows)} dates", flush=True)
    return rows


def noise_damping(n_boot=40, n_dates=10, seed=5):
    """Dispersion of bootstrapped targets vs projected portfolios, in the S-norm."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in all_universe_names():
        rets, groups = load_universe(u)
        c = universe_constraints(u, groups, turnover_max=None)
        R = rets.values
        n = R.shape[1]
        for t in np.linspace(400, len(R) - 1, n_dates).astype(int):
            win = R[t - 251: t + 1]
            S = estimate(win, "lw")               # fixed metric for this date
            Sn = S / np.trace(S) * n + 1e-4 * np.eye(n)
            targets, proj, clip, euc = [], [], [], []
            for b in range(n_boot):
                starts = rng.integers(0, 252 - 21, 12)
                ids = (starts[:, None] + np.arange(21)[None, :]).ravel()
                tgt = hrp(estimate(win[ids], "lw")).weights
                targets.append(tgt)
                proj.append(optim.risk_projection(tgt, S, c))
                clip.append(optim.clip_and_renormalise(tgt, c))
                euc.append(optim.euclidean_projection(tgt, c))
            T, P, C, E = map(np.array, (targets, proj, clip, euc))

            def snorm_pairs(X, M):
                out = []
                for i in range(len(X)):
                    for j in range(i + 1, len(X)):
                        dlt = X[i] - X[j]
                        out.append(np.sqrt(dlt @ M @ dlt))
                return np.array(out)

            dT_S, dP_S, dC_S = snorm_pairs(T, Sn), snorm_pairs(P, Sn), snorm_pairs(C, Sn)
            I = np.eye(n)
            dT_E, dE_E, dC_E = snorm_pairs(T, I), snorm_pairs(E, I), snorm_pairs(C, I)
            rows.append({"universe": u, "date": rets.index[t],
                         "ratio_TE_proj": float(np.max(dP_S / np.maximum(dT_S, 1e-15))),
                         "mean_ratio_TE_proj": float(dP_S.sum() / dT_S.sum()),
                         "ratio_clip_Snorm": float(np.max(dC_S / np.maximum(dT_S, 1e-15))),
                         "mean_ratio_clip_Snorm": float(dC_S.sum() / dT_S.sum()),
                         "ratio_euclid": float(np.max(dE_E / np.maximum(dT_E, 1e-15))),
                         "ratio_clip_euclid": float(np.max(dC_E / np.maximum(dT_E, 1e-15))),
                         "mean_ratio_clip_euclid": float(dC_E.sum() / dT_E.sum()),
                         "frac_pairs_clip_expands": float(np.mean(dC_E > dT_E * (1 + 1e-9)))})
        print(f"[noise] {u}", flush=True)
    return rows


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("attribution", "all"):
        rows = []
        for u in all_universe_names():
            rows += attribution(u)
        pd.DataFrame(rows).to_csv(RAW / "compliance_cost.csv", index=False)
    if what in ("noise", "all"):
        pd.DataFrame(noise_damping()).to_csv(RAW / "noise_damping.csv", index=False)
