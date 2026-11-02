"""Statistical checks on the six-universe backtests.

1. Model Confidence Set (Hansen, Lunde & Nason, 2011), T_max statistic, circular block
   bootstrap, alpha = 0.10, on (a) negative daily returns and (b) squared daily returns,
   over the mandate-compliant methods.
2. Probabilistic and Deflated Sharpe ratios (Bailey & Lopez de Prado, 2012, 2014) for CA-HRP,
   charging for every CA-HRP configuration evaluated during the project (22 trials).
3. Risk-forecast calibration: predicted (ex-ante) vs realised volatility over each holding
   month, for every method.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cahrp.backtest import all_universe_names, load_universe  # noqa: E402
from cahrp.estimators import estimate  # noqa: E402

RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"
COST = 10
COMPLIANT = ["MinVar-C", "MVO-C", "HRP-Proj", "CA-HRP-B", "CA-HRP", "CA-HRP-R"]
N_TRIALS = 22   # 8 design-study + 5 robust-design + 9 ablation configurations of CA-HRP


def load(u):
    with open(RAW / f"main6_{u}.pkl", "rb") as f:
        return pickle.load(f)


def block_indices(T, block, rng):
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T, nb)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T


def mcs(L: pd.DataFrame, alpha=0.10, n_boot=1000, block=21, seed=0):
    """Model Confidence Set with the T_max statistic.

    Returns (p-values, survivors). An eliminated model's MCS p-value is the running maximum
    of the elimination-test p-values up to its removal; survivors get p = 1 by convention.
    """
    rng = np.random.default_rng(seed)
    X = L.values
    T, m = X.shape
    boots = [block_indices(T, block, rng) for _ in range(n_boot)]
    alive = list(range(m))
    pvals, running = {}, 0.0
    while len(alive) > 1:
        Y = X[:, alive]
        dbar = Y.mean(0) - Y.mean()
        bmeans = np.array([Y[idx].mean(0) - Y[idx].mean() for idx in boots])
        sd = np.sqrt(((bmeans - dbar) ** 2).mean(0))
        t = dbar / sd
        tboot = ((bmeans - dbar) / sd).max(1)
        p = float(np.mean(tboot >= t.max()))
        if p >= alpha:
            break
        running = max(running, p)
        worst = alive[int(np.argmax(t))]
        pvals[L.columns[worst]] = running
        alive.remove(worst)
    for i in alive:
        pvals[L.columns[i]] = 1.0
    return pvals, [L.columns[i] for i in alive]


def run_mcs():
    rows = []
    for u in all_universe_names():
        res = load(u)
        R = pd.DataFrame({m: res[m].net(COST) for m in COMPLIANT}).dropna()
        for lname, L in [("return (-r)", -R), ("risk (r^2)", R ** 2)]:
            pv, survivors = mcs(L)
            for m in COMPLIANT:
                rows.append({"universe": u, "loss": lname, "method": m,
                             "in_MCS_90": m in survivors, "mcs_p": pv.get(m, 1.0)})
        print(f"[mcs] {u}", flush=True)
    pd.DataFrame(rows).to_csv(TAB / "mcs.csv", index=False)


def psr(r: pd.Series, sr_star_daily: float):
    sr = r.mean() / r.std()
    T = len(r)
    g3 = ((r - r.mean()) ** 3).mean() / r.std() ** 3
    g4 = ((r - r.mean()) ** 4).mean() / r.std() ** 4
    z = (sr - sr_star_daily) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(norm.cdf(z)), float(sr), float(g3), float(g4)


def run_dsr():
    em = 0.5772156649
    rows = []
    for u in all_universe_names():
        res = load(u)
        srs = np.array([res[m].net(COST).mean() / res[m].net(COST).std() for m in COMPLIANT])
        v = srs.var(ddof=1)
        sr0 = np.sqrt(v) * ((1 - em) * norm.ppf(1 - 1 / N_TRIALS) + em * norm.ppf(1 - 1 / (N_TRIALS * np.e)))
        r = res["CA-HRP"].net(COST)
        p0, sr, g3, g4 = psr(r, 0.0)
        pd_, _, _, _ = psr(r, sr0)
        rows.append({"universe": u, "SR_ann": sr * np.sqrt(252), "skew": g3, "kurt": g4,
                     "PSR_0": p0, "SR0_ann": sr0 * np.sqrt(252), "DSR": pd_, "T": len(r)})
    pd.DataFrame(rows).to_csv(TAB / "dsr.csv", index=False)
    print("[dsr] done", flush=True)


def run_calibration():
    methods = ["HRP", "MinVar-C", "MVO-C", "HRP-Proj", "CA-HRP", "CA-HRP-R"]
    rows = []
    for u in all_universe_names():
        res = load(u)
        rets, _ = load_universe(u)
        R = rets.values
        for m in methods:
            W = res[m].weights
            g = res[m].gross
            dates = list(W.index)
            z, pred, real = [], [], []
            for i, d in enumerate(dates[:-1]):
                t = rets.index.get_loc(d)
                cov = estimate(R[t - 251: t + 1], "lw")
                w = W.loc[d].values
                sig = np.sqrt(w @ cov @ w)
                seg = g.loc[(g.index > d) & (g.index <= dates[i + 1])]
                if len(seg) < 5:
                    continue
                h = len(seg)
                rm = (1 + seg).prod() - 1
                z.append(rm / (sig * np.sqrt(h)))
                pred.append(sig * np.sqrt(252))
                real.append(seg.std() * np.sqrt(252))
            z = np.array(z)
            rows.append({"universe": u, "method": m, "bias_stat": float(z.std()),
                         "mean_pred_vol": float(np.mean(pred)), "mean_real_vol": float(np.mean(real)),
                         "real_over_pred": float(np.mean(real) / np.mean(pred)), "months": len(z)})
        print(f"[calibration] {u}", flush=True)
    pd.DataFrame(rows).to_csv(TAB / "risk_calibration.csv", index=False)


if __name__ == "__main__":
    run_dsr()
    run_calibration()
    run_mcs()
