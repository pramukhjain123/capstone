"""Performance, risk and portfolio-structure metrics."""
import numpy as np
import pandas as pd

from .estimators import ledoit_wolf_cov

ANN = 252


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def cvar(r: pd.Series, q: float = 0.05) -> float:
    cut = r.quantile(q)
    return float(r[r <= cut].mean())


def perf(r: pd.Series) -> dict:
    years = len(r) / ANN
    cagr = (1 + r).prod() ** (1 / years) - 1
    vol = r.std() * np.sqrt(ANN)
    down = r[r < 0].std() * np.sqrt(ANN)
    mdd = max_drawdown(r)
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": (r.mean() * ANN) / vol if vol > 0 else np.nan,
        "Sortino": (r.mean() * ANN) / down if down > 0 else np.nan,
        "MaxDD": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "CVaR5": cvar(r),
    }


def effective_n(w: np.ndarray) -> float:
    return float(1.0 / np.sum(w ** 2))


def risk_concentration(w: np.ndarray, cov: np.ndarray) -> float:
    """Herfindahl index of percentage risk contributions (1/n = perfectly balanced)."""
    rc = w * (cov @ w)
    rc = rc / rc.sum()
    return float(np.sum(rc ** 2))


def structure(weights: pd.DataFrame, rets: pd.DataFrame, lookback: int = 252) -> dict:
    effn, rcon = [], []
    for d, w in weights.iterrows():
        t = rets.index.get_loc(d)
        cov = ledoit_wolf_cov(rets.values[t - lookback + 1: t + 1])
        effn.append(effective_n(w.values))
        rcon.append(risk_concentration(w.values, cov) * len(w))  # 1.0 = equal risk
    return {"EffN": float(np.mean(effn)), "RiskConc": float(np.mean(rcon))}


def sharpe_diff_ci(r1: pd.Series, r2: pd.Series, block: int = 21, n_boot: int = 2000,
                   seed: int = 0, level: float = 0.95):
    """Annualised Sharpe difference with a circular-block-bootstrap percentile interval."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([r1.values, r2.values])
    T = len(x)
    ann = np.sqrt(ANN)

    def d(a):
        s = a.mean(0) / a.std(0) * ann
        return s[0] - s[1]

    nb = int(np.ceil(T / block))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T, nb)
        ids = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        boots[b] = d(x[ids])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(d(x)), float(lo), float(hi), float(np.mean(boots > 0))


def sharpe_diff_test(r1: pd.Series, r2: pd.Series, block: int = 21, n_boot: int = 2000,
                     seed: int = 0) -> float:
    """Two-sided p-value for Sharpe(r1) - Sharpe(r2) = 0 via stationary block bootstrap
    (Ledoit & Wolf, 2008 style, circular blocks)."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([r1.values, r2.values])
    T = len(x)

    def sr(a):
        return a.mean(0) / a.std(0)

    obs = sr(x)
    d_obs = obs[0] - obs[1]
    diffs = np.empty(n_boot)
    nb = int(np.ceil(T / block))
    for b in range(n_boot):
        starts = rng.integers(0, T, nb)
        ids = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        s = sr(x[ids])
        diffs[b] = s[0] - s[1]
    centred = diffs - diffs.mean()
    return float(np.mean(np.abs(centred) >= abs(d_obs)))
