"""Option-implied volatility inputs for CA-HRP (CA-HRP-IV).

Each asset's volatility is rescaled by the implied-to-realised volatility ratio of the
option-implied volatility index of its asset class:

    sigma_i^IV(d) = sigma_i^hist(d) * IV_k(d-) / RV_k(d-)

where IV_k is the index level (VIX, VXN, GVZ, OVX, MOVE, India VIX) on the last trading day
strictly before the rebalance date d, and RV_k is the realised volatility of the index's
underlying over the 252 trading days strictly before d. Correlations stay historical
(Ledoit-Wolf). Assets with no mapped index (and indices without data yet) are scaled by the
median ratio of the available indices on that date. The mapping
below was fixed before any IV backtest was run.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .estimators import cov_to_corr, estimate

ROOT = Path(__file__).resolve().parents[2]

# index -> underlying used for its realised volatility
UNDERLYING = {"^VIX": "SPY", "^VXN": "QQQ", "^GVZ": "GLD", "^OVX": "USO",
              "^MOVE": "TNX_BP", "^INDIAVIX": "NIFTYBEES.NS"}

IV_MAP = {
    "etf": {"SPY": "^VIX", "IWM": "^VIX", "EFA": "^VIX", "EEM": "^VIX", "VGK": "^VIX", "EWJ": "^VIX",
            "VNQ": "^VIX", "RWX": "^VIX", "QQQ": "^VXN",
            "GLD": "^GVZ", "SLV": "^GVZ", "USO": "^OVX", "DBC": "^OVX",
            "TLT": "^MOVE", "IEF": "^MOVE", "SHY": "^MOVE", "TIP": "^MOVE", "LQD": "^MOVE"},
    # HYG and EMB (credit) are deliberately left unmapped: no suitable index.
    "india_etf": {"NIFTYBEES.NS": "^INDIAVIX", "BANKBEES.NS": "^INDIAVIX", "JUNIORBEES.NS": "^INDIAVIX",
                  "PSUBNKBEES.NS": "^INDIAVIX", "MON100.NS": "^VXN", "GOLDBEES.NS": "^GVZ"},
}


@lru_cache(maxsize=1)
def _iv_data():
    iv = pd.read_csv(ROOT / "data" / "raw" / "implied_vol.csv", index_col=0, parse_dates=True).sort_index()
    etf = pd.read_csv(ROOT / "data" / "raw" / "etf_prices.csv", index_col=0, parse_dates=True)
    ind = pd.read_csv(ROOT / "data" / "raw" / "india_etf_prices.csv", index_col=0, parse_dates=True)
    from .backtest import repair_split_glitches
    ind = repair_split_glitches(ind.ffill().dropna(how="any"))
    und = {}
    for k in ["SPY", "QQQ", "GLD", "USO"]:
        und[k] = etf[k].dropna().pct_change().dropna()
    und["NIFTYBEES.NS"] = ind["NIFTYBEES.NS"].pct_change().dropna()
    und["TNX_BP"] = (iv["^TNX"].dropna().diff() * 100).dropna()   # daily yield change in bp
    return iv.drop(columns="^TNX"), und


def iv_ratio(index: str, date: pd.Timestamp, window: int = 252) -> float:
    """IV / RV for one index, using only information strictly before `date`."""
    iv, und = _iv_data()
    s = iv[index].dropna()
    s = s[s.index < date]
    u = und[UNDERLYING[index]]
    u = u[u.index < date].iloc[-window:]
    if len(s) == 0 or len(u) < window // 2:
        return np.nan
    ivv = float(s.iloc[-1])
    if index == "^MOVE":
        rv = float(u.std() * np.sqrt(252))                 # bp p.a., same units as MOVE
    else:
        ivv = ivv / 100.0
        rv = float(u.std() * np.sqrt(252))
    return ivv / rv if rv > 0 else np.nan


def iv_covariance(R: np.ndarray, date: pd.Timestamp, tickers, universe: str) -> np.ndarray:
    """Covariance with historical (LW) correlations and IV-rescaled volatilities."""
    base = estimate(R, "lw")
    corr = cov_to_corr(base)
    sd = np.sqrt(np.diag(base))
    mp = IV_MAP.get(universe, {})
    ratios = {k: iv_ratio(k, date) for k in set(mp.values())}
    finite = [v for v in ratios.values() if np.isfinite(v)]
    # Unmapped assets (and indices without data yet) get the median ratio of the available
    # indices, so that the variance risk premium does not make them look artificially safe.
    neutral = float(np.median(finite)) if finite else 1.0
    scale = np.array([ratios.get(mp[t], np.nan) if t in mp else neutral for t in tickers])
    scale = np.where(np.isfinite(scale), scale, neutral)
    sd_iv = sd * scale
    return corr * np.outer(sd_iv, sd_iv)


def iv_ratios_frame(universe: str, dates) -> pd.DataFrame:
    idx = sorted(set(IV_MAP[universe].values()))
    return pd.DataFrame({k: [iv_ratio(k, d) for d in dates] for k in idx}, index=dates)
