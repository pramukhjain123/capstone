"""Covariance / correlation estimators."""
import numpy as np
from sklearn.covariance import LedoitWolf


def sample_cov(returns: np.ndarray) -> np.ndarray:
    return np.cov(returns, rowvar=False)


def ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage towards a scaled identity (Ledoit & Wolf, 2004)."""
    return LedoitWolf().fit(returns).covariance_


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def estimate(returns: np.ndarray, method: str = "sample") -> np.ndarray:
    if method == "sample":
        return sample_cov(returns)
    if method == "lw":
        return ledoit_wolf_cov(returns)
    raise ValueError(f"unknown covariance method {method!r}")
