# Constraint-Aware Hierarchical Risk Parity for Robust Portfolio Allocation

B.Tech capstone project — Pramukh Rajendra Jain (23BPS1114), VIT Chennai.

Hierarchical Risk Parity (HRP) builds diversified portfolios without inverting a
covariance matrix, but its recursive bisection has no way to respect the rules real
portfolios must follow: per-asset limits, sector / asset-class caps and floors, and
turnover budgets. In practice HRP output is "fixed" by clipping and renormalising,
which breaks constraints silently and distorts the risk allocation.

**Constraint-Aware HRP (CA-HRP)** keeps HRP's hierarchical risk allocation as the
target and enforces every constraint exactly:

1. **Robust estimation** — Ledoit–Wolf shrinkage covariance.
2. **Hierarchical risk allocation** — standard HRP tree clustering + recursive bisection.
3. **Tracking-error projection** — the feasible portfolio (box, group caps/floors,
   turnover) whose ex-ante returns track the HRP portfolio most closely,
   `min (w − w_HRP)' Σ (w − w_HRP)`. Weight removed from capped assets is re-assigned
   to the assets that best replicate their risk.

A solver-free variant, **CA-HRP-B**, clips each bisection split into a provably
non-empty *capacity interval* so box limits hold exactly by construction
(Proposition 1 in the paper).

## Headline results

Read the full paper: [`paper/main.pdf`](paper/main.pdf)

| Multi-asset ETFs (constraints bind) | Sharpe (full) | Sharpe (held-out test) | Vol | Max DD | Constraint breaches |
|---|---|---|---|---|---|
| HRP (unconstrained) | 0.833 | 0.830 | 2.0% | −8.8% | 100% of rebalances |
| HRP-Clip (common fix) | 0.709 | 0.802 | 9.2% | −22.8% | 46% |
| HRP-Proj (Euclidean) | 0.764 | 0.805 | 9.9% | −22.8% | 0% |
| Constrained min-variance | 0.935 | 0.977 | 7.3% | −17.5% | 0% |
| **CA-HRP (this work)** | **0.908** | **0.929** | **7.4%** | **−18.8%** | **0%** |

All figures are net of 10 bps transaction costs. In the NIFTY 50 and US large-cap universes,
where the constraints rarely bind, CA-HRP matches HRP and is the top method for US stocks
(Sharpe 0.793). Across 630 rebalances CA-HRP never breached a constraint.

## Repository layout

| Path | Contents |
|---|---|
| `data/raw/` | Daily adjusted closes + group maps for 3 universes (multi-asset ETFs, NIFTY 50 stocks, US large caps), 2006–2026 |
| `src/cahrp/` | Library: estimators, HRP & capacity-aware bisection, projections, baselines, backtester, metrics |
| `scripts/download_data.py` | Re-downloads the price data (Yahoo Finance) |
| `scripts/design_study.py` | Chooses the CA-HRP design on the **validation** period only |
| `scripts/run_experiments.py` | All backtests, ablations, lookback robustness, stability and constraint-tightness studies |
| `scripts/summarise.py` | Builds every table and figure |
| `results/tables/`, `results/figures/` | Outputs used in the paper |
| `paper/` | LaTeX source and compiled PDF of the research paper |
| `tests/` | Unit tests (feasibility guarantees, projection identity, HRP sanity) |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests
python scripts/design_study.py          # validation-period design choice
python scripts/run_experiments.py all   # ~15 min on a laptop
python scripts/summarise.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Protocol

Monthly rebalancing, 252-day estimation window, weights drift between rebalances,
10 bps cost per unit of two-sided turnover. Constraint sets were fixed before running
any backtest (`src/cahrp/backtest.py`, `CONSTRAINT_SPECS`). The CA-HRP design was
selected on the first part of each sample (to 2016 / 2018 / 2015) and the later
period is reported separately as a held-out test.
