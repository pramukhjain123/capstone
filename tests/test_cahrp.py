import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cahrp import optim  # noqa: E402
from cahrp.constraints import make_constraints, violations  # noqa: E402
from cahrp.hrp import hrp  # noqa: E402
from cahrp.strategies import ca_hrp  # noqa: E402


def random_problem(seed, n=20, T=300, k=4):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 3))
    B = rng.normal(size=(3, n))
    vol = rng.uniform(0.3, 3.0, n)
    R = (F @ B + rng.normal(size=(T, n))) * vol / 100
    groups = [f"g{i % k}" for i in range(n)]
    return R, groups


@pytest.mark.parametrize("seed", range(25))
def test_capacity_bisection_meets_box_exactly(seed):
    R, groups = random_problem(seed)
    n = R.shape[1]
    u = np.random.default_rng(seed).uniform(1.2 / n, 3.0 / n)
    c = make_constraints(groups, upper=u, group_max=0.6)
    for split in ("list", "tree"):
        res = hrp(np.cov(R, rowvar=False), "single", split, constraints=c)
        w = res.weights
        assert abs(w.sum() - 1) < 1e-10
        assert (w >= -1e-12).all()
        assert (w <= c.upper + 1e-10).all()


@pytest.mark.parametrize("seed", range(10))
def test_ca_hrp_is_fully_feasible(seed):
    R, groups = random_problem(seed)
    c = make_constraints(groups, upper=0.12, group_max=0.30,
                         group_min={"g0": 0.2}, turnover_max=0.3)
    w0 = ca_hrp(R, c, None)
    assert violations(w0, c)["any"] == 0
    # next period, turnover must be respected relative to w0
    R2, _ = random_problem(seed + 100)
    w1 = ca_hrp(R2, c, w0)
    v = violations(w1, c, w0)
    assert v["any"] == 0, v


def test_projection_is_identity_when_feasible():
    R, groups = random_problem(1)
    c = make_constraints(groups, upper=1.0, group_max=1.0)
    res = hrp(np.cov(R, rowvar=False), "single", "list")
    w = optim.tree_projection(res.weights, res.nodes, c)
    assert np.allclose(w, res.weights, atol=1e-5)


def test_plain_hrp_matches_reference_properties():
    R, _ = random_problem(3)
    res = hrp(np.cov(R, rowvar=False))
    assert abs(res.weights.sum() - 1) < 1e-12
    assert (res.weights > 0).all()


def test_backtest_charges_costs_and_keeps_every_day():
    import pandas as pd
    from cahrp.backtest import run_backtest
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=600)
    rets = pd.DataFrame(rng.normal(0, 0.01, (600, 5)), index=idx, columns=list("abcde"))
    c = make_constraints(list("aabbc"), upper=1.0, group_max=1.0)
    flip = {"k": 0}

    def strat(R, c_, wp):  # alternates between two portfolios -> positive turnover
        flip["k"] += 1
        return np.array([1, 0, 0, 0, 0.0]) if flip["k"] % 2 else np.array([0, 1, 0, 0, 0.0])

    res = run_backtest(rets, strat, c, lookback=252)
    first = res.turnover.index.min()
    assert len(res.gross) == (rets.index > first).sum()          # no day skipped
    assert res.turnover.iloc[1:].min() > 1.5
    diff = res.gross.sum() - res.net(50).sum()
    assert np.isclose(diff, res.turnover.iloc[1:].sum() * 50 / 1e4)
