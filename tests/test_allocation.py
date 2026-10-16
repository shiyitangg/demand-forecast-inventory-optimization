import numpy as np
import pytest

from src.allocation import expected_sold, segments, solve_greedy, solve_lp

TAUS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]


def test_segments_are_concave_and_cover_quantiles():
    q = np.array([[2.0, 3.0, 4.0, 5.0, 6.0, 8.0]])
    lengths, slopes = segments(q, TAUS)
    assert lengths.sum() == 8.0
    assert np.all(np.diff(slopes) < 0)
    assert slopes[0] == pytest.approx(0.75)


def test_expected_sold_is_monotone_and_capped():
    q = np.array([[2.0, 3.0, 4.0, 5.0, 6.0, 8.0]] * 3)
    e = expected_sold(np.array([0.0, 2.0, 100.0]), q, TAUS)
    assert e[0] == 0 and e[1] == pytest.approx(1.5)
    assert e[2] == pytest.approx(expected_sold(np.array([8.0]), q[:1], TAUS)[0])


def test_lp_matches_greedy_and_respects_budget():
    rng = np.random.default_rng(0)
    n = 60
    q = np.sort(rng.gamma(2.0, 3.0, size=(n, len(TAUS))), axis=1)
    price = rng.uniform(0.5, 10, n)
    group = np.array(["A", "B", "C"])[rng.integers(0, 3, n)]
    budgets = {g: 0.4 * np.sum((price * q[:, 3])[group == g]) for g in "ABC"}
    lengths, slopes = segments(q, TAUS)
    x_lp = solve_lp(lengths, slopes, price, group, budgets)
    x_gr = solve_greedy(lengths, slopes, price, group, budgets)
    obj_lp = expected_sold(x_lp, q, TAUS).sum()
    obj_gr = expected_sold(x_gr, q, TAUS).sum()
    assert obj_lp == pytest.approx(obj_gr, rel=1e-6)
    for g in "ABC":
        assert np.sum((price * x_lp)[group == g]) <= budgets[g] + 1e-6
    # LP should beat scaling everyone's q0.9 to the same budget
    assert obj_lp >= expected_sold(0.4 * q[:, 3], q, TAUS).sum() - 1e-9


def test_large_budget_stocks_top_quantile():
    q = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    lengths, slopes = segments(q, TAUS)
    x = solve_lp(lengths, slopes, np.array([1.0]), np.array(["A"]), {"A": 1000.0})
    assert x[0] == pytest.approx(6.0)
