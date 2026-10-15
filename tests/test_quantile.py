import numpy as np

from src.features import FeatureContext
from src.quantile import (
    build_cumulative_rows,
    coverage,
    cumulative_features,
    cumulative_target,
    normal_quantiles,
    pinball,
    sort_quantiles,
)
from tests.test_features import make_data


def test_cumulative_target_sums_window():
    data, cal = make_data()
    ctx = FeatureContext(data, cal)
    assert np.array_equal(cumulative_target(ctx, 60, 3), data.sales[:, 60:63].sum(axis=1))


def test_cumulative_features_do_not_use_future_sales():
    data, cal = make_data()
    before = cumulative_features(FeatureContext(data, cal), 60, 3)
    data.sales[:, 60:] = 999
    after = cumulative_features(FeatureContext(data, cal), 60, 3)
    for k in before:
        assert np.allclose(before[k], after[k], equal_nan=True), k


def test_window_calendar_features():
    data, cal = make_data()
    f = cumulative_features(FeatureContext(data, cal), 60, 3)
    assert f["snap_window"].tolist() == [9, 15, 9]
    assert "price_target" not in f


def test_build_rows_shapes():
    data, cal = make_data()
    X, y, sid, orig = build_cumulative_rows(FeatureContext(data, cal), [60, 61], 2)
    assert len(X) == len(y) == len(sid) == len(orig) == 6


def test_pinball_and_coverage():
    y = np.array([10.0, 0.0])
    q = np.array([8.0, 2.0])
    assert np.allclose(pinball(y, q, 0.9), [0.9 * 2, 0.1 * 2])
    assert coverage(y, q) == 0.5


def test_sort_and_normal_quantiles():
    q = sort_quantiles(np.array([[3.0, 1.0, 2.0]]))
    assert q.tolist() == [[1.0, 2.0, 3.0]]
    nq = normal_quantiles(np.array([10.0, 1.0]), np.array([2.0, 5.0]), [0.5, 0.9])
    assert np.isclose(nq[0, 0], 10.0) and nq[0, 1] > 10.0
    assert nq.min() >= 0
