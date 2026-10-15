import numpy as np

from src import baselines as bl
from src.metrics import rmsse, rmsse_scale, wape


def test_naive_and_seasonal_naive():
    train = np.arange(60).reshape(1, 60)
    fs = np.array([0])
    assert bl.naive(train, fs, 3).tolist() == [[59, 59, 59]]
    # 52 weeks before the origin (week 60) is week 8
    assert bl.seasonal_naive(train, fs, 3).tolist() == [[8, 9, 10]]


def test_moving_average_ignores_prelaunch_weeks():
    train = np.array([[0, 0, 0, 6, 2], [1, 2, 3, 4, 5]])
    fs = np.array([3, 0])
    out = bl.moving_average(train, fs, 2, window=4)
    assert np.allclose(out[0], 4.0)  # only weeks 3-4 count
    assert np.allclose(out[1], 3.5)  # weeks 1-4


def test_croston_on_regular_intermittent_demand():
    # demand of 6 every 3rd week -> rate 2 per week
    train = np.tile([6, 0, 0], 200).reshape(1, -1)
    out = bl.croston(train, np.array([0]), 1, alpha=0.2)
    assert abs(out[0, 0] - 2.0) < 0.05


def test_tsb_decays_after_demand_stops():
    train = np.array([[5] * 20 + [0] * 60])
    out = bl.tsb(train, np.array([0]), 1, alpha=0.1, beta=0.1)
    assert out[0, 0] < 0.05


def test_unlaunched_series_forecast_zero():
    train = np.zeros((1, 10), dtype=int)
    fs = np.array([10])  # never sold before the origin
    assert bl.croston(train, fs, 2).tolist() == [[0.0, 0.0]]
    assert bl.tsb(train, fs, 2).tolist() == [[0.0, 0.0]]


def test_wape():
    y = np.array([[2.0, 0.0], [4.0, 4.0]])
    yhat = np.array([[1.0, 1.0], [4.0, 2.0]])
    assert wape(y, yhat) == 4 / 10


def test_rmsse_scale_and_value():
    train = np.array([[0, 0, 1, 3, 1], [5, 5, 5, 5, 5]])
    fs = np.array([2, 0])
    scale = rmsse_scale(train, fs)
    assert np.isclose(scale[0], (4 + 4) / 2)  # diffs after launch: +2, -2
    assert np.isnan(scale[1])  # constant history -> undefined
    r = rmsse(np.array([[3.0, 1.0]]), np.array([[1.0, 1.0]]), scale[:1])
    assert np.isclose(r[0], np.sqrt(2.0 / 4.0))
