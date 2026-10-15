import numpy as np
import pandas as pd

from src.bootstrap import paired_bootstrap_wape_diff
from src.data import WeeklyData
from src.features import FeatureContext, build_rows
from src.prepare_weekly import first_sale_index
from src.run_lgbm import train_origins

T = 80


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    sales = rng.poisson(2.0, size=(3, T)).astype(np.int32)
    sales[2, :20] = 0
    price = np.full((3, T), 1.5, dtype=np.float32)
    price[2, :20] = np.nan
    ids = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "item_id": ["i1", "i2", "i3"],
            "dept_id": ["D", "D", "E"],
            "cat_id": ["C", "C", "C"],
            "store_id": ["CA_1", "TX_1", "CA_1"],
            "state_id": ["CA", "TX", "CA"],
        }
    )
    weeks = np.arange(11101, 11101 + T)
    cal = pd.DataFrame(
        {
            "wm_yr_wk": weeks,
            "week_start": pd.date_range("2011-01-29", periods=T, freq="7D"),
            "n_days": 7,
            "snap_CA": 3,
            "snap_TX": 5,
            "snap_WI": 0,
            "n_events": 0,
            "event_sporting": 0,
            "event_cultural": 0,
            "event_national": 0,
            "event_religious": 0,
        }
    )
    data = WeeklyData(ids=ids, sales=sales, price=price, first_sale=first_sale_index(sales), weeks=weeks)
    return data, cal


def test_features_do_not_use_future_sales():
    data, cal = make_data()
    t, h = 60, 3
    before = FeatureContext(data, cal).features(t, h)
    data.sales[:, t:] = 999  # change everything from the origin on
    after = FeatureContext(data, cal).features(t, h)
    for k in before:
        assert np.allclose(before[k], after[k], equal_nan=True), k


def test_feature_values():
    data, cal = make_data()
    ctx = FeatureContext(data, cal)
    f = ctx.features(60, 2)
    assert np.array_equal(f["lag_1"], data.sales[:, 59])
    assert np.array_equal(f["lag_seasonal"], data.sales[:, 60 + 2 - 1 - 52])
    assert np.allclose(f["mean_4"], data.sales[:, 56:60].mean(axis=1))
    assert f["snap_days"].tolist() == [3, 5, 3]


def test_build_rows_skips_unlaunched_series():
    data, cal = make_data()
    data.sales[2, :] = 0
    data.sales[2, 70] = 4
    data.first_sale = first_sale_index(data.sales)
    X, y, sids, _ = build_rows(FeatureContext(data, cal), [60], 1)
    assert sids.tolist() == [0, 1]
    assert len(X) == len(y) == 2


def test_train_origins_targets_before_test_window():
    for h in range(1, 5):
        o = train_origins(200, h, 52)
        assert (o + h - 1).max() == 199
        assert len(o) == 52


def test_bootstrap_identical_methods_has_zero_diff():
    rng = np.random.default_rng(1)
    err = rng.random(500)
    act = rng.random(500) + 1
    out = paired_bootstrap_wape_diff(err, err, act, n_boot=200)
    assert out["wape_diff"] == 0 and out["ci_low"] == 0 and out["ci_high"] == 0


def test_bootstrap_detects_clear_improvement():
    rng = np.random.default_rng(2)
    act = rng.random(1000) + 1
    out = paired_bootstrap_wape_diff(0.5 * act, 0.8 * act, act, n_boot=200)
    assert out["ci_high"] < 0 and out["share_a_better"] == 1.0
