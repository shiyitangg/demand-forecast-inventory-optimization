import numpy as np
import pandas as pd
import pytest

from src.config import PROCESSED_DIR
from src.prepare_weekly import aggregate_days_to_weeks, demand_classes, first_sale_index, week_starts


def test_week_starts_and_sum():
    day_weeks = np.array([1, 1, 1, 2, 2, 3])
    starts = week_starts(day_weeks)
    assert starts.tolist() == [0, 3, 5]
    vals = np.array([[1, 2, 3, 4, 5, 6], [0, 0, 1, 0, 0, 0]])
    out = aggregate_days_to_weeks(vals, starts)
    assert out.tolist() == [[6, 9, 6], [1, 0, 0]]
    assert out.sum() == vals.sum()


def test_first_sale_index():
    sales = np.array([[0, 0, 3, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    assert first_sale_index(sales).tolist() == [2, 0, 4]


def test_demand_classes():
    sales = np.array(
        [
            [5, 5, 5, 5, 5, 5],  # every week, constant size -> smooth
            [0, 4, 0, 0, 4, 0],  # sparse, constant size -> intermittent
            [0, 0, 0, 0, 0, 7],  # one sale -> insufficient
        ]
    )
    classes = demand_classes(sales, first_sale_index(sales), end=6)
    assert classes.tolist() == ["smooth", "intermittent", "insufficient"]


WEEKLY = PROCESSED_DIR / "weekly.parquet"


@pytest.mark.skipif(not WEEKLY.exists(), reason="run python -m src.prepare_weekly first")
def test_processed_weekly_table():
    df = pd.read_parquet(WEEKLY, columns=["id", "week_idx", "sales", "on_sale"])
    n_series = df["id"].nunique()
    n_weeks = df["week_idx"].nunique()
    assert len(df) == n_series * n_weeks
    assert not df.duplicated(["id", "week_idx"]).any()
    assert df["sales"].notna().all()
    assert (df["sales"] >= 0).all()
