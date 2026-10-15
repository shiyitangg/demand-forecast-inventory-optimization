"""Load the processed weekly table as dense (series x week) matrices."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.prepare_weekly import first_sale_index


@dataclass
class WeeklyData:
    ids: pd.DataFrame  # one row per series, same order as matrix rows
    sales: np.ndarray  # int32, (n_series, n_weeks)
    price: np.ndarray  # float32, NaN when not on sale
    first_sale: np.ndarray  # week index of first positive sale per series
    weeks: np.ndarray = None  # wm_yr_wk label of each matrix column

    @property
    def n_weeks(self):
        return self.sales.shape[1]


def load_weekly():
    cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "week_idx", "wm_yr_wk", "sales", "sell_price"]
    df = pd.read_parquet(PROCESSED_DIR / "weekly.parquet", columns=cols)
    n_weeks = int(df["week_idx"].max()) + 1
    n_series = len(df) // n_weeks
    if n_series * n_weeks != len(df):
        raise ValueError("weekly table is not a full series x week grid")
    # prepare_weekly writes rows series-major and week-minor; verify instead of assuming.
    if not (df["week_idx"].to_numpy().reshape(n_series, n_weeks) == np.arange(n_weeks)).all():
        raise ValueError("weekly table is not sorted by series then week")
    ids = df.iloc[::n_weeks][cols[:6]].reset_index(drop=True)
    sales = df["sales"].to_numpy().reshape(n_series, n_weeks).astype(np.int32)
    price = df["sell_price"].to_numpy().reshape(n_series, n_weeks).astype(np.float32)
    demand_class = pd.read_parquet(PROCESSED_DIR / "demand_class.parquet")
    ids["demand_class"] = ids["id"].astype(str).map(demand_class.set_index("id")["demand_class"])
    weeks = df["wm_yr_wk"].to_numpy()[:n_weeks]
    return WeeklyData(ids=ids, sales=sales, price=price, first_sale=first_sale_index(sales), weeks=weeks)
