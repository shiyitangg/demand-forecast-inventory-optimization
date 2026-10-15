"""Stage 1: aggregate M5 daily sales to Walmart weeks and profile the data.

Run from the project root:  python -m src.prepare_weekly

Outputs
  data/processed/weekly.parquet          one row per (series, full week)
  data/processed/calendar_weekly.parquet one row per Walmart week
  reports/stage1_summary.json            measured data-quality numbers
  reports/folds.json                     backtest fold boundaries
"""
import json

import numpy as np
import pandas as pd

from src.config import ADI_CUTOFF, CV2_CUTOFF, PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from src.folds import make_folds

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CHUNK_ROWS = 5000


def week_starts(day_weeks):
    """Positions where the week label changes in a day-ordered array of week labels."""
    day_weeks = np.asarray(day_weeks)
    change = np.flatnonzero(day_weeks[1:] != day_weeks[:-1]) + 1
    return np.concatenate([[0], change])


def aggregate_days_to_weeks(values, starts):
    """Sum a (series x day) matrix into (series x week) using week start positions."""
    return np.add.reduceat(values, starts, axis=1)


def load_calendar():
    cal = pd.read_csv(RAW_DIR / "calendar.csv", parse_dates=["date"])
    if not cal["wm_yr_wk"].is_monotonic_increasing:
        raise ValueError("calendar weeks are not in date order")
    return cal


def build_calendar_weekly(cal):
    cal = cal.copy()
    cal["n_events"] = cal[["event_name_1", "event_name_2"]].notna().sum(axis=1)
    agg = {
        "week_start": ("date", "min"),
        "n_days": ("date", "size"),
        "snap_CA": ("snap_CA", "sum"),
        "snap_TX": ("snap_TX", "sum"),
        "snap_WI": ("snap_WI", "sum"),
        "n_events": ("n_events", "sum"),
    }
    weekly = cal.groupby("wm_yr_wk", sort=True).agg(**agg).reset_index()
    for etype in ["Sporting", "Cultural", "National", "Religious"]:
        has = (cal["event_type_1"].eq(etype) | cal["event_type_2"].eq(etype)).groupby(cal["wm_yr_wk"]).any()
        weekly[f"event_{etype.lower()}"] = weekly["wm_yr_wk"].map(has).astype("int8")
    return weekly


def load_weekly_sales(cal):
    """Read the wide daily file in chunks; return ids, weekly matrix, week labels, raw stats."""
    header = pd.read_csv(RAW_DIR / "sales_train_evaluation.csv", nrows=0).columns
    d_cols = [c for c in header if c.startswith("d_")]
    day_to_week = cal.set_index("d")["wm_yr_wk"]
    expected = cal["d"].iloc[: len(d_cols)].tolist()
    if d_cols != expected:
        raise ValueError("day columns are not in calendar order")
    day_weeks = day_to_week.loc[d_cols].to_numpy()
    starts = week_starts(day_weeks)
    week_labels = day_weeks[starts]
    days_per_week = np.diff(np.concatenate([starts, [len(d_cols)]]))

    dtypes = {c: "float32" for c in d_cols}  # float so missing values stay visible
    ids, mats = [], []
    n_missing = 0
    n_negative = 0
    reader = pd.read_csv(RAW_DIR / "sales_train_evaluation.csv", dtype=dtypes, chunksize=CHUNK_ROWS)
    for chunk in reader:
        vals = chunk[d_cols].to_numpy()
        n_missing += int(np.isnan(vals).sum())
        n_negative += int((vals < 0).sum())
        vals = np.nan_to_num(vals, nan=0.0).astype(np.int32)
        mats.append(aggregate_days_to_weeks(vals, starts))
        ids.append(chunk[ID_COLS])
    ids = pd.concat(ids, ignore_index=True)
    weekly = np.vstack(mats)

    full = days_per_week == 7
    raw_stats = {
        "n_days": len(d_cols),
        "first_day": str(cal.loc[cal["d"] == d_cols[0], "date"].iloc[0].date()),
        "last_day": str(cal.loc[cal["d"] == d_cols[-1], "date"].iloc[0].date()),
        "n_weeks_total": int(len(week_labels)),
        "n_partial_weeks_dropped": int((~full).sum()),
        "partial_weeks_dropped": [int(w) for w in week_labels[~full]],
        "n_missing_daily_values": n_missing,
        "n_negative_daily_values": n_negative,
    }
    return ids, weekly[:, full], week_labels[full], raw_stats


def price_matrix(ids, week_labels):
    """(series x week) weekly sell price; NaN where the item was not on sale that week."""
    prices = pd.read_csv(
        RAW_DIR / "sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    # Look up each price row's series index through an (item x store) table of
    # integer codes, which avoids building millions of Python strings.
    items = pd.Categorical(ids["item_id"])
    stores = pd.Categorical(ids["store_id"])
    sid_table = np.full((len(items.categories), len(stores.categories)), -1, dtype=np.int32)
    sid_table[items.codes, stores.codes] = np.arange(len(ids), dtype=np.int32)
    item_codes = pd.Categorical(prices["item_id"], categories=items.categories).codes
    store_codes = pd.Categorical(prices["store_id"], categories=stores.categories).codes
    known = (item_codes >= 0) & (store_codes >= 0)
    sid = np.full(len(prices), -1, dtype=np.int32)
    sid[known] = sid_table[item_codes[known], store_codes[known]]
    prices["sid"] = sid
    week_pos = pd.Series(np.arange(len(week_labels)), index=week_labels)
    prices = prices[(prices["sid"] >= 0) & prices["wm_yr_wk"].isin(week_pos.index)]
    mat = np.full((len(ids), len(week_labels)), np.nan, dtype=np.float32)
    mat[prices["sid"].to_numpy(), week_pos.loc[prices["wm_yr_wk"]].to_numpy()] = prices["sell_price"].to_numpy()
    return mat


def first_sale_index(sales):
    """Index of the first week with positive sales per row; n_weeks if never sold."""
    pos = sales > 0
    return np.where(pos.any(axis=1), pos.argmax(axis=1), sales.shape[1])


def demand_classes(sales, first_sale, end):
    """Syntetos-Boylan class per series using weeks from first sale up to `end` (exclusive)."""
    classes = []
    for row, start in zip(sales[:, :end], first_sale):
        hist = row[start:]
        nz = hist[hist > 0]
        if len(nz) < 2:
            classes.append("insufficient")
            continue
        adi = len(hist) / len(nz)
        cv2 = (nz.std() / nz.mean()) ** 2
        if adi < ADI_CUTOFF:
            classes.append("smooth" if cv2 < CV2_CUTOFF else "erratic")
        else:
            classes.append("intermittent" if cv2 < CV2_CUTOFF else "lumpy")
    return np.array(classes)


def to_long(ids, sales, prices, week_labels, first_sale):
    n_series, n_weeks = sales.shape
    week_idx = np.tile(np.arange(n_weeks, dtype=np.int16), n_series)
    long = pd.DataFrame()
    for c in ID_COLS:
        cats = pd.Categorical(ids[c])
        long[c] = pd.Categorical.from_codes(np.repeat(cats.codes, n_weeks), cats.categories)
    long["wm_yr_wk"] = np.tile(week_labels.astype(np.int32), n_series)
    long["week_idx"] = week_idx
    long["sales"] = sales.ravel().astype(np.int32)
    long["sell_price"] = prices.ravel()
    long["on_sale"] = ~np.isnan(long["sell_price"].to_numpy())
    long["after_first_sale"] = week_idx >= np.repeat(first_sale, n_weeks)
    return long


def share(x):
    return round(float(np.mean(x)), 4)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cal = load_calendar()
    cal_weekly = build_calendar_weekly(cal)
    ids, sales, week_labels, raw_stats = load_weekly_sales(cal)
    prices = price_matrix(ids, week_labels)
    first_sale = first_sale_index(sales)
    n_series, n_weeks = sales.shape

    folds = make_folds(n_weeks)
    tuning_end = folds[0].train_end  # classes use only data before the first test window
    classes = demand_classes(sales, first_sale, tuning_end)

    on_sale = ~np.isnan(prices)
    after_first = np.arange(n_weeks)[None, :] >= first_sale[:, None]
    zero = sales == 0
    by_cat = {}
    for cat in sorted(ids["cat_id"].unique()):
        rows = (ids["cat_id"] == cat).to_numpy()
        by_cat[cat] = share(zero[rows][after_first[rows]])

    summary = {
        **raw_stats,
        "n_series": int(n_series),
        "n_full_weeks": int(n_weeks),
        "n_rows_weekly": int(n_series * n_weeks),
        "first_full_week": int(week_labels[0]),
        "last_full_week": int(week_labels[-1]),
        "zero_share_all_weeks": share(zero),
        "zero_share_after_first_sale": share(zero[after_first]),
        "zero_share_after_first_sale_by_category": by_cat,
        "leading_zero_share": share(~after_first),
        "price_missing_share": share(~on_sale),
        "weeks_with_sales_but_no_price": int((~on_sale & (sales > 0)).sum()),
        "series_never_sold": int((first_sale == n_weeks).sum()),
        "demand_class_counts_pre_test": {k: int(v) for k, v in zip(*np.unique(classes, return_counts=True))},
        "weekly_sales_quantiles": {
            q: float(np.quantile(sales[after_first], float(q))) for q in ["0.5", "0.9", "0.99"]
        },
        "weekly_sales_max": int(sales.max()),
    }

    long = to_long(ids, sales, prices, week_labels, first_sale)
    long.to_parquet(PROCESSED_DIR / "weekly.parquet", index=False)
    cal_weekly.to_parquet(PROCESSED_DIR / "calendar_weekly.parquet", index=False)
    pd.DataFrame({"id": ids["id"], "demand_class": classes}).to_parquet(
        PROCESSED_DIR / "demand_class.parquet", index=False
    )

    week_dates = cal_weekly.set_index("wm_yr_wk")["week_start"]
    fold_rows = []
    for f in folds:
        d = f.to_dict()
        d["test_first_week"] = int(week_labels[f.test_start])
        d["test_last_week"] = int(week_labels[f.test_end - 1])
        d["test_start_date"] = str(week_dates.loc[week_labels[f.test_start]].date())
        fold_rows.append(d)

    (REPORTS_DIR / "stage1_summary.json").write_text(json.dumps(summary, indent=2))
    (REPORTS_DIR / "folds.json").write_text(json.dumps(fold_rows, indent=2))
    print(json.dumps(summary, indent=2))
    print(json.dumps(fold_rows, indent=2))


if __name__ == "__main__":
    main()
