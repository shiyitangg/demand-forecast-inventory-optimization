"""Stage 2: evaluate baseline forecasters on the rolling-origin folds.

Run from the project root:  python -m src.run_baselines

Choices (MA window, best baseline) are made on the tuning folds only; the
held-out fold is reported separately and never used to pick anything.

Outputs
  reports/stage2_baselines.csv          metrics per method and fold
  reports/stage2_by_class.csv           WAPE per demand class (tuning folds)
  reports/stage2_summary.json           tuning vs holdout means, selections
  data/processed/baseline_errors.parquet per-series errors (for bootstrap later)
"""
import json
from functools import partial

import numpy as np
import pandas as pd

from src import baselines as bl
from src.config import PROCESSED_DIR, REPORTS_DIR
from src.data import load_weekly
from src.folds import make_folds
from src.metrics import rmsse, rmsse_scale, wape, weighted_mean

MA_WINDOWS = [4, 8, 13, 26]
WEIGHT_WEEKS = 4  # revenue weights from the last weeks before each origin

METHODS = {
    "naive": bl.naive,
    "seasonal_naive": bl.seasonal_naive,
    **{f"ma_{w}": partial(bl.moving_average, window=w) for w in MA_WINDOWS},
    "croston": bl.croston,
    "tsb": bl.tsb,
}


def evaluate_fold(data, fold):
    train = data.sales[:, : fold.train_end]
    y = data.sales[:, fold.test_start : fold.test_end].astype(np.float64)
    horizon = y.shape[1]
    # Only series that had launched before the origin can be forecast from history.
    launched = data.first_sale < fold.train_end
    scale = rmsse_scale(train, data.first_sale)
    recent = slice(fold.train_end - WEIGHT_WEEKS, fold.train_end)
    weights = np.sum(train[:, recent] * np.nan_to_num(data.price[:, recent]), axis=1)

    rows, per_series = [], []
    for name, fn in METHODS.items():
        yhat = fn(train, data.first_sale, horizon)
        r = rmsse(y, yhat, scale)
        ys, fs, rs = y[launched], yhat[launched], r[launched]
        rows.append(
            {
                "fold": fold.fold,
                "use_for_tuning": fold.use_for_tuning,
                "method": name,
                "wape": wape(ys, fs),
                "rmsse_mean": float(np.nanmean(rs)),
                "rmsse_weighted": weighted_mean(rs, weights[launched]),
                "n_series": int(launched.sum()),
                "n_rmsse_undefined": int(np.isnan(rs).sum()),
            }
        )
        per_series.append(
            pd.DataFrame(
                {
                    "sid": np.flatnonzero(launched).astype(np.int32),
                    "fold": np.int8(fold.fold),
                    "method": name,
                    "abs_err": np.abs(ys - fs).sum(axis=1).astype(np.float32),
                    "actual": ys.sum(axis=1).astype(np.float32),
                    "rmsse": rs.astype(np.float32),
                }
            )
        )
    return rows, pd.concat(per_series, ignore_index=True)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_weekly()
    folds = make_folds(data.n_weeks)

    rows, errors = [], []
    for fold in folds:
        r, e = evaluate_fold(data, fold)
        rows += r
        errors.append(e)
        print(f"fold {fold.fold} done")
    res = pd.DataFrame(rows)
    errors = pd.concat(errors, ignore_index=True)
    errors["method"] = errors["method"].astype("category")
    errors.to_parquet(PROCESSED_DIR / "baseline_errors.parquet", index=False)
    res.to_csv(REPORTS_DIR / "stage2_baselines.csv", index=False)

    metrics = ["wape", "rmsse_mean", "rmsse_weighted"]
    tuning = res[res["use_for_tuning"]].groupby("method")[metrics].mean()
    holdout = res[~res["use_for_tuning"]].set_index("method")[metrics]

    best_ma = tuning.loc[[f"ma_{w}" for w in MA_WINDOWS], "wape"].idxmin()
    best = tuning["wape"].idxmin()

    # WAPE by demand class, pooled over tuning folds
    tun = errors[errors["fold"].isin([f.fold for f in folds if f.use_for_tuning])].copy()
    tun["demand_class"] = data.ids["demand_class"].to_numpy()[tun["sid"]]
    by_class = (
        tun.groupby(["demand_class", "method"], observed=True)[["abs_err", "actual"]].sum()
        .assign(wape=lambda d: d["abs_err"] / d["actual"])["wape"]
        .unstack("method")
        .round(4)
    )
    by_class.to_csv(REPORTS_DIR / "stage2_by_class.csv")

    summary = {
        "tuning_folds_mean": tuning.round(4).to_dict(orient="index"),
        "holdout_fold": holdout.round(4).to_dict(orient="index"),
        "selected_ma_window_by_tuning_wape": best_ma,
        "best_baseline_by_tuning_wape": best,
        "n_series_evaluated_per_fold": res.groupby("fold")["n_series"].first().to_dict(),
    }
    (REPORTS_DIR / "stage2_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\nTuning folds (1-3), mean:")
    print(tuning.round(4).sort_values("wape"))
    print("\nHoldout fold (4):")
    print(holdout.round(4).sort_values("wape"))
    print("\nWAPE by demand class (tuning folds pooled):")
    print(by_class)
    print(f"\nselected MA window: {best_ma}; best baseline on tuning folds: {best}")


if __name__ == "__main__":
    main()
