"""Stage 4: quantile forecasts of cumulative demand over k = L + R weeks.

Run from the project root:  python -m src.run_quantile

For each fold, models are trained on targets that end before the fold's test
window, then applied at every weekly origin inside that window (features use
only weeks before each origin). Two methods are compared:
  lgbm_quantile  one LightGBM quantile model per tau
  normal         LightGBM tweedie mean + z_tau * sqrt(k) * std of the last 13 weeks

Scoring periods: "tuning" = origins whose target window ends before the
held-out fold; "holdout" = origins inside the held-out fold.

Outputs
  reports/stage4_calibration.csv           coverage and pinball per method, k, tau, period
  reports/stage4_by_class.csv              coverage for the main k by demand class (holdout)
  data/processed/quantile_forecasts.parquet forecasts used by the inventory simulation
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    LEAD_TIMES,
    LGBM_PARAMS,
    MAIN_LEAD_TIME,
    N_TRAIN_ORIGINS_Q,
    PROCESSED_DIR,
    Q_NUM_ROUNDS,
    QUANTILES,
    REPORTS_DIR,
    REVIEW_WEEKS,
)
from src.data import load_weekly
from src.features import CAT_FEATURES, FeatureContext
from src.folds import make_folds
from src.quantile import build_cumulative_rows, coverage, normal_quantiles, pinball, sort_quantiles

BASE = {k: v for k, v in LGBM_PARAMS.items() if k not in ("objective", "tweedie_variance_power")}


def train_model(X, y, params):
    return lgb.train(params, lgb.Dataset(X, y, categorical_feature=CAT_FEATURES), num_boost_round=Q_NUM_ROUNDS)


def forecast_fold(ctx, fold, k, n_weeks):
    last_train_origin = fold.test_start - k  # its target ends at test_start - 1
    train_o = np.arange(max(52, last_train_origin - N_TRAIN_ORIGINS_Q + 1), last_train_origin + 1)
    pred_o = [t for t in range(fold.test_start, fold.test_end) if t + k <= n_weeks]
    if not pred_o:
        return None
    X, y, _, _ = build_cumulative_rows(ctx, train_o, k)
    Xp, yp, sid, orig = build_cumulative_rows(ctx, pred_o, k)

    mean_model = train_model(X, y, {**BASE, "objective": "tweedie", "tweedie_variance_power": 1.1})
    mean = np.clip(mean_model.predict(Xp), 0, None)
    q = np.column_stack(
        [np.clip(train_model(X, y, {**BASE, "objective": "quantile", "alpha": tau}).predict(Xp), 0, None) for tau in QUANTILES]
    )
    q = sort_quantiles(q)
    sigma = np.sqrt(k) * np.nan_to_num(Xp["std_13"].to_numpy(dtype=np.float64), nan=0.0)

    out = pd.DataFrame({"sid": sid.astype(np.int32), "fold": fold.fold, "origin": orig.astype(np.int16), "k": k})
    out["actual"] = yp.astype(np.float32)
    out["mean"] = mean.astype(np.float32)
    out["sigma"] = sigma.astype(np.float32)
    for j, tau in enumerate(QUANTILES):
        out[f"q{tau}"] = q[:, j].astype(np.float32)
    return out


def score(df, holdout_start):
    rows = []
    period = np.where(df["origin"] >= holdout_start, "holdout", "tuning")
    ends_before_holdout = (df["origin"] + df["k"]) <= holdout_start
    keep = (period == "holdout") | ends_before_holdout
    df, period = df[keep], period[keep]
    for (k, per), g in df.groupby([df["k"], period]):
        y = g["actual"].to_numpy(np.float64)
        normal = normal_quantiles(g["mean"].to_numpy(np.float64), g["sigma"].to_numpy(np.float64), QUANTILES)
        for j, tau in enumerate(QUANTILES):
            for method, qcol in [("lgbm_quantile", g[f"q{tau}"].to_numpy(np.float64)), ("normal", normal[:, j])]:
                rows.append(
                    {
                        "k": int(k),
                        "period": per,
                        "method": method,
                        "tau": tau,
                        "coverage": round(coverage(y, qcol), 4),
                        "pinball": round(float(pinball(y, qcol, tau).mean()), 4),
                        "pinball_scaled": round(float(pinball(y, qcol, tau).mean() / y.mean()), 4),
                        "n": int(len(y)),
                    }
                )
    return pd.DataFrame(rows)


def main():
    data = load_weekly()
    ctx = FeatureContext(data, pd.read_parquet(PROCESSED_DIR / "calendar_weekly.parquet"))
    folds = make_folds(data.n_weeks)
    holdout_start = [f for f in folds if not f.use_for_tuning][0].test_start

    frames = []
    for k in [L + REVIEW_WEEKS for L in LEAD_TIMES]:
        for fold in folds:
            t0 = time.time()
            out = forecast_fold(ctx, fold, k, data.n_weeks)
            if out is not None:
                frames.append(out)
                print(f"k={k} fold {fold.fold}: {out['origin'].nunique()} origins, {time.time() - t0:.0f}s")
    fc = pd.concat(frames, ignore_index=True)
    fc.to_parquet(PROCESSED_DIR / "quantile_forecasts.parquet", index=False)

    cal = score(fc, holdout_start)
    cal.to_csv(REPORTS_DIR / "stage4_calibration.csv", index=False)

    main_k = MAIN_LEAD_TIME + REVIEW_WEEKS
    h = fc[(fc["k"] == main_k) & (fc["origin"] >= holdout_start)].copy()
    h["demand_class"] = data.ids["demand_class"].to_numpy()[h["sid"]]
    by_class = h.groupby("demand_class").apply(
        lambda g: pd.Series({f"cov_q{t}": round(float(np.mean(g["actual"] <= g[f"q{t}"])), 3) for t in QUANTILES}),
        include_groups=False,
    )
    by_class.to_csv(REPORTS_DIR / "stage4_by_class.csv")

    pd.set_option("display.width", 160)
    for per in ["tuning", "holdout"]:
        t = cal[(cal["k"] == main_k) & (cal["period"] == per)]
        print(f"\nk={main_k} ({per}): coverage vs nominal tau")
        print(t.pivot(index="tau", columns="method", values="coverage").to_string())
        print(f"k={main_k} ({per}): scaled pinball loss (lower is better)")
        print(t.pivot(index="tau", columns="method", values="pinball_scaled").to_string())
    print(f"\nk={main_k} holdout coverage of lgbm_quantile by demand class:")
    print(by_class.to_string())


if __name__ == "__main__":
    main()
