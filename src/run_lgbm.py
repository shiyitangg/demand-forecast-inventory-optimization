"""Stage 3: global LightGBM (one direct model per horizon step) vs the best baseline.

Run from the project root:  python -m src.run_lgbm

Outputs
  reports/stage3_lgbm.csv              metrics per fold (same definitions as stage 2)
  reports/stage3_bootstrap.json        paired bootstrap of WAPE(lgbm) - WAPE(baseline)
  reports/stage3_importance.csv        gain importance, holdout fold models
  data/processed/lgbm_predictions.parquet  point forecasts (used in later stages)
  data/processed/lgbm_errors.parquet       per-series errors
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.bootstrap import paired_bootstrap_wape_diff
from src.config import (
    BASELINE_FOR_COMPARISON,
    HORIZON_WEEKS,
    LGBM_NUM_ROUNDS,
    LGBM_PARAMS,
    N_BOOT,
    N_TRAIN_ORIGINS,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from src.data import load_weekly
from src.features import CAT_FEATURES, FeatureContext, build_rows
from src.folds import make_folds
from src.metrics import rmsse, rmsse_scale, wape, weighted_mean
from src.run_baselines import WEIGHT_WEEKS


def train_origins(test_start, h, n_origins):
    """Origins whose h-step target falls before the test window."""
    last = test_start - h  # target week last + h - 1 = test_start - 1
    first = max(52, last - n_origins + 1)
    return np.arange(first, last + 1)


def fit_predict_fold(ctx, fold, n_origins=N_TRAIN_ORIGINS, num_rounds=LGBM_NUM_ROUNDS, params=LGBM_PARAMS):
    """Return (sids, yhat matrix [series x horizon], importance frame)."""
    preds, importances, sids = [], [], None
    for h in range(1, HORIZON_WEEKS + 1):
        t0 = time.time()
        X, y, _, _ = build_rows(ctx, train_origins(fold.test_start, h, n_origins), h)
        ds = lgb.Dataset(X, y, categorical_feature=CAT_FEATURES, free_raw_data=True)
        model = lgb.train(params, ds, num_boost_round=num_rounds)
        Xp, _, sid_p, _ = build_rows(ctx, [fold.test_start], h, with_target=False)
        p = np.clip(model.predict(Xp), 0, None)
        if sids is None:
            sids = sid_p
        elif not np.array_equal(sids, sid_p):
            raise ValueError("prediction rows differ across horizons")
        preds.append(p)
        importances.append(
            pd.DataFrame({"feature": X.columns, "gain": model.feature_importance("gain"), "h": h, "fold": fold.fold})
        )
        print(f"  fold {fold.fold} h={h}: {len(y):,} training rows, {time.time() - t0:.0f}s")
        del X, y, ds, model
    return sids, np.column_stack(preds), pd.concat(importances, ignore_index=True)


def main():
    data = load_weekly()
    cal_weekly = pd.read_parquet(PROCESSED_DIR / "calendar_weekly.parquet")
    ctx = FeatureContext(data, cal_weekly)
    folds = make_folds(data.n_weeks)
    base_err = pd.read_parquet(PROCESSED_DIR / "baseline_errors.parquet")

    rows, errors, pred_frames, imps, boots = [], [], [], [], []
    for fold in folds:
        sids, yhat, imp = fit_predict_fold(ctx, fold)
        train = data.sales[:, : fold.train_end]
        y = data.sales[sids, fold.test_start : fold.test_end].astype(np.float64)
        scale = rmsse_scale(train, data.first_sale)[sids]
        recent = slice(fold.train_end - WEIGHT_WEEKS, fold.train_end)
        weights = np.sum(train[sids, recent] * np.nan_to_num(data.price[sids, recent]), axis=1)
        r = rmsse(y, yhat, scale)
        rows.append(
            {
                "fold": fold.fold,
                "use_for_tuning": fold.use_for_tuning,
                "method": "lgbm",
                "wape": wape(y, yhat),
                "rmsse_mean": float(np.nanmean(r)),
                "rmsse_weighted": weighted_mean(r, weights),
                "n_series": int(len(sids)),
                "n_rmsse_undefined": int(np.isnan(r).sum()),
            }
        )
        abs_err = np.abs(y - yhat).sum(axis=1)
        errors.append(
            pd.DataFrame({"sid": sids, "fold": fold.fold, "abs_err": abs_err, "actual": y.sum(axis=1), "rmsse": r})
        )
        pred_frames.append(
            pd.DataFrame(
                {
                    "sid": np.repeat(sids, HORIZON_WEEKS),
                    "fold": fold.fold,
                    "h": np.tile(np.arange(1, HORIZON_WEEKS + 1), len(sids)),
                    "week_idx": np.tile(np.arange(fold.test_start, fold.test_end), len(sids)),
                    "yhat": yhat.ravel().astype(np.float32),
                }
            )
        )
        if not fold.use_for_tuning:
            imps.append(imp)

        b = base_err[(base_err["fold"] == fold.fold) & (base_err["method"] == BASELINE_FOR_COMPARISON)]
        b = b.set_index("sid").loc[sids]
        boot = paired_bootstrap_wape_diff(abs_err, b["abs_err"].to_numpy(), y.sum(axis=1), n_boot=N_BOOT)
        boot.update({"fold": fold.fold, "use_for_tuning": fold.use_for_tuning, "baseline": BASELINE_FOR_COMPARISON})
        boots.append(boot)
        print(f"fold {fold.fold}: lgbm WAPE {rows[-1]['wape']:.4f}, diff vs baseline {boot['wape_diff']:+.4f} "
              f"[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS_DIR / "stage3_lgbm.csv", index=False)
    pd.concat(errors, ignore_index=True).to_parquet(PROCESSED_DIR / "lgbm_errors.parquet", index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(PROCESSED_DIR / "lgbm_predictions.parquet", index=False)
    (REPORTS_DIR / "stage3_bootstrap.json").write_text(json.dumps(boots, indent=2))
    imp = pd.concat(imps).groupby("feature")["gain"].sum().sort_values(ascending=False)
    imp = imp / max(imp.sum(), 1e-12)
    imp.round(4).rename("gain_share").to_csv(REPORTS_DIR / "stage3_importance.csv")

    pd.set_option("display.width", 160)
    print("\nLightGBM per fold:")
    print(res.round(4).to_string(index=False))
    print("\nPaired bootstrap, WAPE(lgbm) - WAPE(" + BASELINE_FOR_COMPARISON + "):")
    print(pd.DataFrame(boots)[["fold", "use_for_tuning", "wape_diff", "ci_low", "ci_high", "share_a_better"]].round(4).to_string(index=False))
    print("\nTop 10 features by gain share (holdout models):")
    print(imp.round(4).head(10).to_string())


if __name__ == "__main__":
    main()
