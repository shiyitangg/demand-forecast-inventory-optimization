"""Stage 6: allocate a limited stock budget across items with a linear program.

Run from the project root:  python -m src.run_allocation

At each weekly origin in the test period, every store gets a budget equal to
a fraction of the value of stocking all its items at their q0.9 forecast of
demand over k = L + R weeks. Three ways to spend it are compared on the
demand that actually followed (single-period view, no carry-over):
  lp          LP maximising expected units sold (src/allocation.py)
  prop_q90    scale every item's q0.9 by the same fraction
  prop_mean   scale every item's mean forecast to use the same budget
Stock is rounded down to whole units so no method exceeds its budget.

Outputs
  reports/stage6_allocation.csv   metrics per origin, budget fraction and method
  reports/stage6_summary.json     averages and paired bootstrap (lp vs prop_q90)
"""
import json
import time
import warnings

import numpy as np
import pandas as pd

from src.allocation import expected_sold, segments, solve_lp
from src.bootstrap import paired_bootstrap_total_ratio
from src.config import (
    BUDGET_FRACTIONS,
    MAIN_LEAD_TIME,
    N_BOOT,
    PROCESSED_DIR,
    QUANTILES,
    REFERENCE_TAU,
    REPORTS_DIR,
    REVIEW_WEEKS,
)
from src.data import load_weekly
from src.folds import make_folds
from src.run_inventory import pivot


def main():
    data = load_weekly()
    folds = make_folds(data.n_weeks)
    start = folds[0].test_start
    holdout_start = [f for f in folds if not f.use_for_tuning][0].test_start
    k = MAIN_LEAD_TIME + REVIEW_WEEKS
    fc = pd.read_parquet(PROCESSED_DIR / "quantile_forecasts.parquet")
    fc = fc[fc["k"] == k]
    origins = np.arange(start, data.n_weeks - k + 1)

    sids = np.flatnonzero(data.first_sale < start)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        price = np.nanmean(data.price[sids, :start], axis=1)
    price = np.nan_to_num(price, nan=float(np.nanmean(price)))
    store = data.ids["store_id"].astype(str).to_numpy()[sids]
    stores = np.unique(store)

    q_all = {tau: pivot(fc, f"q{tau}", sids, origins) for tau in QUANTILES}
    mean_all = pivot(fc, "mean", sids, origins)
    actual_all = pivot(fc, "actual", sids, origins)

    rows = []
    sold_by_series = {(f, m): np.zeros(len(sids)) for f in BUDGET_FRACTIONS for m in ["lp", "prop_q90", "prop_mean"]}
    for j, t in enumerate(origins):
        q = np.column_stack([q_all[tau][:, j] for tau in QUANTILES])
        ref = q_all[REFERENCE_TAU][:, j]
        mean, actual = mean_all[:, j], actual_all[:, j]
        lengths, slopes = segments(q, QUANTILES)
        ref_value = {s: float(np.sum((price * ref)[store == s])) for s in stores}
        mean_value = {s: float(np.sum((price * mean)[store == s])) for s in stores}
        for frac in BUDGET_FRACTIONS:
            budgets = {s: frac * ref_value[s] for s in stores}
            t0 = time.time()
            x_lp = solve_lp(lengths, slopes, price, store, budgets)
            lp_seconds = time.time() - t0
            scale_mean = np.array([budgets[s] / mean_value[s] if mean_value[s] > 0 else 0.0 for s in store])
            methods = {
                "lp": np.floor(x_lp + 1e-9),
                "prop_q90": np.floor(frac * ref),
                "prop_mean": np.floor(scale_mean * mean),
            }
            for name, x in methods.items():
                sold = np.minimum(actual, x)
                sold_by_series[(frac, name)] += sold
                rows.append(
                    {
                        "origin": int(t),
                        "period": "holdout" if t >= holdout_start else "pre_holdout",
                        "budget_fraction": frac,
                        "method": name,
                        "units_sold": float(sold.sum()),
                        "fill_rate": float(sold.sum() / actual.sum()),
                        "revenue": float(np.sum(price * sold)),
                        "budget_used": float(np.sum(price * x) / sum(budgets.values())),
                        "expected_units_model": float(expected_sold(x, q, QUANTILES).sum()),
                        "lp_seconds": round(lp_seconds, 2) if name == "lp" else np.nan,
                    }
                )
        print(f"origin {t}: done")

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS_DIR / "stage6_allocation.csv", index=False)
    summary = {"k": k, "n_origins": int(len(origins)), "n_series": int(len(sids)), "bootstrap_lp_vs_prop_q90_units": {}}
    for frac in BUDGET_FRACTIONS:
        # positive rel_diff = LP sells more units than proportional scaling
        b = paired_bootstrap_total_ratio(sold_by_series[(frac, "lp")], sold_by_series[(frac, "prop_q90")], N_BOOT)
        summary["bootstrap_lp_vs_prop_q90_units"][str(frac)] = b
    agg = res.groupby(["period", "budget_fraction", "method"])[
        ["fill_rate", "revenue", "budget_used", "expected_units_model"]
    ].mean().round(4)
    summary["means"] = {f"{p}|{f}|{m}": v for (p, f, m), v in agg.to_dict(orient="index").items()}
    (REPORTS_DIR / "stage6_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 160)
    print("\nMean over origins (fill_rate = units sold / actual demand):")
    print(agg.to_string())
    print("\nPaired bootstrap over series, all origins: units sold, LP vs prop_q90")
    for frac, b in summary["bootstrap_lp_vs_prop_q90_units"].items():
        print(f"  budget {frac}: {b['rel_diff']:+.2%} [{b['ci_low']:+.2%}, {b['ci_high']:+.2%}]")
    print(f"\nLP solve time per problem: median {res['lp_seconds'].median():.1f}s")


if __name__ == "__main__":
    main()
