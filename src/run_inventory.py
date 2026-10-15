"""Stage 5: compare three replenishment policies in a lost-sales simulation.

Run from the project root:  python -m src.run_inventory

Policies (order-up-to level S at each weekly review, covering k = L + 1 weeks):
  A_ma4_fixed_ss   k * MA4 + FIXED_SAFETY_WEEKS * MA4 (ignores the cost ratio)
  B_normal_ss      LightGBM mean + z(tau*) * sqrt(k) * std of last 13 weeks
  C_quantile       LightGBM quantile forecast at tau* = b / (b + h)

Everything is simulated on observed weekly sales from the first test week to
the last full week, using forecasts made out of sample (stage 4). Costs are
counted from week L on, after the first order has arrived. Holding and
lost-sale costs are assumptions scaled by each item's average price before
the simulation period.

Outputs
  reports/stage5_inventory.csv   metrics for every policy, cost ratio and lead time
  reports/stage5_main.json       main scenario metrics and paired bootstrap of costs
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.baselines import moving_average
from src.bootstrap import paired_bootstrap_total_ratio
from src.config import (
    COST_RATIOS,
    FIXED_SAFETY_WEEKS,
    HOLDING_RATE_WEEKLY,
    LEAD_TIMES,
    MAIN_COST_RATIO,
    MAIN_LEAD_TIME,
    N_BOOT,
    PROCESSED_DIR,
    QUANTILES,
    REPORTS_DIR,
    REVIEW_WEEKS,
)
from src.data import load_weekly
from src.folds import make_folds
from src.inventory import simulate, summarise


def pivot(fc, col, sids, origins):
    m = fc.pivot(index="sid", columns="origin", values=col).reindex(index=sids, columns=origins)
    if m.isna().any().any():
        raise ValueError(f"missing forecasts for {col}")
    return m.to_numpy(np.float64)


def main():
    if REVIEW_WEEKS != 1:
        raise ValueError("the simulation assumes weekly review")
    data = load_weekly()
    folds = make_folds(data.n_weeks)
    start = folds[0].test_start
    holdout_start = [f for f in folds if not f.use_for_tuning][0].test_start
    fc_all = pd.read_parquet(PROCESSED_DIR / "quantile_forecasts.parquet")

    sids = np.flatnonzero(data.first_sale < start)  # launched before the simulation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        price = np.nanmean(data.price[sids, :start], axis=1)
    price = np.nan_to_num(price, nan=float(np.nanmean(price)))
    holding = HOLDING_RATE_WEEKLY * price

    rows, main_out = [], {}
    for L in LEAD_TIMES:
        k = L + REVIEW_WEEKS
        fc = fc_all[fc_all["k"] == k]
        origins = np.arange(start, data.n_weeks - k + 1)
        weeks = np.arange(start, origins[-1] + L + 1)
        demand = data.sales[sids][:, weeks]
        mean = pivot(fc, "mean", sids, origins)
        sigma = pivot(fc, "sigma", sids, origins)
        ma4 = np.column_stack([moving_average(data.sales[sids, :t], data.first_sale[sids], 1, 4)[:, 0] for t in origins])
        counted = np.arange(L, len(weeks))
        holdout_cols = counted[weeks[counted] >= holdout_start]

        for ratio in COST_RATIOS:
            tau = ratio / (ratio + 1)
            qcol = f"q{round(tau, 2)}"
            if round(tau, 2) not in QUANTILES:
                raise ValueError(f"no quantile forecast for tau={tau}")
            policies = {
                "A_ma4_fixed_ss": ma4 * (k + FIXED_SAFETY_WEEKS),
                "B_normal_ss": mean + norm.ppf(tau) * sigma,
                "C_quantile": pivot(fc, qcol, sids, origins),
            }
            per_series = {}
            for name, S in policies.items():
                sim = simulate(np.clip(S, 0, None), demand, L)
                for period, cols in [("all", counted), ("holdout_weeks", holdout_cols)]:
                    m, ps = summarise(sim, demand, holding, ratio * holding, cols)
                    rows.append({"lead_time": L, "cost_ratio": ratio, "tau": round(tau, 2), "policy": name, "period": period, **m})
                    if period == "all":
                        per_series[name] = ps
            if L == MAIN_LEAD_TIME and ratio == MAIN_COST_RATIO:
                main_out["bootstrap_C_vs_B"] = paired_bootstrap_total_ratio(per_series["C_quantile"], per_series["B_normal_ss"], N_BOOT)
                main_out["bootstrap_C_vs_A"] = paired_bootstrap_total_ratio(per_series["C_quantile"], per_series["A_ma4_fixed_ss"], N_BOOT)
        print(f"lead time {L}: {len(origins)} reviews, {len(counted)} costed weeks, {len(sids)} series")

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS_DIR / "stage5_inventory.csv", index=False)
    main = res[(res["lead_time"] == MAIN_LEAD_TIME) & (res["cost_ratio"] == MAIN_COST_RATIO)]
    main_out["assumptions"] = {
        "lead_time_weeks": MAIN_LEAD_TIME,
        "review_weeks": REVIEW_WEEKS,
        "cost_ratio_lost_to_holding": MAIN_COST_RATIO,
        "holding_rate_weekly_share_of_price": HOLDING_RATE_WEEKLY,
        "fixed_safety_weeks_policy_A": FIXED_SAFETY_WEEKS,
        "lost_sales": True,
    }
    main_out["metrics"] = main.drop(columns=["lead_time", "cost_ratio"]).round(4).to_dict(orient="records")
    (REPORTS_DIR / "stage5_main.json").write_text(json.dumps(main_out, indent=2))

    pd.set_option("display.width", 160)
    cols = ["policy", "total_cost", "holding_cost", "lost_sales_cost", "fill_rate", "stockout_week_share", "days_of_inventory"]
    for period in ["all", "holdout_weeks"]:
        print(f"\nMain scenario (L={MAIN_LEAD_TIME}, cost ratio {MAIN_COST_RATIO}), period={period}:")
        print(main[main["period"] == period][cols].round(3).to_string(index=False))
    for key in ["bootstrap_C_vs_B", "bootstrap_C_vs_A"]:
        b = main_out[key]
        print(f"{key}: cost {b['rel_diff']:+.2%} [{b['ci_low']:+.2%}, {b['ci_high']:+.2%}]")

    sens = res[res["period"] == "all"].pivot_table(index=["lead_time", "cost_ratio"], columns="policy", values="total_cost")
    sens = sens.div(sens["C_quantile"], axis=0).round(3)
    print("\nSensitivity: total cost relative to C_quantile (1.0 = same; >1 = more expensive):")
    print(sens.to_string())
    fill = res[res["period"] == "all"].pivot_table(index=["lead_time", "cost_ratio"], columns="policy", values="fill_rate").round(3)
    print("\nFill rate by scenario:")
    print(fill.to_string())


if __name__ == "__main__":
    main()
