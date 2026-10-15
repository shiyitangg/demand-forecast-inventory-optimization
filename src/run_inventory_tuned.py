"""Stage 5b: choose each policy's service level by simulated cost, then test it.

Run from the project root:  python -m src.run_inventory_tuned

Stage 5 set the service level of policies B and C to the single-period
critical ratio b / (b + h). Here each policy gets one tunable knob, chosen to
minimise simulated cost on the weeks before the held-out fold:
  A_ma4_fixed_ss  safety stock in weeks of MA4 demand (SAFETY_WEEK_GRID)
  B_normal_ss     service level tau for z(tau) (QUANTILES)
  C_quantile      which quantile forecast to order up to (QUANTILES)
The chosen setting is then scored on the held-out fold's weeks only. The
simulation runs continuously over all weeks; only the scoring is split.

Outputs
  reports/stage5b_tuned.csv     tuning-week cost of every candidate, every scenario
  reports/stage5b_holdout.csv   holdout-week metrics of the chosen settings
  reports/stage5b_main.json     main scenario and paired bootstrap on holdout weeks
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
from src.run_inventory import pivot

SAFETY_WEEK_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def candidates(inp, k):
    """Yield (policy, setting, order-up-to matrix) for every candidate."""
    for w in SAFETY_WEEK_GRID:
        yield "A_ma4_fixed_ss", w, inp["ma4"] * (k + w)
    for tau in QUANTILES:
        yield "B_normal_ss", tau, inp["mean"] + norm.ppf(tau) * inp["sigma"]
    for tau in QUANTILES:
        yield "C_quantile", tau, inp["q"][tau]


def main():
    if REVIEW_WEEKS != 1:
        raise ValueError("the simulation assumes weekly review")
    data = load_weekly()
    folds = make_folds(data.n_weeks)
    start = folds[0].test_start
    holdout_start = [f for f in folds if not f.use_for_tuning][0].test_start
    fc_all = pd.read_parquet(PROCESSED_DIR / "quantile_forecasts.parquet")

    sids = np.flatnonzero(data.first_sale < start)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        price = np.nanmean(data.price[sids, :start], axis=1)
    price = np.nan_to_num(price, nan=float(np.nanmean(price)))
    holding = HOLDING_RATE_WEEKLY * price

    tune_rows, hold_rows, main_out = [], [], {}
    for L in LEAD_TIMES:
        k = L + REVIEW_WEEKS
        fc = fc_all[fc_all["k"] == k]
        origins = np.arange(start, data.n_weeks - k + 1)
        weeks = np.arange(start, origins[-1] + L + 1)
        demand = data.sales[sids][:, weeks]
        inp = {
            "mean": pivot(fc, "mean", sids, origins),
            "sigma": pivot(fc, "sigma", sids, origins),
            "ma4": np.column_stack(
                [moving_average(data.sales[sids, :t], data.first_sale[sids], 1, 4)[:, 0] for t in origins]
            ),
            "q": {tau: pivot(fc, f"q{tau}", sids, origins) for tau in QUANTILES},
        }
        counted = np.arange(L, len(weeks))
        tune_cols = counted[weeks[counted] < holdout_start]
        hold_cols = counted[weeks[counted] >= holdout_start]

        sims = {}
        for policy, setting, S in candidates(inp, k):
            sims[(policy, setting)] = simulate(np.clip(S, 0, None), demand, L)

        for ratio in COST_RATIOS:
            lost_cost = ratio * holding
            chosen = {}
            for (policy, setting), sim in sims.items():
                m, _ = summarise(sim, demand, holding, lost_cost, tune_cols)
                tune_rows.append({"lead_time": L, "cost_ratio": ratio, "policy": policy, "setting": setting, **m})
                if policy not in chosen or m["total_cost"] < chosen[policy][1]:
                    chosen[policy] = (setting, m["total_cost"])
            per_series = {}
            for policy, (setting, _) in chosen.items():
                m, ps = summarise(sims[(policy, setting)], demand, holding, lost_cost, hold_cols)
                hold_rows.append(
                    {"lead_time": L, "cost_ratio": ratio, "policy": policy, "chosen_setting": setting,
                     "critical_ratio": round(ratio / (ratio + 1), 3), **m}
                )
                per_series[policy] = ps
            if L == MAIN_LEAD_TIME and ratio == MAIN_COST_RATIO:
                main_out["bootstrap_holdout_C_vs_B"] = paired_bootstrap_total_ratio(
                    per_series["C_quantile"], per_series["B_normal_ss"], N_BOOT)
                main_out["bootstrap_holdout_C_vs_A"] = paired_bootstrap_total_ratio(
                    per_series["C_quantile"], per_series["A_ma4_fixed_ss"], N_BOOT)
        print(f"lead time {L}: {len(tune_cols)} tuning weeks, {len(hold_cols)} holdout weeks")

    tune = pd.DataFrame(tune_rows)
    hold = pd.DataFrame(hold_rows)
    tune.to_csv(REPORTS_DIR / "stage5b_tuned.csv", index=False)
    hold.to_csv(REPORTS_DIR / "stage5b_holdout.csv", index=False)
    main = hold[(hold["lead_time"] == MAIN_LEAD_TIME) & (hold["cost_ratio"] == MAIN_COST_RATIO)]
    main_out["metrics_holdout_weeks"] = main.round(4).to_dict(orient="records")
    (REPORTS_DIR / "stage5b_main.json").write_text(json.dumps(main_out, indent=2))

    pd.set_option("display.width", 160)
    cols = ["policy", "chosen_setting", "critical_ratio", "total_cost", "holding_cost", "lost_sales_cost",
            "fill_rate", "stockout_week_share", "days_of_inventory"]
    print(f"\nMain scenario (L={MAIN_LEAD_TIME}, cost ratio {MAIN_COST_RATIO}), holdout weeks, tuned settings:")
    print(main[cols].round(3).to_string(index=False))
    for key in ["bootstrap_holdout_C_vs_B", "bootstrap_holdout_C_vs_A"]:
        b = main_out[key]
        print(f"{key}: cost {b['rel_diff']:+.2%} [{b['ci_low']:+.2%}, {b['ci_high']:+.2%}]")

    chosen = hold.pivot_table(index=["lead_time", "cost_ratio"], columns="policy", values="chosen_setting")
    print("\nChosen setting per scenario (A: safety weeks; B, C: service level):")
    print(chosen.to_string())
    rel = hold.pivot_table(index=["lead_time", "cost_ratio"], columns="policy", values="total_cost")
    print("\nHoldout-week cost relative to C_quantile (tuned):")
    print(rel.div(rel["C_quantile"], axis=0).round(3).to_string())


if __name__ == "__main__":
    main()
