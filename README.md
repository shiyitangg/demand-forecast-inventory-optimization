# Demand Forecasting & Inventory Optimization (M5)

Weekly demand forecasts for 30,490 Walmart item-store series (Kaggle M5), turned into
replenishment and budget-allocation decisions and scored by simulated cost.

The pipeline goes from point forecasts to probabilistic forecasts to decisions:

1. Aggregate daily sales to Walmart weeks; design a 4-origin rolling backtest.
2. Baselines: naive, seasonal naive, moving averages, Croston, TSB.
3. Global LightGBM (one direct model per horizon step), compared with the best
   baseline by a paired bootstrap.
4. Quantile LightGBM for cumulative demand over lead time + review period, checked
   for coverage and pinball loss against a normal approximation.
5. Lost-sales inventory simulation comparing three order-up-to policies, with
   sensitivity to cost ratio and lead time.
6. Linear program that allocates a limited stock budget across items.

Every number below was measured by the scripts in this repository on the M5 data.
Costs, lead times and budgets are **assumptions** (M5 has none of them), so the
conclusions read "under these assumptions, policy X beats policy Y", not real profit.

## Results

Backtest: 4 rolling origins, 4-week test windows, 2016-01-30 to 2016-05-20. Folds 1-3
are used for every choice (MA window, best baseline, service levels); fold 4
(2016-04-23 to 2016-05-20) is held out and only reported.

| Stage | Result (held-out fold unless stated) |
|---|---|
| Data | 30,490 series x 277 full weeks = 8,445,730 rows; no missing or negative values. 20.8% of series-weeks are before an item's first sale; 23.9% of weeks after first sale have zero sales. |
| Baselines | Best on folds 1-3: 4-week moving average (WAPE 0.395). Held out: WAPE 0.379. |
| LightGBM | WAPE 0.3525 vs 0.3794 for MA4: difference -0.027, 95% CI [-0.030, -0.024]. Better than MA4 on all four folds (differences -0.012, -0.024, -0.027, -0.027). |
| Quantiles (3-week demand) | Coverage of the 0.9 / 0.95 / 0.98 quantiles: 0.879 / 0.937 / 0.972 (LightGBM quantile) vs 0.815 / 0.864 / 0.903 (normal). Scaled pinball loss at 0.98: 0.0254 vs 0.0388. |
| Inventory, critical-ratio rule | Lead time 2 weeks, lost-sale cost 9x holding cost, all 16 weeks: quantile policy costs 8.6% less than MA4 + fixed safety stock (CI [-9.1%, -8.0%]) but **4.3% more** than LightGBM mean + normal safety stock (CI [+3.8%, +4.7%]). |
| Inventory, tuned service level | Service level chosen by cost on pre-holdout weeks, scored on the 4 held-out weeks: quantile policy 0.9% cheaper than normal safety stock (CI [-1.4%, -0.3%]) and 9.8% cheaper than MA4 + fixed safety stock (CI [-10.7%, -9.0%]). At cost ratio 49 the normal policy costs 12-15% more. |
| Budget allocation (LP) | With each store's budget at 50% of the value of stocking every item at its 0.9 quantile, the LP sells 26.7% more units than scaling all items down proportionally (CI [+25.5%, +27.7%]); +8.4% at 70% and +1.7% at 90%. On the held-out origins it earns **less revenue** (-9% at the 50% budget, -8% at 70%, -3% at 90%), because it maximises units under a dollar budget and so favours cheap items. |

Confidence intervals come from a paired bootstrap that resamples series (1,000 draws).

## What I would take from this

- A global LightGBM beats the best simple baseline by about 2.7 WAPE points, and the
  gain is consistent across folds; nearly all of its gain-based importance comes from
  recent sales levels, not price or calendar features.
- Quantile models are better calibrated than a normal approximation in the upper
  tail, where inventory decisions live.
- Better calibration did not automatically mean lower cost. Using the single-period
  critical ratio b/(b+h) as the service level, the normal policy was cheaper at
  moderate cost ratios. Once each policy's service level was tuned on cost, the
  quantile policy was cheapest or tied in 11 of 12 lead-time/cost scenarios (the
  normal policy was 0.7% cheaper in the remaining one), and the tuned
  service levels were below the critical ratio (0.8 instead of 0.9 in the main case).
  My reading, not separately proven, is that unsold stock carries over to the next
  week, which the single-period rule ignores.
- The allocation objective matters as much as the optimiser: maximising units and
  maximising revenue pick different items.

## Limitations

- **Assumed costs and lead times.** Holding cost 0.48% of price per week (25% a year),
  lost-sale cost 4x-49x holding, lead times 1-3 weeks, weekly review. None come from M5.
- **Sales are not demand.** M5 records sales; when an item was out of stock, true
  demand is censored. The simulation treats observed sales as demand.
- **Short evaluation window.** All tests fall in February-May 2016 (no holiday season).
  The held-out fold is 4 weeks; for the 3-week quantile target it holds only 2 origins.
- **Tuning.** LightGBM hyperparameters and Croston/TSB smoothing (0.1) were fixed, not
  tuned. Choices that were tuned (MA window, service levels) used folds 1-3 only.
- **Bootstrap independence.** Series are resampled as if independent; series sharing a
  store or item are correlated, so true intervals are probably wider.
- **Coverage is pooled.** Coverage close to nominal across all series does not mean each
  series is well calibrated; erratic items under-cover most (0.852 at the 0.9 quantile).
- **Single-period allocation.** The LP allocates one 3-week window at a time with no
  carry-over, and assumes P(D > 0) = 1 below the median, which is optimistic for
  intermittent items. With only store budget constraints the LP is a fractional
  knapsack; a greedy solver gives the same objective (checked in the tests). The LP is
  there so that further constraints can be added without changing the method.

## How to run

Requires Python 3.12 (tested with 3.12.15 on an 8 GB Apple Silicon Mac) and a Kaggle
account that has accepted the M5 Forecasting - Accuracy competition rules. Data is not
included in this repository.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
kaggle competitions download -c m5-forecasting-accuracy -p data/raw
unzip -o data/raw/m5-forecasting-accuracy.zip -d data/raw
python -m src.prepare_weekly        # stage 1
python -m src.run_baselines         # stage 2
python -m src.run_lgbm              # stage 3, about 5-6 minutes
python -m src.run_quantile          # stage 4, about 15 minutes
python -m src.run_inventory         # stage 5
python -m src.run_inventory_tuned   # stage 5b
python -m src.run_allocation        # stage 6
python -m pytest -q
```

Measured outputs are written to `reports/` (committed); intermediate tables go to
`data/processed/` (not committed).

## Repository layout

```
src/
  config.py              paths, backtest settings and every assumption
  prepare_weekly.py      daily -> weekly aggregation, data profile, folds
  folds.py               rolling-origin folds
  data.py                loads the weekly table as series x week matrices
  baselines.py           naive, seasonal naive, moving average, Croston, TSB
  metrics.py             WAPE, RMSSE
  features.py            leakage-safe features for direct h-step models
  run_lgbm.py            LightGBM per horizon step + paired bootstrap
  quantile.py            cumulative targets, pinball loss, coverage
  run_quantile.py        quantile models vs normal approximation
  inventory.py           lost-sales order-up-to simulator
  run_inventory*.py      policy comparison and service-level tuning
  allocation.py          piecewise-linear expected sales, LP and greedy check
  run_allocation.py      budget allocation experiment
tests/                   unit tests, including leakage checks for features
reports/                 measured results (CSV / JSON)
DESIGN.md                design decisions
reports/decision_memo.md one-page summary for a decision maker
```
