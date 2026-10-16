# Design notes

Why each part of the pipeline is built the way it is. Settings live in `src/config.py`.

## Data and time grain

- **Weekly, not daily.** Walmart weeks (Saturday to Friday, `wm_yr_wk`) match a weekly
  review cycle and cut the table from ~59M daily rows to 8.4M weekly rows, which fits
  an 8 GB laptop. The daily file is read in chunks of 5,000 series and summed with
  `np.add.reduceat`; the result was checked against a pandas melt + groupby.
- **Partial week dropped.** The data ends two days into week 11617, so that week is
  removed rather than treated as low demand.
- **Launch handling.** Weeks before an item's first sale are kept in the table but
  flagged. They line up with weeks without a price (20.8% each, and no week has sales
  without a price), so they are treated as "not yet on sale". Baselines ignore them,
  features count history only from launch, and a series is forecast at an origin only
  if it sold before that origin.

## Backtest

- 4 rolling origins, each followed by a 4-week test window; windows do not overlap and
  the last ends at the last full week. Training for a fold uses only weeks before its
  test window.
- Folds 1-3 are used for every choice; fold 4 is held out and reported once. Choices
  made on folds 1-3: moving-average window, the baseline LightGBM is compared to, the
  service level of each inventory policy.
- Demand classes (smooth / erratic / intermittent / lumpy, Syntetos-Boylan cut-offs
  ADI 1.32, CV^2 0.49) use only weeks before fold 1.

## Metrics

- **WAPE** = sum |error| / sum actual over all series and weeks in a window. It is
  scale-free in aggregate and weights high-volume series naturally.
- **RMSSE** per series, scaled by the in-sample one-step naive error from launch; it is
  undefined (and excluded, with a count) for series with a constant history. Reported
  as a plain mean and weighted by the last 4 weeks of revenue, as in M5.
- **Paired bootstrap**: resample series with replacement 1,000 times, recompute the
  metric difference on the same resampled series for both methods.

## LightGBM

- **Direct models, one per horizon step** (h = 1..4) instead of a recursive model, so
  errors do not feed back into lags. For horizon h and origin t the target is week
  t+h-1; training origins are the latest 52 whose target falls before the test window.
- **Features**: lags 1-4; the same week 52 weeks earlier; rolling means over 4, 8, 13,
  26, 52 weeks; 13-week standard deviation and zero share; weeks since launch and since
  last sale; price of the target week, its ratio to the 13-week mean price and change
  from last week (M5 has no promotion flag, so a price drop is the proxy); SNAP days,
  events, week of year and month of the target week; store, department, category, state.
  Target-week price and calendar are allowed because M5 publishes them in advance.
- **Leakage test**: features are recomputed after overwriting all sales from the
  origin onward with 999; every feature must be unchanged.
- Tweedie objective (power 1.1) for non-negative, zero-heavy counts. Hyperparameters
  are fixed and were not tuned.

## Quantile forecasts

- **Target is cumulative demand over k = L + R weeks** (review R = 1, lead time L = 1-3),
  because an order-up-to level must cover exactly that window. Predicting the sum
  directly avoids having to model correlation between weekly errors.
- One LightGBM quantile model per tau in {0.5, 0.7, 0.8, 0.9, 0.95, 0.98}; crossing
  quantiles are fixed by sorting each row.
- **Comparison**: LightGBM tweedie mean + z(tau) * sqrt(k) * 13-week standard deviation,
  the textbook normal safety-stock assumption.
- Forecasts are made at every weekly origin in the test period with the model of the
  fold containing that origin, so the inventory simulation only sees out-of-sample
  forecasts. Scoring for tuning uses origins whose target window ends before fold 4.

## Inventory simulation

- Periodic review every week, order-up-to level S, lead time L, **lost sales** (unmet
  demand is gone, not backordered). Each week: receive orders placed L weeks ago,
  review and order max(0, S - on hand - in transit), serve demand, charge holding on
  ending stock. S is rounded up to whole units.
- Costs are scaled by each item's average price before the simulation: holding 0.48%
  of price per unit-week; lost sale = cost ratio x holding, ratio in {4, 9, 19, 49},
  i.e. critical ratios 0.8, 0.9, 0.95, 0.98.
- Initial stock is each policy's own first S; costs are counted from week L on, after
  the first order lands.
- **Policies**: (A) k x MA4 + fixed safety stock in weeks of MA4; (B) LightGBM mean +
  normal safety stock; (C) LightGBM quantile.
- **Two versions**: stage 5 sets the service level to b/(b+h) and A's safety stock to
  one week. Stage 5b gives each policy one knob (A: safety weeks 0-3; B and C: tau in
  the quantile grid), picks it by cost on pre-holdout weeks, and scores the held-out
  weeks only. The simulation runs continuously; only scoring is split.

## Budget allocation

- Expected units sold from stocking x is E[min(D, x)] = integral of P(D > u) from 0 to x.
  From the quantile forecasts this is a concave piecewise-linear function: segment j
  runs between consecutive quantiles with slope equal to the average survival on it;
  below the median P(D > 0) is assumed to be 1; beyond the 0.98 quantile the slope is 0.
- LP: maximise sum of slope x segment stock subject to one budget per store; solved with
  HiGHS dual simplex. Presolve is turned off because it took ~100 s on this shape
  (one dense row per store, ~180k bounded columns) versus ~1 s without it.
- A greedy fractional-knapsack solver is used in the tests to confirm the LP optimum.
- Compared with two heuristics that use the same budget: scale every item's 0.9
  quantile, or scale every item's mean forecast. Evaluated on realised demand, one
  3-week window at a time, stock rounded down so no method exceeds its budget.
