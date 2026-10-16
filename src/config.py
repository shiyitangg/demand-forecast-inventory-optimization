"""Project-wide paths and backtest settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

# Backtest design (weekly data): 4 rolling origins, each followed by a
# HORIZON_WEEKS test window. Windows do not overlap and the last one ends at
# the last full week. Only the last fold is held out from all tuning.
HORIZON_WEEKS = 4
N_FOLDS = 4
FOLD_STEP_WEEKS = 4

# Syntetos-Boylan demand classification thresholds.
ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49

# Stage 3: LightGBM, one direct model per horizon step. Parameters are fixed
# (not tuned); training uses the most recent N_TRAIN_ORIGINS origins whose
# targets are all observed before the fold's test window.
N_TRAIN_ORIGINS = 52
LGBM_NUM_ROUNDS = 500
LGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "seed": 42,
    "verbose": -1,
}
BASELINE_FOR_COMPARISON = "ma_4"  # best baseline on tuning folds in stage 2
N_BOOT = 1000

# Stage 4: quantile forecasts of cumulative demand over k = L + R weeks.
# Weekly review (R = 1) and assumed lead times L in LEAD_TIMES; L = 2 is the
# main case. M5 has no lead-time data, so these are assumptions.
REVIEW_WEEKS = 1
LEAD_TIMES = [1, 2, 3]
MAIN_LEAD_TIME = 2
QUANTILES = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
N_TRAIN_ORIGINS_Q = 26
Q_NUM_ROUNDS = 300

# Stage 5: inventory simulation (all values are assumptions, not M5 data).
HOLDING_RATE_WEEKLY = 0.25 / 52  # holding cost per unit-week as a share of price
MAIN_COST_RATIO = 9  # lost-sale cost / holding cost; critical ratio 0.9
COST_RATIOS = [4, 9, 19, 49]  # critical ratios 0.8, 0.9, 0.95, 0.98
FIXED_SAFETY_WEEKS = 1.0  # policy A: safety stock = this many weeks of MA4 demand

# Stage 6: budget-constrained allocation. Each store's budget is a fraction of
# what it would cost to stock every item at its q0.9 forecast.
BUDGET_FRACTIONS = [0.5, 0.7, 0.9]
REFERENCE_TAU = 0.9
