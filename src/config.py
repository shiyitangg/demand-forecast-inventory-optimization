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
