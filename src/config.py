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
