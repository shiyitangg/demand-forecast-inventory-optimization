"""Rolling-origin backtest folds on a weekly index (week_idx = 0..n_weeks-1)."""
from dataclasses import asdict, dataclass

from src.config import FOLD_STEP_WEEKS, HORIZON_WEEKS, N_FOLDS


@dataclass(frozen=True)
class Fold:
    fold: int
    train_end: int  # exclusive: training uses week_idx < train_end
    test_start: int  # inclusive
    test_end: int  # exclusive
    use_for_tuning: bool

    def to_dict(self):
        return asdict(self)


def make_folds(n_weeks, horizon=HORIZON_WEEKS, n_folds=N_FOLDS, step=FOLD_STEP_WEEKS):
    """Return folds ordered oldest to newest; the last test window ends at n_weeks."""
    if step < horizon:
        raise ValueError("step < horizon would make test windows overlap")
    first_start = n_weeks - horizon - (n_folds - 1) * step
    if first_start <= 0:
        raise ValueError("not enough weeks for the requested folds")
    folds = []
    for k in range(n_folds):
        start = first_start + k * step
        folds.append(
            Fold(
                fold=k + 1,
                train_end=start,
                test_start=start,
                test_end=start + horizon,
                use_for_tuning=k < n_folds - 1,
            )
        )
    return folds


def split_masks(week_idx, fold):
    """Boolean train/test masks for an array-like of week_idx values."""
    train = week_idx < fold.train_end
    test = (week_idx >= fold.test_start) & (week_idx < fold.test_end)
    return train, test
