"""Point-forecast error metrics on (series x horizon) arrays."""
import numpy as np


def wape(y, yhat):
    """Sum of absolute errors divided by sum of actuals, over all cells."""
    denom = np.sum(y)
    return float(np.sum(np.abs(y - yhat)) / denom) if denom > 0 else np.nan


def rmsse_scale(train, first_sale):
    """Mean squared one-step naive error per series, using weeks from first sale on.

    Returns NaN where the in-sample naive error is zero (RMSSE undefined).
    """
    diffs = np.diff(train.astype(np.float64), axis=1)
    # diff at column j compares weeks j and j+1; keep only pairs fully after launch
    valid = np.arange(diffs.shape[1])[None, :] >= first_sale[:, None]
    sq = np.where(valid, diffs**2, 0.0)
    n = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = sq.sum(axis=1) / n
    scale[(n == 0) | (scale == 0)] = np.nan
    return scale


def rmsse(y, yhat, scale):
    """Per-series RMSSE; NaN where scale is NaN."""
    mse = np.mean((y - yhat) ** 2, axis=1)
    return np.sqrt(mse / scale)


def weighted_mean(values, weights):
    ok = ~np.isnan(values) & (weights > 0)
    if not ok.any():
        return np.nan
    return float(np.sum(values[ok] * weights[ok]) / np.sum(weights[ok]))
