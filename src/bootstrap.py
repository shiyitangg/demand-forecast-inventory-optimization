"""Paired bootstrap over series for the difference in WAPE between two methods."""
import numpy as np


def paired_bootstrap_wape_diff(abs_a, abs_b, actual, n_boot=1000, seed=0, batch=50):
    """WAPE(a) - WAPE(b), resampling series with replacement.

    abs_a, abs_b, actual: per-series sums over the test window, aligned by series.
    Negative values mean method a has lower error.
    """
    abs_a, abs_b, actual = (np.asarray(x, dtype=np.float64) for x in (abs_a, abs_b, actual))
    n = len(actual)
    rng = np.random.default_rng(seed)
    diffs = []
    for start in range(0, n_boot, batch):
        k = min(batch, n_boot - start)
        idx = rng.integers(0, n, size=(k, n))
        diffs.append((abs_a[idx].sum(axis=1) - abs_b[idx].sum(axis=1)) / actual[idx].sum(axis=1))
    diffs = np.concatenate(diffs)
    point = (abs_a.sum() - abs_b.sum()) / actual.sum()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "wape_diff": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "share_a_better": float(np.mean(diffs < 0)),
        "n_series": int(n),
        "n_boot": int(n_boot),
    }


def paired_bootstrap_total_ratio(a, b, n_boot=1000, seed=0, batch=50):
    """Relative difference sum(a) / sum(b) - 1, resampling series with replacement.

    Negative values mean method a has the lower total (e.g. lower cost).
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = len(a)
    rng = np.random.default_rng(seed)
    stats = []
    for start in range(0, n_boot, batch):
        k = min(batch, n_boot - start)
        idx = rng.integers(0, n, size=(k, n))
        stats.append(a[idx].sum(axis=1) / b[idx].sum(axis=1) - 1)
    stats = np.concatenate(stats)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return {
        "rel_diff": float(a.sum() / b.sum() - 1),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "share_a_lower": float(np.mean(stats < 0)),
        "n_series": int(n),
        "n_boot": int(n_boot),
    }
