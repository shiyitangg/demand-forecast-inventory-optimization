"""Baseline forecasters, vectorised over series.

Every function takes the training matrix (series x weeks before the origin),
each series' first-sale week, and the horizon, and returns (series x horizon)
forecasts. Weeks before a series' first sale are ignored (not yet launched).
"""
import numpy as np


def naive(train, first_sale, horizon):
    return np.repeat(train[:, -1:].astype(np.float64), horizon, axis=1)


def seasonal_naive(train, first_sale, horizon, season=52):
    t_end = train.shape[1]
    if t_end < season:
        raise ValueError("not enough history for seasonal naive")
    cols = np.arange(t_end - season, t_end - season + horizon)
    # with horizon > season the pattern would need to repeat; not needed here
    return train[:, cols].astype(np.float64)


def moving_average(train, first_sale, horizon, window):
    t_end = train.shape[1]
    start = np.maximum(t_end - window, first_sale)
    cols = np.arange(t_end)[None, :]
    mask = (cols >= start[:, None]) & (cols < t_end)
    n = mask.sum(axis=1)
    total = np.where(mask, train, 0).sum(axis=1)
    mean = np.divide(total, n, out=np.zeros(len(n)), where=n > 0)
    return np.repeat(mean[:, None], horizon, axis=1)


def croston(train, first_sale, horizon, alpha=0.1):
    """Classic Croston: smooth non-zero sizes (z) and inter-demand intervals (p)."""
    n_series, t_end = train.shape
    z = np.zeros(n_series)
    p = np.ones(n_series)
    q = np.zeros(n_series)  # weeks since last demand
    started = np.zeros(n_series, dtype=bool)
    for t in range(t_end):
        y = train[:, t].astype(np.float64)
        init = (~started) & (t == first_sale)
        z[init] = y[init]
        started |= init
        upd = started & ~init
        q[upd] += 1
        hit = upd & (y > 0)
        z[hit] += alpha * (y[hit] - z[hit])
        p[hit] += alpha * (q[hit] - p[hit])
        q[hit] = 0
    fc = np.where(started, z / p, 0.0)
    return np.repeat(fc[:, None], horizon, axis=1)


def tsb(train, first_sale, horizon, alpha=0.1, beta=0.1):
    """Teunter-Syntetos-Babai: smooth demand probability every week, size on demand."""
    n_series, t_end = train.shape
    z = np.zeros(n_series)
    d = np.zeros(n_series)
    started = np.zeros(n_series, dtype=bool)
    for t in range(t_end):
        y = train[:, t].astype(np.float64)
        init = (~started) & (t == first_sale)
        z[init] = y[init]
        d[init] = 1.0
        started |= init
        upd = started & ~init
        hit = upd & (y > 0)
        d[upd] += beta * (hit[upd].astype(float) - d[upd])
        z[hit] += alpha * (y[hit] - z[hit])
    fc = np.where(started, d * z, 0.0)
    return np.repeat(fc[:, None], horizon, axis=1)
