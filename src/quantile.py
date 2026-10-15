"""Cumulative-demand targets, features and probabilistic scoring.

For origin t (first unseen week) and window k, the target is total sales in
weeks t .. t+k-1. History features are the h=1 features (weeks < t); calendar
and price features are summarised over the k target weeks, which are known in
advance in M5.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.features import CAT_FEATURES


def cumulative_target(ctx, t, k):
    return ctx.c_y[:, t + k] - ctx.c_y[:, t]


def cumulative_features(ctx, t, k):
    if t + k > ctx.T:
        raise ValueError("target window runs past the data")
    f = ctx.features(t, 1)
    for name in ["price_target", "price_ratio_13", "price_change", "snap_days"] + list(ctx.cal):
        f.pop(name, None)
    window = slice(t, t + k)
    # items off sale for the whole window get NaN, which LightGBM treats as missing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f["price_window"] = np.nanmean(ctx.data.price[:, window], axis=1) if k > 1 else ctx.data.price[:, t]
    last = f["price_last"]
    with np.errstate(invalid="ignore", divide="ignore"):
        f["price_window_change"] = f["price_window"] / last - 1
    f["snap_window"] = ctx.snap[:, window].sum(axis=1)
    f["events_window"] = np.full(len(last), ctx.cal["n_events"][window].sum())
    f["week_of_year"] = np.full(len(last), ctx.cal["week_of_year"][t])
    return f


def build_cumulative_rows(ctx, origins, k, with_target=True):
    frames, targets, sids, orig = [], [], [], []
    for t in origins:
        keep = ctx.data.first_sale < t
        with np.errstate(invalid="ignore"):
            f = cumulative_features(ctx, t, k)
        frames.append(pd.DataFrame({c: np.asarray(v)[keep] for c, v in f.items()}))
        if with_target:
            targets.append(cumulative_target(ctx, t, k)[keep])
        sids.append(np.flatnonzero(keep))
        orig.append(np.full(keep.sum(), t))
    X = pd.concat(frames, ignore_index=True)
    for c in X.columns:
        if c not in CAT_FEATURES:
            X[c] = X[c].astype(np.float32)
    y = np.concatenate(targets) if with_target else None
    return X, y, np.concatenate(sids), np.concatenate(orig)


def sort_quantiles(q):
    """Fix quantile crossing by sorting each row (rearrangement)."""
    return np.sort(q, axis=1)


def normal_quantiles(mean, sigma, taus):
    z = norm.ppf(np.asarray(taus))
    return np.clip(mean[:, None] + sigma[:, None] * z[None, :], 0, None)


def pinball(y, q, tau):
    diff = y - q
    return np.maximum(tau * diff, (tau - 1) * diff)


def coverage(y, q):
    return float(np.mean(y <= q))
