"""Features for direct h-step-ahead models.

For a forecast origin t (the first unseen week) and horizon h, the target is
sales in week t+h-1. History features use only weeks < t. Price and calendar
features of the target week are allowed because M5 publishes planned prices
and the event calendar ahead of time.
"""
import numpy as np
import pandas as pd

ROLL_WINDOWS = [4, 8, 13, 26, 52]
CAT_FEATURES = ["store_id", "dept_id", "cat_id", "state_id"]


def _cumsum0(x):
    out = np.zeros((x.shape[0], x.shape[1] + 1), dtype=np.float64)
    np.cumsum(x, axis=1, out=out[:, 1:])
    return out


class FeatureContext:
    """Precomputed cumulative sums so each origin's features are O(n_series)."""

    def __init__(self, data, cal_weekly):
        y = data.sales.astype(np.float64)
        n, T = y.shape
        cols = np.arange(T)[None, :]
        launched = cols >= data.first_sale[:, None]
        on_sale = ~np.isnan(data.price)

        self.data = data
        self.y = y
        self.T = T
        self.c_y = _cumsum0(y)
        self.c_y2 = _cumsum0(y**2)
        self.c_valid = _cumsum0(launched)
        self.c_zero = _cumsum0(launched & (y == 0))
        self.c_price = _cumsum0(np.nan_to_num(data.price))
        self.c_onsale = _cumsum0(on_sale)
        # index of the most recent week with a sale, at or before each week (-1 if none)
        idx = np.where(y > 0, cols, -1)
        self.last_sale = np.maximum.accumulate(idx, axis=1)

        self.codes = {c: pd.Categorical(data.ids[c]).codes.astype(np.int16) for c in CAT_FEATURES}

        cal = cal_weekly.set_index("wm_yr_wk").loc[data.weeks].reset_index()
        if len(cal) != T:
            raise ValueError("calendar does not cover every week")
        state = data.ids["state_id"].astype(str).to_numpy()
        snap = np.zeros((n, T), dtype=np.float32)
        for s in np.unique(state):
            snap[state == s] = cal[f"snap_{s}"].to_numpy()
        self.snap = snap
        week_start = pd.to_datetime(cal["week_start"])
        self.cal = {
            "n_events": cal["n_events"].to_numpy(),
            "event_sporting": cal["event_sporting"].to_numpy(),
            "event_cultural": cal["event_cultural"].to_numpy(),
            "event_national": cal["event_national"].to_numpy(),
            "event_religious": cal["event_religious"].to_numpy(),
            "week_of_year": week_start.dt.isocalendar().week.to_numpy().astype(np.int16),
            "month": week_start.dt.month.to_numpy(),
        }

    def _window(self, c, t, w):
        lo = max(t - w, 0)
        return c[:, t] - c[:, lo]

    def features(self, t, h):
        """Feature columns (dict of arrays) for origin t and horizon h."""
        if t < 52 or t > self.T:
            raise ValueError("origin needs 52 weeks of history")
        target_week = t + h - 1
        y = self.y
        f = {}
        for k in range(1, 5):
            f[f"lag_{k}"] = y[:, t - k]
        f["lag_seasonal"] = y[:, target_week - 52]
        for w in ROLL_WINDOWS:
            n = self._window(self.c_valid, t, w)
            with np.errstate(invalid="ignore", divide="ignore"):
                f[f"mean_{w}"] = np.where(n > 0, self._window(self.c_y, t, w) / n, np.nan)
        n13 = self._window(self.c_valid, t, 13)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = self._window(self.c_y, t, 13) / n13
            m2 = self._window(self.c_y2, t, 13) / n13
            f["std_13"] = np.where(n13 > 1, np.sqrt(np.maximum(m2 - m**2, 0)), np.nan)
            f["zero_share_13"] = np.where(n13 > 0, self._window(self.c_zero, t, 13) / n13, np.nan)
        f["weeks_since_launch"] = t - self.data.first_sale
        last = self.last_sale[:, t - 1]
        f["weeks_since_last_sale"] = np.where(last >= 0, t - 1 - last, np.nan)

        price = self.data.price
        f["price_target"] = price[:, target_week] if target_week < self.T else np.full(len(y), np.nan)
        f["price_last"] = price[:, t - 1]
        n_on = self._window(self.c_onsale, t, 13)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_price = np.where(n_on > 0, self._window(self.c_price, t, 13) / n_on, np.nan)
            f["price_ratio_13"] = f["price_target"] / mean_price
            f["price_change"] = f["price_target"] / f["price_last"] - 1

        if target_week < self.T:
            f["snap_days"] = self.snap[:, target_week]
            for k, v in self.cal.items():
                f[k] = np.full(len(y), v[target_week])
        for c, codes in self.codes.items():
            f[c] = codes
        return f

    def target(self, t, h):
        return self.y[:, t + h - 1]


def build_rows(ctx, origins, h, with_target=True):
    """Stack features over origins, keeping series launched before each origin."""
    frames, targets, sids, orig = [], [], [], []
    for t in origins:
        keep = ctx.data.first_sale < t
        f = ctx.features(t, h)
        frames.append(pd.DataFrame({k: np.asarray(v)[keep] for k, v in f.items()}))
        if with_target:
            targets.append(ctx.target(t, h)[keep])
        sids.append(np.flatnonzero(keep))
        orig.append(np.full(keep.sum(), t))
    X = pd.concat(frames, ignore_index=True)
    for c in X.columns:
        if c not in CAT_FEATURES:
            X[c] = X[c].astype(np.float32)
    y = np.concatenate(targets) if with_target else None
    return X, y, np.concatenate(sids), np.concatenate(orig)
