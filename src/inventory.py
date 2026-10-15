"""Periodic-review, order-up-to inventory simulation with lost sales.

Timeline of each week i (vectorised over series):
  1. orders placed at review i - L arrive
  2. review (if i is a review week): inventory position = on hand + in transit;
     order max(0, S_i - position), arriving at the start of week i + L
  3. demand is served from on hand; any shortfall is lost
  4. holding cost is charged on the ending on-hand stock
An order placed at week i is the last one that can serve weeks i .. i + L, so
S_i should cover demand over k = L + 1 weeks (review period R = 1).
"""
import numpy as np


def simulate(order_up_to, demand, lead_time, init_on_hand=None):
    """Return per-week arrays (series x weeks): on_hand (end of week), sales, lost, orders.

    order_up_to: (n_series, n_reviews) target levels, rounded up to whole units
    demand:      (n_series, n_reviews + lead_time) demand in the simulated weeks
    """
    S = np.ceil(np.asarray(order_up_to, dtype=np.float64))
    d = np.asarray(demand, dtype=np.float64)
    n, n_reviews = S.shape
    n_weeks = d.shape[1]
    if n_weeks != n_reviews + lead_time:
        raise ValueError("demand must cover the review weeks plus the lead time")
    if lead_time < 1:
        raise ValueError("lead_time must be at least 1 week")

    on_hand = S[:, 0].copy() if init_on_hand is None else np.asarray(init_on_hand, dtype=np.float64).copy()
    arrivals = np.zeros((n, n_weeks + lead_time))
    out = {key: np.zeros((n, n_weeks)) for key in ["on_hand", "sales", "lost", "orders"]}
    for i in range(n_weeks):
        on_hand += arrivals[:, i]
        if i < n_reviews:
            in_transit = arrivals[:, i + 1 : i + lead_time].sum(axis=1)
            order = np.maximum(S[:, i] - on_hand - in_transit, 0.0)
            arrivals[:, i + lead_time] += order
            out["orders"][:, i] = order
        sales = np.minimum(on_hand, d[:, i])
        on_hand -= sales
        out["sales"][:, i] = sales
        out["lost"][:, i] = d[:, i] - sales
        out["on_hand"][:, i] = on_hand
    return out


def summarise(sim, demand, holding_cost, lost_cost, weeks):
    """Aggregate metrics over the selected week columns."""
    oh = sim["on_hand"][:, weeks]
    lost = sim["lost"][:, weeks]
    d = np.asarray(demand, dtype=np.float64)[:, weeks]
    hold = holding_cost[:, None] * oh
    short = lost_cost[:, None] * lost
    per_series = hold.sum(axis=1) + short.sum(axis=1)
    return {
        "total_cost": float(per_series.sum()),
        "holding_cost": float(hold.sum()),
        "lost_sales_cost": float(short.sum()),
        "fill_rate": float(1 - lost.sum() / d.sum()),
        "stockout_week_share": float(np.mean(lost > 0)),
        "days_of_inventory": float(7 * oh.sum() / d.sum()),
    }, per_series
