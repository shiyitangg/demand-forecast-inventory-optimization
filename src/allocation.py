"""Budget-constrained stock allocation across items as a linear program.

Expected units sold from stocking x units is E[min(D, x)] = integral of
P(D > u) from 0 to x. With quantile forecasts q_tau this is approximated as a
concave piecewise-linear function: between consecutive quantiles the slope is
the average survival probability on that segment, and it is flat beyond the
highest quantile. Below the median we assume P(D > 0) = 1, which overstates
the first segment's slope for intermittent items.

LP:  maximise   sum_ij slope_j * y_ij
     subject to sum_{i in group g} sum_j price_i * y_ij <= budget_g   (per group)
                0 <= y_ij <= segment length_ij
Stock for item i is x_i = sum_j y_ij. Because slopes decrease along j, the LP
fills segments in order without extra constraints.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def segments(q, taus):
    """Segment lengths (n_items x n_taus) and slopes (n_taus,) from sorted quantiles."""
    q = np.maximum(np.sort(np.asarray(q, dtype=np.float64), axis=1), 0)
    taus = np.asarray(taus, dtype=np.float64)
    if not np.all(np.diff(taus) > 0):
        raise ValueError("taus must be increasing")
    edges = np.column_stack([np.zeros(len(q)), q])
    lengths = np.diff(edges, axis=1)
    surv = np.concatenate([[1.0], 1 - taus])
    slopes = (surv[:-1] + surv[1:]) / 2
    return lengths, slopes


def expected_sold(x, q, taus):
    """Model-implied E[min(D, x)] under the piecewise-linear approximation."""
    lengths, slopes = segments(q, taus)
    starts = np.cumsum(lengths, axis=1) - lengths
    filled = np.clip(np.asarray(x, dtype=np.float64)[:, None] - starts, 0, lengths)
    return filled @ slopes


def solve_lp(lengths, slopes, price, group, budgets):
    """Return stock per item from the LP above (HiGHS solver)."""
    n, m = lengths.shape
    groups = np.unique(group)
    g_index = np.searchsorted(groups, group)
    cols = np.arange(n * m)
    rows = np.repeat(g_index, m)
    vals = np.repeat(np.asarray(price, dtype=np.float64), m)
    A = csr_matrix((vals, (rows, cols)), shape=(len(groups), n * m))
    b = np.array([budgets[g] for g in groups], dtype=np.float64)
    c = -np.tile(slopes, n)
    bounds = np.column_stack([np.zeros(n * m), lengths.ravel()])
    # HiGHS presolve is very slow on this shape (one dense row per group plus
    # ~180k bounded columns); dual simplex without presolve solves it in seconds.
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs-ds", options={"presolve": False})
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x.reshape(n, m).sum(axis=1)


def solve_greedy(lengths, slopes, price, group, budgets):
    """Same problem solved by fractional knapsack per group (used to check the LP)."""
    n, m = lengths.shape
    x = np.zeros(n)
    price = np.asarray(price, dtype=np.float64)
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        item = np.repeat(idx, m)
        seg_len = lengths[idx].ravel()
        ratio = np.tile(slopes, len(idx)) / price[item]
        order = np.argsort(-ratio, kind="stable")
        left = budgets[g]
        for o in order:
            if left <= 0:
                break
            take = min(seg_len[o], left / price[item[o]])
            x[item[o]] += take
            left -= take * price[item[o]]
    return x
