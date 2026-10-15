import numpy as np
import pytest

from src.bootstrap import paired_bootstrap_total_ratio
from src.inventory import simulate, summarise


def test_constant_demand_with_exact_target_never_stocks_out():
    # demand 5/week, L=2 -> target covers 3 weeks = 15
    demand = np.full((1, 8), 5.0)
    sim = simulate(np.full((1, 6), 15.0), demand, lead_time=2)
    assert sim["lost"].sum() == 0
    # deterministic demand: stock runs down to exactly zero as each order lands
    assert sim["on_hand"][0].tolist() == [10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_order_arrives_after_lead_time():
    demand = np.zeros((1, 4))
    sim = simulate(np.array([[0.0, 7.0]]), demand, lead_time=2, init_on_hand=[0.0])
    assert sim["orders"][0].tolist() == [0.0, 7.0, 0.0, 0.0]
    # the order placed in week 1 is on hand from week 3
    assert sim["on_hand"][0].tolist() == [0.0, 0.0, 0.0, 7.0]


def test_lost_sales_are_not_backordered():
    demand = np.array([[10.0, 0.0, 0.0]])
    sim = simulate(np.array([[4.0, 4.0]]), demand, lead_time=1)
    assert sim["sales"][0, 0] == 4 and sim["lost"][0, 0] == 6
    assert sim["on_hand"][0, 2] == 4  # refilled to target, shortage not carried


def test_in_transit_counts_toward_position():
    demand = np.zeros((1, 6))
    sim = simulate(np.full((1, 3), 9.0), demand, lead_time=3, init_on_hand=[0.0])
    # first review orders 9; later reviews see it in transit and order nothing
    assert sim["orders"][0, :3].tolist() == [9.0, 0.0, 0.0]


def test_targets_rounded_up_and_bad_shapes_rejected():
    sim = simulate(np.array([[0.2]]), np.array([[0.0, 0.0]]), lead_time=1)
    assert sim["on_hand"][0, 0] == 1.0
    with pytest.raises(ValueError):
        simulate(np.ones((1, 3)), np.ones((1, 3)), lead_time=1)


def test_summarise_costs():
    demand = np.array([[10.0, 0.0, 0.0]])
    sim = simulate(np.array([[4.0, 4.0]]), demand, lead_time=1)
    m, per_series = summarise(sim, demand, np.array([1.0]), np.array([9.0]), np.arange(3))
    assert m["lost_sales_cost"] == 54.0
    assert m["holding_cost"] == sim["on_hand"].sum()
    assert per_series[0] == m["total_cost"]
    assert m["fill_rate"] == pytest.approx(0.4)


def test_total_ratio_bootstrap():
    rng = np.random.default_rng(0)
    b = rng.random(500) + 1
    out = paired_bootstrap_total_ratio(0.9 * b, b, n_boot=200)
    assert out["rel_diff"] == pytest.approx(-0.1)
    assert out["ci_low"] == pytest.approx(-0.1) and out["ci_high"] == pytest.approx(-0.1)
