from __future__ import annotations

import os

import pytest

from src.model.readers.dataset_reader import load_dataset_instance

DATASET = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@pytest.mark.parametrize("R,expected", [(30, 30), (100, 100), (200, 200), (300, 300)])
def test_request_counts(R, expected):
    inst = load_dataset_instance(DATASET, R)
    assert len(inst.requests) == expected


def test_graph_is_full_602_for_every_size():
    # the routing graph is the full network regardless of request count
    for R in (30, 100, 200, 300):
        inst = load_dataset_instance(DATASET, R)
        assert len(inst.nodes) == 602
        assert len(inst.arcs) == 3632


def test_vehicles_and_depot():
    inst = load_dataset_instance(DATASET, 30)
    assert len(inst.vehicles) == 10
    for v in inst.vehicles:
        assert v.start_node == 0
        assert v.end_node == 1
        assert v.battery_capacity_kWh == 30.0


def test_epsilon_weights_present():
    inst = load_dataset_instance(DATASET, 30)
    assert abs(inst.objective.epsilon1 - 1.0 / 3) < 1e-9
    assert abs(inst.objective.epsilon2 - 1.0 / 3) < 1e-9


def test_degradation_absent_two_term_fallback():
    inst = load_dataset_instance(DATASET, 30)
    assert not inst.degradation.is_populated


def test_price_schedule_full_24h():
    inst = load_dataset_instance(DATASET, 30)
    assert len(inst.prices.hourly_multiplier) == 24
    assert all(m > 0 for m in inst.prices.hourly_multiplier)


def test_travel_time_derived_flag():
    inst = load_dataset_instance(DATASET, 30)
    assert all(a.travel_time_is_derived for a in inst.arcs)
    assert all(a.travel_time_min == a.distance_km for a in inst.arcs)
