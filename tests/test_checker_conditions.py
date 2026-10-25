from __future__ import annotations

import dataclasses

from src.model.types import Request
from src.verify.checker import VehicleRoute, check_solution
from tests.mini_instance import build_mini_instance

# Conditions of the independent checker not covered by test_verifier.py:
# precedence, overcharging, and validity of the charging amounts themselves.


def _one_route(arcs, served, charge=None):
    return {"veh_1": VehicleRoute(arcs=arcs, charge=charge or {}, served=served)}


def _single_request_instance(pickup, delivery, initial_soc=None):
    inst = build_mini_instance(num_vehicles=1)
    inst = dataclasses.replace(inst, requests=(Request(id=1, pickup_node=pickup, delivery_node=delivery),))
    if initial_soc is not None:
        object.__setattr__(inst.vehicles[0], "initial_soc_kWh", initial_soc)
    return inst


def test_precedence_in_order_passes():
    inst = _single_request_instance(pickup=2, delivery=3)
    rep = check_solution(inst, _one_route([(0, 2), (2, 3), (3, 1)], [1]))
    assert rep.passed, rep.summary()


def test_precedence_reversal_fails():
    # Same connected, elementary path, but the request's delivery (2) comes before its pickup (3).
    inst = _single_request_instance(pickup=3, delivery=2)
    rep = check_solution(inst, _one_route([(0, 2), (2, 3), (3, 1)], [1]))
    assert not rep.passed
    assert any("precedence violated" in v for v in rep.violations)
    # precedence is the ONLY defect: everything else about this route is valid
    assert all("precedence violated" in v for v in rep.violations), rep.summary()


def test_overcharge_fails():
    inst = _single_request_instance(pickup=2, delivery=3)          # full battery, 30 kWh
    rep = check_solution(inst, _one_route([(0, 2), (2, 3), (3, 1)], [1], {(0, 2): 20.0}))
    assert not rep.passed
    assert any("overcharged" in v for v in rep.violations)


def test_allowed_charge_level_passes():
    inst = _single_request_instance(pickup=2, delivery=3, initial_soc=10.0)
    rep = check_solution(inst, _one_route([(0, 2), (2, 3), (3, 1)], [1], {(0, 2): 5.0}))
    assert rep.passed, rep.summary()


def test_charge_amount_outside_allowed_levels_fails():
    inst = _single_request_instance(pickup=2, delivery=3, initial_soc=10.0)
    rep = check_solution(inst, _one_route([(0, 2), (2, 3), (3, 1)], [1], {(0, 2): 7.0}))
    assert not rep.passed
    assert any("not an allowed charging level" in v for v in rep.violations)


def test_charge_on_disabled_arc_fails():
    inst = _single_request_instance(pickup=2, delivery=3, initial_soc=10.0)
    sites = dict(inst.charging.sites_by_node)
    sites[0] = dataclasses.replace(sites[0], is_enabled=False)
    inst = dataclasses.replace(inst, charging=dataclasses.replace(inst.charging, sites_by_node=sites))
    rep = check_solution(inst, _one_route([(0, 2), (2, 3), (3, 1)], [1], {(0, 2): 5.0}))
    assert not rep.passed
    assert any("not charging-enabled" in v for v in rep.violations)


def test_negative_charge_amount_fails():
    inst = _single_request_instance(pickup=2, delivery=3)
    rep = check_solution(inst, _one_route([(0, 2), (2, 3), (3, 1)], [1], {(0, 2): -5.0}))
    assert not rep.passed
    assert any("negative charging amount" in v for v in rep.violations)
