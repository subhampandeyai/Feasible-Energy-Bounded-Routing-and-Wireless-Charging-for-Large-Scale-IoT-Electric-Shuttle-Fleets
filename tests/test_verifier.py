from __future__ import annotations

from src.verify.checker import VehicleRoute, check_solution
from tests.mini_instance import build_mini_instance


def _clean_routes():
    # veh_1 serves r1 via 0->2->3->1; veh_2 serves r2 via 0->4->5->1
    return {
        "veh_1": VehicleRoute(arcs=[(0, 2), (2, 3), (3, 1)], charge={}, served=[1]),
        "veh_2": VehicleRoute(arcs=[(0, 4), (4, 5), (5, 1)], charge={}, served=[2]),
    }


def test_clean_solution_passes():
    inst = build_mini_instance()
    rep = check_solution(inst, _clean_routes())
    assert rep.passed, rep.summary()


def test_request_served_twice_fails():
    inst = build_mini_instance()
    routes = _clean_routes()
    routes["veh_2"].served = [1, 2]  # request 1 now served twice
    rep = check_solution(inst, routes)
    assert not rep.passed
    assert any("served by 2 vehicles" in v for v in rep.violations)


def test_request_served_zero_fails():
    inst = build_mini_instance()
    routes = _clean_routes()
    routes["veh_1"].served = []  # request 1 served by nobody
    rep = check_solution(inst, routes)
    assert not rep.passed
    assert any("served by NO vehicle" in v for v in rep.violations)


def test_nonexistent_arc_fails():
    inst = build_mini_instance()
    routes = _clean_routes()
    routes["veh_1"].arcs = [(0, 2), (2, 1)]  # (2,1) is not a real arc
    routes["veh_1"].served = [1]
    rep = check_solution(inst, routes)
    assert not rep.passed
    assert any("does not exist" in v for v in rep.violations)


def test_subtour_fails():
    inst = build_mini_instance()
    routes = _clean_routes()
    # veh_1: valid path plus a disjoint cycle 4<->5 (using real arcs 4->5,5->3,3->4)
    routes["veh_1"].arcs = [(0, 2), (2, 3), (3, 1), (4, 5), (5, 3), (3, 4)]
    rep = check_solution(inst, routes)
    assert not rep.passed


def test_soc_depletion_fails():
    inst = build_mini_instance()
    # force a vehicle with a tiny battery so the route depletes it
    v = inst.vehicles[0]
    object.__setattr__(v, "battery_capacity_kWh", 0.5)
    object.__setattr__(v, "initial_soc_kWh", 0.5)
    routes = _clean_routes()
    rep = check_solution(inst, routes)
    assert not rep.passed
    assert any("SOC" in v for v in rep.violations)


def test_time_windows_skipped_when_null():
    inst = build_mini_instance()
    rep = check_solution(inst, _clean_routes())
    assert any("time windows" in s for s in rep.skipped)
