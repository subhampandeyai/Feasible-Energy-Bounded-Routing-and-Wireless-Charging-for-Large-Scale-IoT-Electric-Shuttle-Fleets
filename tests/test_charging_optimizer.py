from __future__ import annotations

import dataclasses
import itertools

from src.model.types import PriceSchedule
from src.solve.charging_optimizer import ALLOWED_LEVELS, optimize_charging, plan_is_soc_valid
from tests.mini_instance import build_mini_instance

# The per-route charging program is exact on the discrete level set. This checks
# it against brute-force enumeration of every level assignment on a route
# where charging is required and prices differ by node.

SEQ = [0, 2, 3, 4, 5, 1]


def _instance(initial_soc=5.0, consumption=1.0):
    inst = build_mini_instance(num_vehicles=1)
    prices = PriceSchedule(hourly_multiplier=tuple([1.0] * 24),
                           base_price_per_node={0: 0.95, 1: 0.9, 2: 0.6, 3: 0.85, 4: 0.55, 5: 0.99})
    inst = dataclasses.replace(inst, prices=prices)
    v = inst.vehicles[0]
    object.__setattr__(v, "initial_soc_kWh", initial_soc)
    object.__setattr__(v, "max_energy_consumption_kWh_per_km", consumption)
    return inst


def _arcdist(inst):
    return {(a.from_node, a.to_node): a.distance_km for a in inst.arcs}


def _brute_force(inst, reserve):
    """Minimum cost over every assignment of ALLOWED_LEVELS to the route's arcs, under the
    optimiser's own semantics: charge on arc (i,j) before traversing it, SOC never above
    capacity after charging, never below the reserve after the arc."""
    v, ad = inst.vehicles[0], _arcdist(inst)
    arcs = list(zip(SEQ, SEQ[1:]))
    best = None
    for amounts in itertools.product(ALLOWED_LEVELS, repeat=len(arcs)):
        soc, cost, ok = v.initial_soc_kWh, 0.0, True
        for (i, j), amt in zip(arcs, amounts):
            if amt > 0 and not inst.charging.arc_enabled(i, j):
                ok = False; break
            soc += amt
            if soc > v.battery_capacity_kWh + 1e-9:
                ok = False; break
            soc -= v.max_energy_consumption_kWh_per_km * ad[(i, j)]
            if soc < reserve - 1e-9:
                ok = False; break
            cost += inst.prices.base_price_per_node[i] * amt
        if ok and (best is None or cost < best - 1e-12):
            best = cost
    return best


def test_dp_matches_brute_force_without_reserve():
    inst = _instance()
    plan = optimize_charging(inst, inst.vehicles[0], SEQ, _arcdist(inst), reserve_kWh=0.0)
    assert plan.feasible
    assert abs(plan.energy_cost - _brute_force(inst, 0.0)) < 1e-9
    assert plan_is_soc_valid(plan, inst.vehicles[0], 0.0)


def test_dp_matches_brute_force_with_reserve():
    inst = _instance()
    plan = optimize_charging(inst, inst.vehicles[0], SEQ, _arcdist(inst), reserve_kWh=3.0)
    assert plan.feasible
    assert abs(plan.energy_cost - _brute_force(inst, 3.0)) < 1e-9
    assert plan_is_soc_valid(plan, inst.vehicles[0], 3.0)
    assert min(plan.soc_profile) >= 3.0 - 1e-9


def test_dp_uses_only_allowed_levels():
    inst = _instance()
    plan = optimize_charging(inst, inst.vehicles[0], SEQ, _arcdist(inst), reserve_kWh=3.0)
    assert all(a in ALLOWED_LEVELS and a > 0 for a in plan.charge.values())


def test_dp_reports_infeasible_when_no_plan_exists():
    # Start below the reserve with no way to recover before the first arc: no plan exists.
    inst = _instance(initial_soc=1.0, consumption=5.0)
    plan = optimize_charging(inst, inst.vehicles[0], SEQ, _arcdist(inst), reserve_kWh=25.0)
    assert not plan.feasible
    assert _brute_force(inst, 25.0) is None
