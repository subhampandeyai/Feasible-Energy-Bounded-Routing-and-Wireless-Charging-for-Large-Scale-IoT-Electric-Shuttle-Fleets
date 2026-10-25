from __future__ import annotations

import dataclasses

from src.model.types import Request
from src.solve.monolithic_milp import solve_monolithic
from src.solve.solver_interface import HighsSolver
from src.verify.checker import check_solution
from tests.mini_instance import build_mini_instance

# The monolithic MILP must answer the same question the router and the checker
# do: precedence enforced, and charging amounts restricted to the site levels.


def test_monolithic_respects_precedence():
    # Pickup 3, delivery 2: the only arc into node 2 leaves the depot, so no
    # route visits 3 before 2. Without precedence rows the MILP would claim the
    # request on 0->2->3->1 (delivery first); with them it must leave it unserved.
    inst = build_mini_instance(num_vehicles=1)
    inst = dataclasses.replace(inst, requests=(Request(id=1, pickup_node=3, delivery_node=2),))
    res = solve_monolithic(inst, HighsSolver(time_limit_s=60))
    assert res.served == {}
    assert res.unserved == (1,)


def test_monolithic_charges_only_at_allowed_levels_and_is_checker_clean():
    # Low initial SOC and high consumption: serving both requests forces charging.
    inst = build_mini_instance(num_vehicles=1)
    v = inst.vehicles[0]
    object.__setattr__(v, "initial_soc_kWh", 1.0)
    object.__setattr__(v, "max_energy_consumption_kWh_per_km", 0.8)
    res = solve_monolithic(inst, HighsSolver(time_limit_s=60))
    assert sorted(res.served) == [1, 2]
    charges = [a for r in res.routes.values() for a in r.charge.values()]
    assert charges, "this instance cannot be served without charging"
    assert all(min(abs(a - l) for l in (5.0, 15.0, 20.0)) < 1e-6 for a in charges)
    rep = check_solution(inst, res.routes)
    assert rep.passed, rep.summary()
