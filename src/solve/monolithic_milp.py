from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.model.formulation import build_vehicle_subproblem
from src.model.milp_ir import MILPProblem
from src.model.types import Instance
from src.solve.column_generation import ARTIFICIAL_COST_PER_REQUEST
from src.solve.solver_interface import SolverInterface, SolveResult
from src.verify.checker import VehicleRoute

logger = logging.getLogger(__name__)

# Monolithic fleet-wide MILP: every vehicle's routing/charging/SOC
# subproblem (build_vehicle_subproblem in src/model/formulation.py, the same
# constraint families the decomposition methods use) assembled into one
# MILPProblem, coupled by equation (10), sum_k q_u^k = 1, relaxed to
# sum_k q_u^k + slack_u = 1 with slack_u costed at ARTIFICIAL_COST_PER_REQUEST
# (the same unserved penalty as column_generation.py). The slack only keeps
# the model feasible when not every request can be served; there is no
# decomposition or other relaxation.
#
# At 10 vehicles and R=30 the model has tens of thousands of binaries and may
# not close within the time limit; the result reports what HiGHS proves
# (objective, mip_gap, mip_dual_bound).
#
# PARITY: each vehicle block enforces pickup-before-delivery precedence
# (include_precedence) and restricts charging to the site's discrete levels
# (discrete_charge_levels), so the model answers the same question the
# router, the charging DP and the independent checker do. Without them the
# MILP can claim a request on a delivery-first path or charge e.g. 7.38 kWh,
# and its plans are then not comparable (tests/test_monolithic_parity.py).


@dataclass
class MonolithicResult:
    solve_result: SolveResult
    routes: Dict[str, VehicleRoute]
    served: Dict[int, str]  # request_id -> vehicle_id that serves it (only entries actually served)
    unserved: Tuple[int, ...]
    true_routing_cost: float  # objective MINUS slack penalty terms - the actual epsilon1/epsilon2 cost
    penalty_cost: float  # slack terms' contribution to the objective


def build_monolithic_milp(instance: Instance) -> Tuple[MILPProblem, Dict[str, dict], Dict[int, int]]:
    """Assemble one fleet-wide MILPProblem out of per-vehicle subproblems
    (build_vehicle_subproblem, free-choice q, mu=None i.e. no Lagrangian
    term - this is the RAW per-vehicle cost, coupling is added here instead)
    plus one relaxed assignment-coupling row per request.

    Returns (problem, per_vehicle_var_index, slack_var_by_request) where
    per_vehicle_var_index[vehicle.id] holds that vehicle's SubproblemVarIndex
    (x/c/B/q/t dicts), reindexed to the merged problem's variable numbering,
    and slack_var_by_request[request_id] is the merged problem's index for
    that request's unserved-penalty slack variable.
    """
    problem = MILPProblem(name="monolithic_fleet", minimize=True)
    per_vehicle_var_index: Dict[str, dict] = {}

    request_ids = tuple(r.id for r in instance.requests)
    # accumulate, per request, the merged-problem q-variable index for every
    # vehicle that has one, to build the coupling row.
    q_vars_by_request: Dict[int, List[int]] = {u: [] for u in request_ids}

    for veh in instance.vehicles:
        sub_problem, sub_idx = build_vehicle_subproblem(
            instance, veh, instance.requests,
            multipliers=None, include_mtz=True, include_soc=True,
            include_redundant_67=True, include_flow_conservation=True,
            include_precedence=True, include_gg_flow=False,
            discrete_charge_levels=True,
        )
        offset = problem.num_vars
        # copy variables
        for i in range(sub_problem.num_vars):
            problem.add_variable(
                name=f"{veh.id}__{sub_problem.var_names[i]}",
                lower=sub_problem.var_lower[i],
                upper=sub_problem.var_upper[i],
                integer=sub_problem.var_integer[i],
                obj=sub_problem.obj_coeffs[i],
            )
        # copy constraints, reindexed by offset
        for r in range(sub_problem.num_rows):
            coeffs = {offset + var_idx: coeff for var_idx, coeff in sub_problem._row_entries[r]}
            problem.add_constraint(
                f"{veh.id}__{sub_problem.row_names[r]}",
                coeffs, sub_problem.row_lower[r], sub_problem.row_upper[r],
            )

        reindexed_idx = {
            "x": {k: offset + v for k, v in sub_idx.x.items()},
            "c": {k: offset + v for k, v in sub_idx.c.items()},
            "B": {k: offset + v for k, v in sub_idx.B.items()},
            "q": {k: offset + v for k, v in sub_idx.q.items()},
            "t": {k: offset + v for k, v in sub_idx.t.items()},
        }
        per_vehicle_var_index[veh.id] = reindexed_idx
        for u in request_ids:
            q_vars_by_request[u].append(reindexed_idx["q"][u])

    # --- fleet-wide coupling: sum_k q_u^k + slack_u = 1, slack costed at
    # the unserved penalty. Equality (not <= 1): a request is served by
    # exactly one vehicle or its slack absorbs the gap, so it is never
    # dropped silently and never double-counted.
    slack_var_by_request: Dict[int, int] = {}
    for u in request_ids:
        slack_idx = problem.add_variable(
            name=f"slack_unserved_{u}", lower=0.0, upper=1.0, integer=False,
            obj=ARTIFICIAL_COST_PER_REQUEST,
        )
        slack_var_by_request[u] = slack_idx
        coeffs = {qv: 1.0 for qv in q_vars_by_request[u]}
        coeffs[slack_idx] = 1.0
        problem.add_constraint(f"coupling_{u}", coeffs, 1.0, 1.0)

    return problem, per_vehicle_var_index, slack_var_by_request


def solve_monolithic(instance: Instance, solver: SolverInterface) -> MonolithicResult:
    problem, per_vehicle_var_index, slack_var_by_request = build_monolithic_milp(instance)
    logger.info("monolithic fleet MILP: %d vehicles, %d requests, %d variables, %d rows",
                len(instance.vehicles), len(instance.requests), problem.num_vars, problem.num_rows)

    result = solver.solve(problem)

    routes: Dict[str, VehicleRoute] = {}
    served: Dict[int, str] = {}
    true_routing_cost = 0.0
    penalty_cost = 0.0

    if result.col_values:
        arc_dist = {(a.from_node, a.to_node): a.distance_km for a in instance.arcs}
        eps1 = instance.objective.epsilon1
        eps2 = instance.objective.epsilon2

        for veh in instance.vehicles:
            idx = per_vehicle_var_index[veh.id]
            arcs = [(i, j) for (i, j), vi in idx["x"].items() if result.col_values[vi] > 0.5]
            charge = {(i, j): result.col_values[idx["c"][(i, j)]]
                      for (i, j) in idx["c"] if result.col_values[idx["c"][(i, j)]] > 1e-6}
            served_here = [u for u, vi in idx["q"].items() if result.col_values[vi] > 0.5]
            routes[veh.id] = VehicleRoute(arcs=arcs, charge=charge, served=served_here)
            for u in served_here:
                served[u] = veh.id
            for (i, j) in arcs:
                true_routing_cost += eps1 * arc_dist.get((i, j), 0.0)
            for (i, j), amt in charge.items():
                price = instance.prices.base_price_per_node.get(i, 0.0)
                eta = 1.0 if instance.charging.arc_enabled(i, j) else 0.0
                true_routing_cost += eps2 * price * eta * amt

        for u, vi in slack_var_by_request.items():
            penalty_cost += ARTIFICIAL_COST_PER_REQUEST * result.col_values[vi]

    unserved = tuple(sorted(u for u in slack_var_by_request if u not in served))

    return MonolithicResult(
        solve_result=result, routes=routes, served=served, unserved=unserved,
        true_routing_cost=true_routing_cost, penalty_cost=penalty_cost,
    )
