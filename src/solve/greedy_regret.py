from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.model.types import Instance, Vehicle
from src.solve.column_generation import _empty_route_cost
from src.solve.labeling import reconstruct, solve_labeling
from src.solve.subproblem import find_subtours
from src.verify.checker import VehicleRoute

logger = logging.getLogger(__name__)

# Regret-2 greedy construction baseline. Every feasibility check (SOC,
# charging, elementarity, precedence) goes through solve_labeling's
# mandatory_ids mechanism, the same constraint code the other methods use.
#
# CONSTRUCTION: maintain each vehicle's current mandatory set (the requests
# already committed to it). For each round, for every (unassigned request,
# vehicle) pair, ask solve_labeling "what does this vehicle's route cost if
# it must serve its current mandatory set PLUS this one more request" - the
# marginal insertion cost is (that cost) - (vehicle's current committed
# cost). Regret-2: for each unassigned request, the "regret" is the gap
# between its cheapest feasible insertion and its second-cheapest across
# all vehicles - insert the request with the LARGEST regret first (the one
# that will hurt most if we delay it and its best vehicle gets taken).
# A request with only one feasible vehicle gets regret = +inf and is
# inserted first.
#
# The method is construction only; it has no local-search phase.


@dataclass
class GreedyResult:
    per_vehicle_selected: Dict[str, Optional["_ConstructedRoute"]]
    unserved: Tuple[int, ...]
    wall_time_s: float
    rounds: int


@dataclass
class _ConstructedRoute:
    vehicle_id: str
    served: Tuple[int, ...]
    cost: float
    arcs: List[Tuple[int, int]]
    charge: Dict[Tuple[int, int], float]


def _route_cost_for_mandatory_set(
    instance: Instance, vehicle: Vehicle, mandatory: frozenset, all_request_ids: Tuple[int, ...],
    time_budget_s: float, label_budget: int,
) -> Optional[Tuple[float, List[Tuple[int, int]], Dict[Tuple[int, int], float]]]:
    """Cheapest feasible route for `vehicle` that serves at least
    `mandatory` (free-choice on everything else, mu=0 so no artificial
    pricing bias - this is a real, unweighted route cost). Returns None if
    infeasible within budget (UNDETERMINED is logged, never silently
    treated as infeasible - see the caller's handling)."""
    best, stats = solve_labeling(
        instance, vehicle, all_request_ids, mu=None, time_budget_s=time_budget_s,
        mandatory_ids=mandatory, label_budget=label_budget, top_k=1,
    )
    if best is None:
        if stats["status"] in ("LABEL_BUDGET_EXCEEDED", "TIME_BUDGET_EXCEEDED"):
            logger.info("greedy_regret: vehicle %s mandatory=%s - UNDETERMINED (budget exhausted, "
                        "no incumbent) - treated as infeasible for THIS insertion attempt only, "
                        "not a proof", vehicle.id, sorted(mandatory))
        return None
    arcs, charge = reconstruct(best)
    if find_subtours(arcs):
        logger.warning("greedy_regret: vehicle %s mandatory=%s returned a non-elementary route - "
                        "discarded, not used", vehicle.id, sorted(mandatory))
        return None
    return best.cost, arcs, charge


def solve_greedy_regret(
    instance: Instance,
    time_budget_per_insertion_s: float = 15.0,
    label_budget_per_insertion: int = 200000,
    wall_budget_s: float = 3600.0,
) -> GreedyResult:
    """Regret-2 greedy construction (see the module comment); construction
    only, no local search."""
    t0 = time.time()
    vehicles = list(instance.vehicles)
    all_request_ids = tuple(r.id for r in instance.requests)
    unassigned = set(all_request_ids)
    mandatory_by_vehicle: Dict[str, frozenset] = {v.id: frozenset() for v in vehicles}
    # Each vehicle starts at the cost of its empty start->end route, which is
    # nonzero; starting at 0 would under-count the fleet objective by every
    # idle vehicle's route (the same rule as the master LP's `= 1` vehicle
    # rows in column_generation.py).
    committed_cost: Dict[str, float] = {v.id: _empty_route_cost(instance, v) for v in vehicles}
    committed_route: Dict[str, Optional[_ConstructedRoute]] = {v.id: None for v in vehicles}

    rounds = 0
    while unassigned:
        if time.time() - t0 > wall_budget_s:
            logger.warning("greedy_regret: wall budget %.1fs exceeded with %d requests still unassigned: %s",
                            wall_budget_s, len(unassigned), sorted(unassigned))
            break
        rounds += 1

        # For every unassigned request, compute its insertion cost into every
        # vehicle: the cheapest feasible route serving that vehicle's current
        # mandatory set plus this request, minus what the vehicle already costs.
        insertion_costs: Dict[int, List[Tuple[float, str, List[Tuple[int, int]], Dict[Tuple[int, int], float]]]] = {}
        any_feasible_this_round = False
        for rid in sorted(unassigned):
            per_vehicle: List[Tuple[float, str, List[Tuple[int, int]], Dict[Tuple[int, int], float]]] = []
            for veh in vehicles:
                candidate_mandatory = mandatory_by_vehicle[veh.id] | {rid}
                result = _route_cost_for_mandatory_set(
                    instance, veh, candidate_mandatory, all_request_ids,
                    time_budget_per_insertion_s, label_budget_per_insertion,
                )
                if result is None:
                    continue
                cost, arcs, charge = result
                marginal = cost - committed_cost[veh.id]
                per_vehicle.append((marginal, veh.id, arcs, charge))
            if per_vehicle:
                per_vehicle.sort(key=lambda x: x[0])
                insertion_costs[rid] = per_vehicle
                any_feasible_this_round = True

        if not any_feasible_this_round:
            logger.warning("greedy_regret: round %d - NO remaining unassigned request has ANY "
                            "feasible insertion into ANY vehicle within this round's budget - "
                            "stopping. %d requests remain unassigned: %s",
                            rounds, len(unassigned), sorted(unassigned))
            break

        # Regret-2: pick the request with the largest gap between its best
        # and second-best vehicle; a request with one feasible vehicle has
        # regret = +inf and goes first.
        def regret_key(rid: int) -> float:
            options = insertion_costs[rid]
            if len(options) == 1:
                return float("inf")
            return options[1][0] - options[0][0]

        chosen_rid = max(insertion_costs.keys(), key=regret_key)
        best_marginal, best_vehicle_id, best_arcs, best_charge = insertion_costs[chosen_rid][0]

        mandatory_by_vehicle[best_vehicle_id] = mandatory_by_vehicle[best_vehicle_id] | {chosen_rid}
        committed_cost[best_vehicle_id] = committed_cost[best_vehicle_id] + best_marginal
        committed_route[best_vehicle_id] = _ConstructedRoute(
            vehicle_id=best_vehicle_id,
            served=tuple(sorted(mandatory_by_vehicle[best_vehicle_id])),
            cost=committed_cost[best_vehicle_id],
            arcs=best_arcs, charge=best_charge,
        )
        unassigned.discard(chosen_rid)
        logger.info("greedy_regret: round %d - inserted request %d into %s (marginal_cost=%.4f, "
                    "regret=%.4f) - %d requests remain", rounds, chosen_rid, best_vehicle_id,
                    best_marginal, regret_key(chosen_rid), len(unassigned))

    # Idle vehicles keep their empty-route cost: they get a _ConstructedRoute
    # with served=() and no arcs, not None, so their cost is not read as 0.
    for veh in vehicles:
        if committed_route[veh.id] is None:
            committed_route[veh.id] = _ConstructedRoute(
                vehicle_id=veh.id, served=(), cost=committed_cost[veh.id], arcs=[], charge={},
            )

    return GreedyResult(
        per_vehicle_selected=committed_route,
        unserved=tuple(sorted(unassigned)),
        wall_time_s=time.time() - t0,
        rounds=rounds,
    )


def to_vehicle_routes(selected: Dict[str, Optional[_ConstructedRoute]]) -> Dict[str, VehicleRoute]:
    out = {}
    for vid, route in selected.items():
        if route is not None:
            out[vid] = VehicleRoute(arcs=route.arcs, charge=route.charge, served=list(route.served))
    return out
