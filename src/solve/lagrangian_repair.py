from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from src.model.types import Instance
from src.solve.cluster_route import _route_cost
from src.solve.column_generation import ARTIFICIAL_COST_PER_REQUEST
from src.solve.insertion_router import _charge_repair
from src.solve.subproblem import SubproblemResult
from src.verify.checker import VehicleRoute, check_solution

logger = logging.getLogger(__name__)

# Primal recovery for the Lagrangian (Gavish-Graves flow) baseline.
#
# The Lagrangian relaxes the "served exactly once" coupling, so its per-vehicle
# subproblem solutions may serve a request on several vehicles or on none, may
# carry continuous charging amounts, and may visit a delivery before its pickup.
# This function assembles them into a plan that is feasible for the original
# problem, using only what the subproblems produced:
#
#   1. Keep a vehicle's route only if its subproblem solution is a single
#      subtour-free start->end path (walked from the solution's x values).
#   2. On that path, a request counts as served only if its pickup is visited
#      strictly before its delivery and the subproblem selected it.
#   3. A request claimed by several vehicles is kept on the first vehicle in
#      fleet order that serves it (deterministic) and dropped from the others.
#   4. Charging is set by the same reactive minimum-level rule the
#      proposed router uses (src/solve/insertion_router._charge_repair), so
#      amounts come from the allowed levels {5, 15, 20} kWh. A path that cannot
#      be made SOC-feasible is dropped.
#   5. Every kept route is validated with the independent checker; a route the
#      checker rejects is dropped, never patched.
#
# Unserved requests are allowed and are costed at ARTIFICIAL_COST_PER_REQUEST,
# the same penalty the column-generation and monolithic baselines use, so the
# returned objective is on the same scale as theirs. Returns None only when no
# vehicle route survives.


def _walk(start: int, end: int, arcs: List[Tuple[int, int]]) -> Optional[List[int]]:
    """Node sequence of a single start->end path using every arc once, else None."""
    if not arcs:
        return None
    succ: Dict[int, int] = {}
    for i, j in arcs:
        if i in succ:
            return None                     # branching: not a simple path
        succ[i] = j
    seq, seen = [start], {start}
    while seq[-1] in succ:
        nxt = succ[seq[-1]]
        if nxt in seen:
            return None                     # cycle
        seq.append(nxt)
        seen.add(nxt)
    if seq[-1] != end or len(seq) - 1 != len(arcs):
        return None                         # does not end at e, or disconnected fragments
    return seq


def repair_from_subproblems(
    instance: Instance,
    sub_results: Dict[str, SubproblemResult],
    solver=None,
) -> Optional[dict]:
    arcdist = {(a.from_node, a.to_node): a.distance_km for a in instance.arcs}
    req_by_id = {r.id: r for r in instance.requests}
    claimed: set = set()
    routes: Dict[str, VehicleRoute] = {}

    for veh in instance.vehicles:
        res = sub_results.get(veh.id)
        if res is None or not res.subtour_free:
            continue
        arcs = [a for a, v in res.x_values.items() if v > 0.5]
        seq = _walk(veh.start_node, veh.end_node, arcs)
        if seq is None:
            continue
        pos = {n: k for k, n in enumerate(seq)}
        selected = set(res.served_requests) | {u for u, q in res.q_values.items() if q > 0.5}
        served = []
        for u in sorted(selected):
            r = req_by_id.get(u)
            if r is None or u in claimed:
                continue
            p, d = pos.get(r.pickup_node), pos.get(r.delivery_node)
            if p is not None and d is not None and p < d:
                served.append(u)
        if not served:
            continue                         # an empty vehicle may not drive a route
        charge, ok = _charge_repair(instance, veh, seq, arcdist)
        if not ok:
            logger.info("lagrangian_repair: %s path is not SOC-feasible under reactive charging - dropped", veh.id)
            continue
        path_arcs = [(seq[k], seq[k + 1]) for k in range(len(seq) - 1)]
        route = VehicleRoute(arcs=path_arcs, charge=charge or {}, served=served)
        rep = check_solution(instance, {veh.id: route})
        own = [v for v in rep.violations if "served by NO vehicle" not in v]
        if own:
            logger.info("lagrangian_repair: %s route rejected by checker (%s) - dropped", veh.id, own[0])
            continue
        routes[veh.id] = route
        claimed.update(served)

    if not routes:
        logger.info("lagrangian_repair: no subproblem route could be recovered")
        return None
    unserved = [r.id for r in instance.requests if r.id not in claimed]
    objective = (sum(_route_cost(instance, rt.arcs, rt.charge) for rt in routes.values())
                 + ARTIFICIAL_COST_PER_REQUEST * len(unserved))
    logger.info("lagrangian_repair: recovered %d vehicle route(s) serving %d request(s); %d unserved; objective %.4f",
                len(routes), len(claimed), len(unserved), objective)
    return {"solution": routes, "objective": objective, "value": objective,
            "served": sorted(claimed), "unserved": unserved}
