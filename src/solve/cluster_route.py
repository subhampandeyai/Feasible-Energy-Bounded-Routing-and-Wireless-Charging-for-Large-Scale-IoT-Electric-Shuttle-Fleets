from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from src.model.types import Instance, Vehicle
from src.solve.insertion_router import route_cluster_insertion, try_add_one
from src.solve.labeling import reconstruct, solve_labeling
from src.solve.subproblem import find_subtours
from src.verify.checker import VehicleRoute, check_solution

logger = logging.getLogger(__name__)

# Cluster-first / route-second solver.
#
# Solving routing and assignment jointly is dominated by the combinatorial
# blow-up of the labeling solver's (onboard, delivered) state space when many
# requests are simultaneously optional inside one call. This module instead
# fixes the assignment first, by spatial partitioning, and then routes each
# vehicle over its own small cluster.
#
# INSTANCE PROPERTIES THIS RELIES ON:
#   * No time windows and no max-ride-time. The only constraints are: arcs
#     exist, a single connected start->end path, SOC within [0, capacity] with
#     charging, pickup-before-delivery, and each request served exactly once.
#     There is no inter-vehicle temporal coupling, so the fleet problem
#     separates once the assignment is fixed.
#   * All vehicles are structurally identical (same start, end, battery and
#     consumption), so the assignment is a pure partitioning choice and a
#     request no single vehicle can serve alone cannot be served by any.
#
# METHOD.
#   1. Exclude proven-infeasible requests (solo-feasibility oracle below).
#   2. Partition the remaining requests into up to K spatial clusters (K =
#      number of vehicles), balanced by count, using request midpoints. This
#      fixes the assignment outside any MILP or labeling search.
#   3. Route each vehicle over its own cluster. The "insertion" engine grows
#      the route by node-disjoint insertion (src/solve/insertion_router.py).
#      The "labeling" engine solves labeling over the cluster with every
#      request mandatory, and if that fails it adds requests one at a time,
#      keeping the largest feasible prefix.
#   4. Offer every request a vehicle could not place to the other vehicles in
#      reinsertion passes.
#   5. Every accepted route passes the independent checker.


@dataclass
class VehiclePlan:
    vehicle_id: str
    served: Tuple[int, ...]
    cost: float
    arcs: List[Tuple[int, int]]
    charge: Dict[Tuple[int, int], float]


@dataclass
class ClusterResult:
    per_vehicle: Dict[str, VehiclePlan]
    served: Tuple[int, ...]
    unserved_feasible: Tuple[int, ...]      # feasible but we failed to place (NOT a proof of infeasibility)
    proven_infeasible: Tuple[int, ...]      # no single vehicle can serve solo (search exhausted)
    undetermined_feasibility: Tuple[int, ...]  # solo search hit budget - status unknown
    wall_time_s: float


def _midpoint(instance: Instance, rid: int) -> Tuple[float, float]:
    r = instance.request_by_id(rid)
    p = instance.nodes[r.pickup_node]
    d = instance.nodes[r.delivery_node]
    return ((p.x_km + d.x_km) / 2.0, (p.y_km + d.y_km) / 2.0)


def solo_feasibility(
    instance: Instance,
    vehicle: Vehicle,
    time_budget_s: float = 8.0,
) -> Tuple[List[int], List[int], List[int]]:
    """Classify every request as feasible / proven-infeasible / undetermined
    by asking whether a single identical vehicle can serve it alone.

    feasible: labeling returned a route whose OWN structure passes the checker
      (the checker's fleet-wide 'other request unserved' lines are filtered -
      they are not this route's defect).
    proven_infeasible: labeling exhausted its search and returned no route
      (status INFEASIBLE) - a real proof for identical vehicles.
    undetermined: labeling hit its time budget without a route - NOT a proof.
    """
    feasible: List[int] = []
    infeasible: List[int] = []
    undetermined: List[int] = []
    for r in instance.requests:
        best, stats = solve_labeling(
            instance, vehicle, (r.id,), mu={}, time_budget_s=time_budget_s,
            mandatory_ids=frozenset({r.id}), top_k=1,
        )
        if best is not None and r.id in best.delivered:
            arcs, charge = reconstruct(best)
            if find_subtours(arcs):
                undetermined.append(r.id)
                continue
            route = VehicleRoute(arcs=arcs, charge=charge, served=[r.id])
            rep = check_solution(instance, {vehicle.id: route})
            own = [v for v in rep.violations if "served by NO vehicle" not in v]
            (feasible if not own else undetermined).append(r.id)
        elif stats["status"] == "INFEASIBLE":
            infeasible.append(r.id)
        else:
            undetermined.append(r.id)
    return feasible, infeasible, undetermined


def _balanced_spatial_clusters(
    instance: Instance, request_ids: List[int], k: int
) -> List[List[int]]:
    """Partition request_ids into k count-balanced spatial clusters by
    midpoint, via Lloyd-style k-means with a balancing assignment. Deterministic
    (seeded by request order), no randomness, so results reproduce."""
    if not request_ids:
        return [[] for _ in range(k)]
    pts = {rid: _midpoint(instance, rid) for rid in request_ids}
    # init centroids: spread picks (farthest-first) for determinism/coverage
    centroids: List[Tuple[float, float]] = [pts[request_ids[0]]]
    while len(centroids) < k:
        far, fardist = None, -1.0
        for rid in request_ids:
            dmin = min(math.hypot(pts[rid][0] - c[0], pts[rid][1] - c[1]) for c in centroids)
            if dmin > fardist:
                fardist, far = dmin, pts[rid]
        centroids.append(far)

    target = math.ceil(len(request_ids) / k)
    clusters: List[List[int]] = [[] for _ in range(k)]
    for _ in range(25):
        clusters = [[] for _ in range(k)]
        counts = [0] * k
        # assign each request to nearest centroid that still has room; if the
        # nearest is full, spill to the next-nearest with room (balancing).
        order = sorted(request_ids, key=lambda rid: min(
            math.hypot(pts[rid][0] - c[0], pts[rid][1] - c[1]) for c in centroids))
        for rid in order:
            ranked = sorted(range(k), key=lambda ci: math.hypot(
                pts[rid][0] - centroids[ci][0], pts[rid][1] - centroids[ci][1]))
            placed = False
            for ci in ranked:
                if counts[ci] < target:
                    clusters[ci].append(rid); counts[ci] += 1; placed = True; break
            if not placed:
                ci = ranked[0]; clusters[ci].append(rid); counts[ci] += 1
        new_centroids = []
        for ci in range(k):
            if clusters[ci]:
                xs = [pts[r][0] for r in clusters[ci]]
                ys = [pts[r][1] for r in clusters[ci]]
                new_centroids.append((sum(xs) / len(xs), sum(ys) / len(ys)))
            else:
                new_centroids.append(centroids[ci])
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return clusters


def _solve_vehicle_set(
    instance: Instance, vehicle: Vehicle, req_ids: Tuple[int, ...],
    time_budget_s: float, label_budget: Optional[int],
) -> Optional[Tuple[float, Tuple[int, ...], List[Tuple[int, int]], Dict[Tuple[int, int], float]]]:
    """Solve labeling with candidate set = ONLY req_ids, all mandatory. Fast
    regime (small optional set). Returns (cost, served, arcs, charge) if a
    checker-clean route serving all of req_ids is found, else None."""
    if not req_ids:
        return 0.0, (), [], {}
    best, stats = solve_labeling(
        instance, vehicle, req_ids, mu={}, time_budget_s=time_budget_s,
        mandatory_ids=frozenset(req_ids), label_budget=label_budget, top_k=1,
    )
    if best is None or not frozenset(req_ids).issubset(best.delivered):
        return None
    arcs, charge = reconstruct(best)
    if find_subtours(arcs):
        return None
    route = VehicleRoute(arcs=arcs, charge=charge, served=sorted(best.delivered))
    rep = check_solution(instance, {vehicle.id: route})
    own = [v for v in rep.violations if "served by NO vehicle" not in v]
    if own:
        return None
    return best.cost, tuple(sorted(best.delivered)), arcs, charge


def _greedy_grow_vehicle(
    instance: Instance, vehicle: Vehicle, candidate_ids: List[int],
    per_step_budget_s: float, label_budget: Optional[int],
) -> Tuple[List[int], Optional[Tuple[float, Tuple[int, ...], List, Dict]], List[int]]:
    """Add requests from candidate_ids to `vehicle` one at a time (spatial
    nearest-first), keeping only those that stay feasible. candidate set for
    each labeling call = the committed requests + the one being tried, so
    every call stays in the fast labeling regime. Returns
    (committed_ids, best_plan_tuple, dropped_ids)."""
    committed: List[int] = []
    best_plan = None
    dropped: List[int] = []
    # order candidates by nearest to the vehicle's running centroid (start with
    # start-node position, then the growing cluster centroid)
    remaining = list(candidate_ids)
    while remaining:
        if committed:
            cx = sum(_midpoint(instance, r)[0] for r in committed) / len(committed)
            cy = sum(_midpoint(instance, r)[1] for r in committed) / len(committed)
        else:
            s = instance.nodes[vehicle.start_node]
            cx, cy = s.x_km, s.y_km
        remaining.sort(key=lambda r: math.hypot(_midpoint(instance, r)[0] - cx,
                                                 _midpoint(instance, r)[1] - cy))
        rid = remaining.pop(0)
        trial = tuple(sorted(committed + [rid]))
        res = _solve_vehicle_set(instance, vehicle, trial, per_step_budget_s, label_budget)
        if res is not None:
            committed = list(trial)
            best_plan = res
        else:
            dropped.append(rid)
    return committed, best_plan, dropped


def _route_cost(instance: Instance, arcs: List[Tuple[int, int]],
                charge: Dict[Tuple[int, int], float]) -> float:
    arcdist = {(a.from_node, a.to_node): a.distance_km for a in instance.arcs}
    eps1 = instance.objective.epsilon1
    eps2 = instance.objective.epsilon2
    dist = sum(arcdist.get(a, 0.0) for a in arcs)
    charge_cost = 0.0
    for (i, j), c in charge.items():
        if instance.charging.arc_enabled(i, j):
            price = instance.prices.base_price_per_node.get(i, 0.0)
            charge_cost += eps2 * price * c
    return eps1 * dist + charge_cost


def solve_cluster_route(
    instance: Instance,
    solo_budget_s: float = 8.0,
    n_candidates: int = 14,
    reinsert_passes: int = 3,
    precomputed_solo: Optional[Tuple[List[int], List[int], List[int]]] = None,
    engine: str = "insertion",
    # used only by the labeling engine:
    cluster_solve_budget_s: float = 20.0,
    greedy_step_budget_s: float = 8.0,
    label_budget: Optional[int] = 400000,
) -> ClusterResult:
    t0 = time.time()
    vehicles = list(instance.vehicles)
    k = len(vehicles)

    if precomputed_solo is not None:
        feasible, infeasible, undetermined = precomputed_solo
    else:
        logger.info("cluster_route: solo-feasibility scan (%d requests, %.1fs/req)...",
                    len(instance.requests), solo_budget_s)
        feasible, infeasible, undetermined = solo_feasibility(instance, vehicles[0], solo_budget_s)
    logger.info("cluster_route: solo scan done: %d feasible, %d proven-infeasible=%s, %d undetermined=%s",
                len(feasible), len(infeasible), sorted(infeasible),
                len(undetermined), sorted(undetermined))

    placeable = sorted(set(feasible) | set(undetermined))
    clusters = _balanced_spatial_clusters(instance, placeable, k)
    logger.info("cluster_route: %d placeable requests -> cluster sizes %s (engine=%s)",
                len(placeable), [len(c) for c in clusters], engine)

    plans: Dict[str, VehiclePlan] = {}
    pool: List[int] = []
    for veh, cluster in zip(vehicles, clusters):
        if not cluster:
            plans[veh.id] = VehiclePlan(veh.id, (), 0.0, [], {})
            continue
        if engine == "insertion":
            served, arcs, charge = route_cluster_insertion(
                instance, veh, list(cluster), n_candidates=n_candidates)
            cost = _route_cost(instance, arcs, charge)
            plans[veh.id] = VehiclePlan(veh.id, tuple(sorted(served)), cost, arcs, charge)
            dropped = sorted(set(cluster) - set(served))
            pool.extend(dropped)
            logger.info("cluster_route: %s insertion-routed cluster of %d -> served=%d %s, dropped=%s",
                        veh.id, len(cluster), len(served), sorted(served), dropped)
        else:
            res = _solve_vehicle_set(instance, veh, tuple(sorted(cluster)),
                                     cluster_solve_budget_s, label_budget)
            if res is not None:
                cost, served, arcs, charge = res
                plans[veh.id] = VehiclePlan(veh.id, served, cost, arcs, charge)
            else:
                committed, plan, dropped = _greedy_grow_vehicle(
                    instance, veh, cluster, greedy_step_budget_s, label_budget)
                if plan is not None:
                    cost, served, arcs, charge = plan
                    plans[veh.id] = VehiclePlan(veh.id, served, cost, arcs, charge)
                else:
                    plans[veh.id] = VehiclePlan(veh.id, (), 0.0, [], {})
                pool.extend(dropped)

    # Reinsertion: offer each dropped request to every vehicle, nearest first,
    # through the incremental splice (try_add_one), which is much cheaper than
    # re-routing a whole cluster. Each vehicle's current node sequence is
    # tracked so the splice works on its actual path.
    veh_by_id = {v.id: v for v in vehicles}

    def seq_from_arcs(veh: Vehicle, arcs: List[Tuple[int, int]]) -> List[int]:
        if not arcs:
            return [veh.start_node, veh.end_node]
        succ = {i: j for (i, j) in arcs}
        seq = [veh.start_node]
        cur = veh.start_node
        while cur in succ:
            cur = succ[cur]
            seq.append(cur)
        return seq

    node_seqs: Dict[str, List[int]] = {
        vid: seq_from_arcs(veh_by_id[vid], pl.arcs) for vid, pl in plans.items()
    }

    for p in range(reinsert_passes):
        if not pool:
            break
        still: List[int] = []
        progressed = False
        for rid in pool:
            placed = False

            def vkey(veh):
                pl = plans[veh.id]
                if pl.served:
                    cx = sum(_midpoint(instance, r)[0] for r in pl.served) / len(pl.served)
                    cy = sum(_midpoint(instance, r)[1] for r in pl.served) / len(pl.served)
                else:
                    s = instance.nodes[veh.start_node]; cx, cy = s.x_km, s.y_km
                m = _midpoint(instance, rid)
                return math.hypot(m[0] - cx, m[1] - cy)

            for veh in sorted(vehicles, key=vkey):
                current = list(plans[veh.id].served)
                res = try_add_one(instance, veh, node_seqs[veh.id], current, rid,
                                  n_candidates=n_candidates)
                if res is not None:
                    ns, arcs, charge = res
                    cost = _route_cost(instance, arcs, charge)
                    plans[veh.id] = VehiclePlan(veh.id, tuple(sorted(current + [rid])), cost, arcs, charge)
                    node_seqs[veh.id] = ns
                    placed = True
                    progressed = True
                    logger.info("cluster_route: reinsert pass %d placed req %d into %s (now serves %d)",
                                p, rid, veh.id, len(current) + 1)
                    break
            if not placed:
                still.append(rid)
        pool = still
        if not progressed:
            break

    served_all = tuple(sorted(r for pl in plans.values() for r in pl.served))
    unserved_feasible = tuple(sorted(set(placeable) - set(served_all)))
    return ClusterResult(
        per_vehicle=plans,
        served=served_all,
        unserved_feasible=unserved_feasible,
        proven_infeasible=tuple(sorted(infeasible)),
        undetermined_feasibility=tuple(sorted(set(undetermined) - set(served_all))),
        wall_time_s=time.time() - t0,
    )


def to_vehicle_routes(plans: Dict[str, VehiclePlan]) -> Dict[str, VehicleRoute]:
    out: Dict[str, VehicleRoute] = {}
    for vid, pl in plans.items():
        out[vid] = VehicleRoute(arcs=pl.arcs, charge=pl.charge, served=list(pl.served))
    return out
