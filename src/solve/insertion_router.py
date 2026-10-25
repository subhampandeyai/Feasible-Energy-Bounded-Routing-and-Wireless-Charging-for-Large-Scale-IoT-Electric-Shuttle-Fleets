from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

from src.model.types import Instance, Vehicle
from src.verify.checker import VehicleRoute, check_solution

# Node-disjoint insertion router: the per-vehicle routing engine used by the
# cluster-first method in src/solve/cluster_route.py.
#
# PROBLEM STRUCTURE. Requests carry no time windows and no maximum ride time,
# so the problem is spatial plus energy. On a sparse 5-nearest-neighbour mesh
# the main difficulty is elementarity: routes compete for the same narrow
# corridors.
#
# METHOD. Grow an elementary node path start->...->end one request at a time.
# To add a request, splice its pickup and then its delivery into the current
# sequence using node-disjoint shortest subpaths (Dijkstra that refuses
# already-used nodes) at candidate positions near the pickup and delivery
# nodes, and keep the shortest splice that stays elementary. A forward charge
# repair (_charge_repair) then buys the smallest allowed level on a
# charging-enabled arc wherever SOC would fall below zero; if it cannot, the
# request is not inserted. Every accepted step is validated by the independent
# checker, so the router never emits a route the checker would reject. Three
# request orderings are tried and the one serving the most requests is kept.

CHARGE_LEVELS = (5.0, 15.0, 20.0)


def _dijkstra(instance: Instance, source: int, blocked: set, target: int
              ) -> Optional[List[int]]:
    """Shortest path source->target that never steps onto a node in `blocked`
    (target itself is always allowed). Returns the node path or None."""
    dist = {source: 0.0}
    prev: Dict[int, int] = {}
    pq: List[Tuple[float, int]] = [(0.0, source)]
    while pq:
        d, n = heapq.heappop(pq)
        if n == target:
            break
        if d > dist.get(n, float("inf")) + 1e-9:
            continue
        for a in instance.out_arcs.get(n, ()):
            j = a.to_node
            if j in blocked and j != target:
                continue
            nd = d + a.distance_km
            if nd < dist.get(j, float("inf")) - 1e-9:
                dist[j] = nd
                prev[j] = n
                heapq.heappush(pq, (nd, j))
    if target not in dist:
        return None
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    return list(reversed(path))


def _charge_repair(instance: Instance, vehicle: Vehicle,
                   node_seq: List[int], arcdist: Dict[Tuple[int, int], float]
                   ) -> Tuple[Optional[Dict[Tuple[int, int], float]], bool]:
    """Greedy forward SOC repair: charge the smallest allowed amount on a
    charging-enabled arc whenever the next arc would deplete the battery.
    Returns (charge_dict, feasible)."""
    cons = vehicle.max_energy_consumption_kWh_per_km or instance.energy.consumption_kWh_per_km
    cap = vehicle.battery_capacity_kWh
    arcs = [(node_seq[i], node_seq[i + 1]) for i in range(len(node_seq) - 1)]
    # An "empty" plan is the bare [start, end] pair, and on this graph the arc
    # (start, end) need not exist. Any arc not in the graph makes this node
    # sequence unusable - report infeasible rather than raising KeyError.
    if any(a not in arcdist for a in arcs):
        return None, False
    # feasible with no charging at all?
    soc = vehicle.initial_soc_kWh
    ok = True
    for (i, j) in arcs:
        soc -= cons * arcdist[(i, j)]
        if soc < -1e-9:
            ok = False
            break
    if ok:
        return {}, True
    charge: Dict[Tuple[int, int], float] = {}
    soc = vehicle.initial_soc_kWh
    for (i, j) in arcs:
        need = cons * arcdist[(i, j)]
        if soc - need < 0 and instance.charging.arc_enabled(i, j):
            for c in CHARGE_LEVELS:
                if soc + c <= cap + 1e-9 and soc + c - need >= 0:
                    charge[(i, j)] = c
                    soc += c
                    break
        soc -= need
        if soc < -1e-9:
            return None, False
    return charge, True


def _try_insert(instance: Instance, vehicle: Vehicle, node_seq: List[int],
                pu: int, de: int, arcdist: Dict[Tuple[int, int], float],
                n_candidates: int = 14) -> Optional[List[int]]:
    """Insert pickup pu then delivery de into node_seq (an elementary
    start..end path), keeping the result elementary. Returns the new node
    sequence or None. Candidate splice positions are the sequence indices
    whose node is spatially nearest pu (for pickup) / de (for delivery)."""
    used = set(node_seq)
    if pu in used or de in used:
        return None
    n = len(node_seq)
    nodes = instance.nodes

    def near_positions(target: int) -> List[int]:
        tp = nodes[target]
        idxs = sorted(range(n - 1), key=lambda i: math.hypot(
            nodes[node_seq[i]].x_km - tp.x_km, nodes[node_seq[i]].y_km - tp.y_km))
        return idxs[:n_candidates]

    best: Optional[Tuple[float, List[int]]] = None
    for i in near_positions(pu):
        for j in near_positions(de):
            if j < i:
                continue
            newseq = list(node_seq[:i + 1])
            cur_used = set(newseq)
            # seam: node_seq[i] -> pu
            seg = _dijkstra(instance, node_seq[i], set(node_seq) - {node_seq[i]}, pu)
            if seg is None:
                continue
            tail = seg[1:]
            if any(x in cur_used for x in tail if x != pu):
                continue
            newseq += tail
            cur_used |= set(tail)

            if j == i:
                # pu -> de -> node_seq[i+1] ...
                seg = _dijkstra(instance, pu, cur_used - {pu}, de)
                if seg is None:
                    continue
                tail = seg[1:]
                if any(x in cur_used for x in tail if x != de):
                    continue
                newseq += tail
                cur_used |= set(tail)
                rs = node_seq[i + 1]
                seg = _dijkstra(instance, de, cur_used - {de, rs}, rs)
                if seg is None:
                    continue
                tail = seg[1:]
                if any(x in cur_used for x in tail if x != rs):
                    continue
                newseq += tail
                cur_used |= set(tail)
                newseq += node_seq[i + 2:]
            else:
                # pu -> node_seq[i+1] ... node_seq[j] -> de -> node_seq[j+1] ...
                rs = node_seq[i + 1]
                seg = _dijkstra(instance, pu, cur_used - {pu, rs}, rs)
                if seg is None:
                    continue
                tail = seg[1:]
                if any(x in cur_used for x in tail if x != rs):
                    continue
                newseq += tail
                cur_used |= set(tail)
                mid_stops = node_seq[i + 2:j + 1]
                if any(x in cur_used for x in mid_stops):
                    continue
                newseq += mid_stops
                cur_used |= set(mid_stops)
                seg = _dijkstra(instance, node_seq[j], cur_used - {node_seq[j]}, de)
                if seg is None:
                    continue
                tail = seg[1:]
                if any(x in cur_used for x in tail if x != de):
                    continue
                newseq += tail
                cur_used |= set(tail)
                rs2 = node_seq[j + 1]
                seg = _dijkstra(instance, de, cur_used - {de, rs2}, rs2)
                if seg is None:
                    continue
                tail = seg[1:]
                if any(x in cur_used for x in tail if x != rs2):
                    continue
                newseq += tail
                cur_used |= set(tail)
                newseq += node_seq[j + 2:]

            if len(set(newseq)) != len(newseq):
                continue
            arcs = [(newseq[t], newseq[t + 1]) for t in range(len(newseq) - 1)]
            if any(a not in arcdist for a in arcs):
                continue
            tot = sum(arcdist[a] for a in arcs)
            if best is None or tot < best[0]:
                best = (tot, newseq)
    return best[1] if best is not None else None


def try_add_one(
    instance: Instance, vehicle: Vehicle, node_seq: List[int],
    served: List[int], rid: int, n_candidates: int = 14,
) -> Optional[Tuple[List[int], List[Tuple[int, int]], Dict[Tuple[int, int], float]]]:
    """Incrementally splice ONE request rid into an existing elementary route
    (node_seq / served) without re-routing the rest. Returns
    (new_node_seq, arcs, charge) if the checker-clean insertion succeeds, else
    None. Used for reinsertion, where re-routing the whole cluster would be
    much slower."""
    arcdist = {(a.from_node, a.to_node): a.distance_km for a in instance.arcs}
    req = instance.request_by_id(rid)
    ns = _try_insert(instance, vehicle, node_seq, req.pickup_node,
                     req.delivery_node, arcdist, n_candidates)
    if ns is None:
        return None
    charge, ok = _charge_repair(instance, vehicle, ns, arcdist)
    if not ok:
        return None
    arcs = [(ns[t], ns[t + 1]) for t in range(len(ns) - 1)]
    if any(a not in arcdist for a in arcs):
        return None
    route = VehicleRoute(arcs=arcs, charge=charge, served=sorted(served + [rid]))
    rep = check_solution(instance, {vehicle.id: route})
    own = [v for v in rep.violations if "served by NO vehicle" not in v]
    if own:
        return None
    return ns, arcs, charge


def route_cluster_insertion(
    instance: Instance, vehicle: Vehicle, cluster: List[int],
    n_candidates: int = 14,
) -> Tuple[List[int], List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """Route as many of `cluster`'s requests as possible on `vehicle` via
    node-disjoint insertion, trying several seed orderings and keeping the
    most-served checker-clean result. Returns (served_ids, arcs, charge)."""
    if not cluster:
        return [], [], {}
    arcdist = {(a.from_node, a.to_node): a.distance_km for a in instance.arcs}

    def mid(rid: int) -> Tuple[float, float]:
        r = instance.request_by_id(rid)
        p = instance.nodes[r.pickup_node]
        d = instance.nodes[r.delivery_node]
        return ((p.x_km + d.x_km) / 2, (p.y_km + d.y_km) / 2)

    cx = sum(mid(r)[0] for r in cluster) / len(cluster)
    cy = sum(mid(r)[1] for r in cluster) / len(cluster)
    orderings = [
        sorted(cluster, key=lambda r: math.hypot(mid(r)[0] - cx, mid(r)[1] - cy)),
        sorted(cluster, key=lambda r: -math.hypot(mid(r)[0] - cx, mid(r)[1] - cy)),
        list(cluster),
    ]

    best_served: List[int] = []
    best_seq: List[int] = [vehicle.start_node, vehicle.end_node]
    for od in orderings:
        node_seq = [vehicle.start_node, vehicle.end_node]
        served: List[int] = []
        for r in od:
            req = instance.request_by_id(r)
            ns = _try_insert(instance, vehicle, node_seq, req.pickup_node,
                             req.delivery_node, arcdist, n_candidates)
            if ns is None:
                continue
            charge, ok = _charge_repair(instance, vehicle, ns, arcdist)
            if not ok:
                continue
            arcs = [(ns[t], ns[t + 1]) for t in range(len(ns) - 1)]
            if any(a not in arcdist for a in arcs):
                continue
            route = VehicleRoute(arcs=arcs, charge=charge, served=sorted(served + [r]))
            rep = check_solution(instance, {vehicle.id: route})
            own = [v for v in rep.violations if "served by NO vehicle" not in v]
            if not own:
                node_seq = ns
                served.append(r)
        if len(served) > len(best_served):
            best_served, best_seq = served, node_seq

    arcs = [(best_seq[t], best_seq[t + 1]) for t in range(len(best_seq) - 1)]
    if any(a not in arcdist for a in arcs):
        # nothing was ever inserted and the bare start->end arc does not exist:
        # emit a genuinely empty route rather than an invalid one.
        return [], [], {}
    charge, _ = _charge_repair(instance, vehicle, best_seq, arcdist)
    return sorted(best_served), arcs, (charge or {})
