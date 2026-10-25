from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from src.model.types import Instance, Vehicle
from src.solve.subproblem import SubproblemResult, find_subtours
from src.verify.checker import VehicleRoute, check_solution

# ESPPRC-style labeling solver for the per-vehicle routing + charging + SOC
# subproblem, as an alternative to the MILP path in subproblem.py.
#
# Labeling handles free-choice request selection natively: a request is served
# only if some label's walk visits its pickup node (see solve_labeling below),
# so no binary selection variables are needed and there is no branch-and-bound
# blow-up as the candidate set grows.
#
# Every route this module returns is checked with check_solution before being
# treated as usable; the subtour_free and precedence_clean fields reflect that
# check, exactly as the MILP path's subtour_free reflects find_subtours().

CHARGE_LEVELS = (0.0, 5.0, 15.0, 20.0)


@dataclass
class Label:
    node: int
    cost: float
    soc: float
    onboard: FrozenSet[int]
    delivered: FrozenSet[int]
    parent: Optional["Label"]
    via_arc: Optional[Tuple[int, int]]
    via_charge: float
    visited: FrozenSet[int]  # every node this walk has stepped onto, including the start node


# ELEMENTARITY. Two mechanisms together guarantee that every route returned by
# this module is elementary (no repeated node):
#
#   1. Generation-time guard: a label may never extend onto a node already in
#      its own walk (the `j in lab.visited` check in solve_labeling's expansion
#      loop). A non-elementary label is therefore never created.
#
#   2. Subset dominance in _dominates below. This is required for correctness,
#      not a performance tweak - see the rule stated there.
def _dominates(a: Label, b: Label) -> bool:
    # DOMINANCE RULE FOR ELEMENTARY PATHS.
    #
    # `a` dominates `b` only when all four conditions hold:
    #     identical (onboard, delivered) state,
    #     a.cost <= b.cost,
    #     a.soc  >= b.soc,
    #     a.visited is a SUBSET of b.visited.
    #
    # The visited-subset condition is mandatory. Individual elementarity of a
    # label says nothing about which FUTURE extensions remain open to it: if
    # `a` is cheaper but has already consumed a node that `b` left free, then
    # on a sparse graph the only completion may run through exactly that node,
    # which `a` can no longer use. Discarding `b` in that situation can remove
    # the only label able to complete, so the search exhausts with no route and
    # reports a servable request as infeasible.
    #
    # Requiring a.visited subset-of b.visited means `a` may only dominate `b`
    # when every option still open to `b` is also still open to `a`. Do not
    # weaken this condition.
    if a.onboard != b.onboard or a.delivered != b.delivered:
        return False
    if a.cost > b.cost + 1e-9:
        return False
    if a.soc < b.soc - 1e-9:
        return False
    if not a.visited <= b.visited:
        return False
    return True


def _dist_to_target(instance: Instance, target: int) -> Dict[int, float]:
    dist = {target: 0.0}
    pq: List[Tuple[float, int]] = [(0.0, target)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist.get(node, float("inf")) + 1e-9:
            continue
        for arc in instance.in_arcs.get(node, ()):
            nd = d + arc.distance_km
            if nd < dist.get(arc.from_node, float("inf")) - 1e-9:
                dist[arc.from_node] = nd
                heapq.heappush(pq, (nd, arc.from_node))
    return dist


def solve_labeling(
    instance: Instance,
    vehicle: Vehicle,
    request_ids: Tuple[int, ...],
    mu: Optional[Dict[int, float]] = None,
    time_budget_s: float = 60.0,
    mandatory_ids: Optional[FrozenSet[int]] = None,
    label_budget: Optional[int] = None,
    top_k: int = 1,
) -> Tuple[Optional[Label], dict]:
    """Free-choice ESPPRC-with-recharge labeling solve.

    Every id in request_ids is an OPTIONAL pickup, not mandatory - a request
    is served only if some label's walk visits its pickup node and then its
    delivery node (pickup-before-delivery precedence, tracked through the
    onboard/delivered sets below), as in the MILP path with
    mandatory_requests=None. mu, if given, subtracts mu[rid] from
    the walk's cost at the moment a request's delivery node is reached -
    the same Lagrangian price term the MILP's objective coefficients carry
    (see build_vehicle_subproblem's -mu_u*q_u term).

    mandatory_ids, when given, is enforced IN THE SEARCH (a label is only
    "complete" at e_node if mandatory_ids subseteq delivered) - a post-hoc
    filter cannot force this, since the free-choice optimum has no
    incentive to visit an unprofitable pickup on its own.

    label_budget, when given, stops the search after that many labels are
    CREATED (not wall-clock) and returns the best incumbent found so far,
    marked LABEL_BUDGET_EXCEEDED (unproven). Label count is deterministic
    given the same instance and mu, unlike wall-clock time, so this mode is
    reproducible. The binding cost is the number of reachable
    (onboard, delivered) state combinations when many requests are
    simultaneously optional.

    top_k (default 1): return the top_k distinct-cost complete labels found,
    not just the single best. Pruning uses the K-th best cost found so far
    rather than the single best, so paths that would become the
    2nd/3rd/...-best are not cut off.
    """
    mu = mu or {}
    mandatory_ids = mandatory_ids or frozenset()
    reqs = {r: instance.request_by_id(r) for r in request_ids}
    pickup_of = {r.pickup_node: rid for rid, r in reqs.items()}
    delivery_of = {r.delivery_node: rid for rid, r in reqs.items()}
    all_ids = frozenset(request_ids)

    consumption = vehicle.max_energy_consumption_kWh_per_km or instance.energy.consumption_kWh_per_km
    eps1 = instance.objective.epsilon1
    eps2 = instance.objective.epsilon2
    s_node, e_node = vehicle.start_node, vehicle.end_node

    dist_to_e = _dist_to_target(instance, e_node)

    # Admissible A* completion bound. It must account for the best-case
    # remaining mu subsidy, not just remaining distance: an estimate using
    # eps1*dist_to_e alone is non-negative and therefore over-estimates the
    # cost of finishing a path that still has subsidised requests to deliver,
    # which prunes those paths even when completing them would be cheaper.
    # Subtracting the maximum remaining subsidy (each not-yet-delivered
    # request's mu when beneficial, else 0) keeps the bound admissible.
    def heuristic(node: int, delivered: FrozenSet[int]) -> float:
        d = dist_to_e.get(node)
        if d is None:
            return float("inf")
        remaining = sum(min(0.0, -mu.get(rid, 0.0)) for rid in request_ids if rid not in delivered)
        return eps1 * d + remaining

    start = Label(node=s_node, cost=0.0, soc=vehicle.initial_soc_kWh,
                  onboard=frozenset(), delivered=frozenset(),
                  parent=None, via_arc=None, via_charge=0.0,
                  visited=frozenset({s_node}))

    buckets: Dict[Tuple[int, FrozenSet[int], FrozenSet[int]], List[Label]] = {
        (s_node, frozenset(), frozenset()): [start]
    }
    counter = 0
    pq: List[Tuple[float, int, Label]] = [(heuristic(s_node, frozenset()), 0, start)]
    # top_completions: sorted ascending by cost, capped at top_k distinct
    # (delivered set, cost) completions - top_completions[0] is best_complete.
    top_completions: List[Label] = []
    labels_created = 1
    t0 = time.time()

    def bound_cost() -> float:
        # the K-th best cost found so far (or +inf if fewer than K found) -
        # this is what gates further search, not just the single best, so
        # candidates for the 2nd/3rd/...-best slot are not pruned away.
        if len(top_completions) < top_k:
            return float("inf")
        return top_completions[-1].cost

    # A complete label needs the vehicle back at e_node - delivered can be
    # any SUBSET of all_ids (free choice), never required to equal all_ids.
    while pq:
        if time.time() - t0 > time_budget_s:
            best_complete = top_completions[0] if top_completions else None
            return best_complete, {
                "status": "TIME_BUDGET_EXCEEDED" if best_complete is None else "TIME_BUDGET_EXCEEDED_WITH_INCUMBENT",
                "labels_created": labels_created, "wall": time.time() - t0,
                "top_completions": list(top_completions),
            }
        if label_budget is not None and labels_created >= label_budget:
            best_complete = top_completions[0] if top_completions else None
            return best_complete, {
                "status": "LABEL_BUDGET_EXCEEDED" if best_complete is None else "LABEL_BUDGET_EXCEEDED_WITH_INCUMBENT",
                "labels_created": labels_created, "wall": time.time() - t0,
                "top_completions": list(top_completions),
            }
        priority, _, lab = heapq.heappop(pq)
        key = (lab.node, lab.onboard, lab.delivered)
        if key not in buckets or lab not in buckets[key]:
            continue
        if lab.cost + heuristic(lab.node, lab.delivered) >= bound_cost() - 1e-9:
            continue

        if lab.node == e_node and not lab.onboard and mandatory_ids.issubset(lab.delivered):
            already = any(abs(t.cost - lab.cost) < 1e-9 and t.delivered == lab.delivered for t in top_completions)
            # Admissions are capped per DISTINCT delivered set, not only by
            # total count (top_k). Once mu stabilises, the K cheapest
            # completions tend to be K cost/charging variants of a single
            # served set - the same requests with slightly different charging
            # amounts. Since bound_cost() (the K-th best cost) is the global
            # pruning bound, letting one served set monopolise the K slots
            # prunes partial paths toward structurally different served sets
            # before they ever complete, even when those are the only routes
            # covering a given request. Capping per served set reserves room
            # for structurally distinct routes to survive into
            # top_completions.
            MAX_PER_SERVED_SET = 3
            same_served_set_count = sum(1 for t in top_completions if t.delivered == lab.delivered)
            if not already and same_served_set_count < MAX_PER_SERVED_SET:
                top_completions.append(lab)
                top_completions.sort(key=lambda l: l.cost)
                del top_completions[top_k:]
            continue

        for arc in instance.out_arcs.get(lab.node, ()):
            j = arc.to_node
            if j == s_node:
                continue
            # Elementarity guard (see the note above the Label class):
            # never step onto a node this walk has already visited. e_node is
            # deliberately not exempted - a complete label reaches e_node
            # exactly once, when the walk ends (checked above, before this
            # expansion loop), so e_node is never a valid mid-walk target.
            if j in lab.visited:
                continue
            if j in delivery_of and delivery_of[j] not in lab.onboard:
                continue
            e_ij = consumption * arc.distance_km
            eta = 1.0 if instance.charging.arc_enabled(lab.node, j) else 0.0
            site = instance.charging.sites_by_node.get(lab.node)
            cap = site.large_kWh if (site and site.large_kWh is not None and eta > 0) else 0.0
            price = instance.prices.base_price_per_node.get(lab.node, 0.0)
            charge_options = [c for c in CHARGE_LEVELS if c <= cap + 1e-9] if eta > 0 else [0.0]

            for c in charge_options:
                new_soc = lab.soc - e_ij + eta * c
                if new_soc < -1e-9 or new_soc > vehicle.battery_capacity_kWh + 1e-9:
                    continue
                new_soc = max(0.0, min(vehicle.battery_capacity_kWh, new_soc))
                new_cost = lab.cost + eps1 * arc.distance_km + eps2 * price * eta * c

                new_onboard, new_delivered = lab.onboard, lab.delivered
                if j in pickup_of:
                    rid = pickup_of[j]
                    if rid in lab.onboard or rid in lab.delivered:
                        continue
                    new_onboard = lab.onboard | {rid}
                if j in delivery_of:
                    rid = delivery_of[j]
                    new_onboard = new_onboard - {rid}
                    new_delivered = lab.delivered | {rid}
                    new_cost -= mu.get(rid, 0.0)

                # Prune using the POST-transition delivered set, not
                # lab.delivered: the heuristic's remaining-subsidy bound must
                # reflect what is still unclaimed AFTER this arc, otherwise it
                # over-prunes a path the instant it delivers a request.
                if new_cost + heuristic(j, new_delivered) >= bound_cost() - 1e-9:
                    continue

                new_lab = Label(node=j, cost=new_cost, soc=new_soc,
                                 onboard=new_onboard, delivered=new_delivered,
                                 parent=lab, via_arc=(lab.node, j), via_charge=c,
                                 visited=lab.visited | {j})
                labels_created += 1

                bkey = (j, new_onboard, new_delivered)
                bucket = buckets.setdefault(bkey, [])
                if any(_dominates(existing, new_lab) for existing in bucket):
                    continue
                bucket[:] = [b for b in bucket if not _dominates(new_lab, b)]
                bucket.append(new_lab)
                counter += 1
                heapq.heappush(pq, (new_cost + heuristic(j, new_delivered), counter, new_lab))

    best_complete = top_completions[0] if top_completions else None
    return best_complete, {
        "status": "OPTIMAL" if best_complete is not None else "INFEASIBLE",
        "labels_created": labels_created, "wall": time.time() - t0,
        "top_completions": list(top_completions),
    }


def reconstruct(label: Label) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    arcs: List[Tuple[int, int]] = []
    charge: Dict[Tuple[int, int], float] = {}
    cur = label
    while cur.parent is not None:
        arcs.append(cur.via_arc)  # type: ignore[arg-type]
        if cur.via_charge > 1e-9:
            charge[cur.via_arc] = cur.via_charge  # type: ignore[index]
        cur = cur.parent
    arcs.reverse()
    return arcs, charge


def solve_routing_subproblem_labeling(
    instance: Instance,
    vehicle: Vehicle,
    requests: Tuple,
    multipliers: Optional[Dict[int, float]] = None,
    mandatory_requests: Optional[Set[int]] = None,
    time_budget_s: float = 60.0,
) -> SubproblemResult:
    """SubproblemResult-compatible wrapper around solve_labeling, usable in
    place of solve_routing_subproblem (src/solve/subproblem.py).

    mandatory_requests, when given, is enforced IN THE SEARCH (see
    solve_labeling's mandatory_ids) - a completed label must have delivered
    every id in this set, not filtered after an unconstrained search (which
    would almost always reject, since the free-choice optimum has no reason
    to visit an unprofitable pickup on its own).
    """
    request_ids = tuple(r.id for r in requests)
    mandatory_ids = frozenset(mandatory_requests) if mandatory_requests else frozenset()
    best, stats = solve_labeling(
        instance, vehicle, request_ids, mu=multipliers, time_budget_s=time_budget_s,
        mandatory_ids=mandatory_ids,
    )

    if best is None:
        return SubproblemResult(
            vehicle_id=vehicle.id, dual_value=float("-inf"), dual_is_proven_optimal=False,
            subtour_free=False, precedence_clean=False, solved_to_optimality=False,
            hit_round_cap=False, hit_wall_budget=(stats["status"] == "TIME_BUDGET_EXCEEDED"),
            rounds=1, total_solve_time_s=stats["wall"], remaining_subtours=-1,
        )

    arcs, charge = reconstruct(best)
    proven = stats["status"] == "OPTIMAL"

    # Structural check with the same Tarjan-SCC test the MILP path uses
    # (find_subtours), so both paths are held to the same standard.
    subtours = find_subtours(arcs)
    subtour_free = len(subtours) == 0

    served = tuple(sorted(best.delivered))
    x_values = {a: 1.0 for a in arcs}
    c_values = dict(charge)

    precedence_clean = True
    if subtour_free:
        route = VehicleRoute(arcs=arcs, charge=charge, served=list(served))
        rep = check_solution(instance, {vehicle.id: route})
        precedence_clean = not any("precedence" in v.lower() for v in rep.violations)

    q_values = {rid: (1.0 if rid in best.delivered else 0.0) for rid in request_ids}

    return SubproblemResult(
        vehicle_id=vehicle.id,
        dual_value=best.cost,  # exact labeling optimum IS a valid (in fact tight, when proven) lower bound
        dual_is_proven_optimal=proven,
        subtour_free=subtour_free,
        precedence_clean=precedence_clean,
        solved_to_optimality=proven,
        hit_round_cap=False,
        hit_wall_budget=(not proven),
        rounds=1,
        total_solve_time_s=stats["wall"],
        remaining_subtours=len(subtours),
        remaining_precedence_violations=0 if precedence_clean else 1,
        x_values=x_values,
        c_values=c_values,
        q_values=q_values,
        served_requests=served,
    )
