from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.model.formulation import SubproblemVarIndex, build_vehicle_subproblem
from src.model.milp_ir import MILPProblem
from src.model.types import Instance, Request, Vehicle
from src.solve.solver_interface import SolverInterface

logger = logging.getLogger(__name__)

CutSet = Tuple[int, ...]  # sorted node ids forming a forbidden cycle set


@dataclass
class SubproblemResult:
    vehicle_id: str
    # dual_value: a valid lower bound on this subproblem's TRUE minimum.
    #
    # Two distinct claims must not be conflated here:
    #   (i) each cut-set round only adds valid inequalities, so the relaxed
    #       problem's minimum is non-increasing as the feasible region
    #       shrinks, and a round's PROVEN OPTIMUM is a lower bound on the
    #       true minimum.
    #   (ii) the INCUMBENT of a time-limited, unproven round is NOT a lower
    #       bound. For a minimisation problem it is an upper bound on that
    #       round's own minimum. Using objective_value when is_optimal is
    #       False would insert an upper bound where weak duality needs a
    #       lower one, letting the assembled L(mu) exceed the true optimum.
    #
    # This field therefore takes mip_dual_bound (a valid lower bound whether
    # or not the search finished) whenever available, and falls back to
    # objective_value ONLY when mip_dual_bound is None AND the round proved
    # optimal.
    dual_value: float
    # True iff dual_value was taken from a solve that proved optimality
    # (mip_dual_bound == objective_value in that case, up to solver
    # tolerance). False means dual_value came from an unproven B&B bound - a
    # weaker, but still VALID, lower bound. The outer Lagrangian loop
    # aggregates this per iteration so a dual assembled mostly from unproven
    # bounds is visibly flagged as weaker, not silently trusted the same as
    # one from proven optima.
    dual_is_proven_optimal: bool
    # primal_solution (arc/charge/soc/assignment values) is usable ONLY when
    # subtour_free is True. A solution with subtours must never become the primal
    # incumbent - src/verify/ would reject it and the certified gap would be
    # meaningless.
    subtour_free: bool
    # precedence_clean is True only when every served request's pickup
    # position precedes its delivery position on the walked s-e path. A
    # primal solution with precedence_clean=False must never be used as an
    # incumbent; the checker's precedence test would reject it.
    precedence_clean: bool
    solved_to_optimality: bool  # last round proved optimal (not time-limited)
    hit_round_cap: bool
    hit_wall_budget: bool
    rounds: int
    total_solve_time_s: float  # SUM over all separation rounds, not the last solve
    remaining_subtours: int
    remaining_precedence_violations: int = 0
    x_values: Dict[Tuple[int, int], float] = field(default_factory=dict)
    c_values: Dict[Tuple[int, int], float] = field(default_factory=dict)
    b_values: Dict[int, float] = field(default_factory=dict)
    q_values: Dict[int, float] = field(default_factory=dict)
    served_requests: Tuple[int, ...] = ()


def find_subtours(
    used_arcs: List[Tuple[int, int]]
) -> List[CutSet]:
    """Return node sets forming subtours (directed cycles) in the selected arcs.

    Correct definition, independent of any src/solve internals: a subtour is a
    strongly-connected component of the selected-arc digraph that contains a
    cycle (>= 2 nodes, or a single node with a self-loop). A simple start-to-end
    path is acyclic, so all its nodes are singleton SCCs with no self-loop and
    are never reported. Implemented with Tarjan's algorithm (iterative, no
    recursion-depth limit) so it scales to the full 602-node graph.
    """
    succ: Dict[int, List[int]] = {}
    self_loops: Set[int] = set()
    for i, j in used_arcs:
        if i == j:
            self_loops.add(i)
        succ.setdefault(i, []).append(j)
        succ.setdefault(j, [])

    index_counter = 0
    stack: List[int] = []
    on_stack: Set[int] = set()
    indices: Dict[int, int] = {}
    lowlink: Dict[int, int] = {}
    result: List[CutSet] = []

    for root in list(succ.keys()):
        if root in indices:
            continue
        work: List[Tuple[int, int]] = [(root, 0)]
        while work:
            node, pi = work[-1]
            if pi == 0:
                indices[node] = index_counter
                lowlink[node] = index_counter
                index_counter += 1
                stack.append(node)
                on_stack.add(node)
            recursed = False
            neighbors = succ.get(node, ())
            for k in range(pi, len(neighbors)):
                nxt = neighbors[k]
                if nxt not in indices:
                    work[-1] = (node, k + 1)
                    work.append((nxt, 0))
                    recursed = True
                    break
                elif nxt in on_stack:
                    lowlink[node] = min(lowlink[node], indices[nxt])
            if recursed:
                continue
            if lowlink[node] == indices[node]:
                comp: List[int] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == node:
                        break
                if len(comp) >= 2 or (len(comp) == 1 and comp[0] in self_loops):
                    result.append(tuple(sorted(comp)))
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
    return result


def walk_path(used_arcs: List[Tuple[int, int]], s_node: int) -> List[int]:
    """Walk the unique s-to-e simple path of a subtour-free arc set (flow
    conservation, degree <= 1, no cycles and start/end pinning leave exactly
    one path from s_node). Returns [s_node, ..., last reached node]. Only
    meaningful for an arc set already known to be subtour-free; it does not
    check for cycles itself.
    """
    succ: Dict[int, int] = {}
    for i, j in used_arcs:
        succ[i] = j
    path = [s_node]
    cur = s_node
    seen = {cur}
    while cur in succ:
        nxt = succ[cur]
        if nxt in seen:
            break  # defensive only - should not happen on a subtour-free set
        path.append(nxt)
        seen.add(nxt)
        cur = nxt
    return path


def find_precedence_violations(
    used_arcs: List[Tuple[int, int]],
    s_node: int,
    requests: Tuple[Request, ...],
    q_values: Dict[int, float],
) -> List[Tuple[int, List[Tuple[int, int]]]]:
    """Lazy precedence separation: for each served request (q_u > 0.5), check
    whether its pickup node's path position precedes its delivery node's. Returns a list of (request_id, prefix_arcs) for each
    violation, where prefix_arcs is the exact arc sequence from s_node up to
    and including the delivery node - the arc set the resulting cut forbids
    (conditional on that request's q_u).
    """
    path = walk_path(used_arcs, s_node)
    position = {node: idx for idx, node in enumerate(path)}
    violations: List[Tuple[int, List[Tuple[int, int]]]] = []
    for req in requests:
        if q_values.get(req.id, 0.0) <= 0.5:
            continue
        p_pos = position.get(req.pickup_node)
        d_pos = position.get(req.delivery_node)
        if p_pos is None or d_pos is None:
            # On a subtour-free path q_u = 1 forces both endpoints onto the
            # path. A missing endpoint is a different fault, not a precedence
            # violation, and is logged rather than treated as clean.
            logger.error(
                "request %d has q_u=1 but pickup_pos=%s delivery_pos=%s - "
                "not a precedence violation, but NOT clean either",
                req.id, p_pos, d_pos,
            )
            continue
        if d_pos < p_pos:
            prefix_arcs = [(path[k], path[k + 1]) for k in range(d_pos)]
            violations.append((req.id, prefix_arcs))
    return violations


def _apply_precedence_cuts(
    problem: MILPProblem,
    idx: SubproblemVarIndex,
    violations: List[Tuple[int, List[Tuple[int, int]]]],
) -> None:
    for req_id, prefix_arcs in violations:
        m = len(prefix_arcs)
        if m == 0:
            continue
        coeffs = {idx.x[a]: 1.0 for a in prefix_arcs}
        q_var = idx.q[req_id]
        coeffs[q_var] = coeffs.get(q_var, 0.0) + 1.0
        # sum(prefix arcs) + q_u <= m: at q_u=1, forbids exactly this m-arc
        # prefix; at q_u=0, trivially true (m binary terms can sum to at
        # most m) regardless of the prefix - no big-M needed.
        problem.add_constraint(
            f"precedence_cut_{req_id}_{problem.num_rows}", coeffs, -1e30, float(m)
        )


def _connectivity_cut_rhs(
    cut_set: CutSet,
    requests: Tuple[Request, ...],
    mandatory_requests: Set[int],
    idx: SubproblemVarIndex,
) -> Optional[Tuple[float, Dict[int, float]]]:
    """RHS side of the directed connectivity cut for component S=cut_set:
    sum_{(i,j): i not in S, j in S} x_ij >= r(S).

    Returns (const_rhs, extra_coeffs) such that the full row is
    (entering-arc sum) + extra_coeffs >= const_rhs, or None if S contains no
    request endpoint at all (nothing to anchor a cut to - skip).

    r(S) = 1 (const_rhs=1.0, extra_coeffs={}) if S contains the
    pickup/delivery node of a MANDATORY request - constraint (9) fixes that
    node's degree in every feasible solution, so external inflow to S is
    unconditionally required (an isolated S has zero entering arcs, which is
    then infeasible, exactly killing the isolated-island class rather than
    one member of it).

    r(S) = q_u (const_rhs=0.0, extra_coeffs={q_var: -1.0}, i.e.
    entering-sum - q_u >= 0) if S contains an OPTIONAL request's endpoint -
    conditional on that request actually being served in this solution, so a
    genuinely unvisited optional node is never forbidden. Only the first
    matching optional request found is used as the anchor; this is a valid
    (not necessarily maximal) lower bound either way.

    Cut form: Bianchessi, Mansini & Speranza (2018), eq. 12; Assuncao &
    Mateus (2019), eq. 33.
    """
    cs = set(cut_set)
    optional_q_var: Optional[int] = None
    for req in requests:
        if req.pickup_node in cs or req.delivery_node in cs:
            if req.id in mandatory_requests:
                return (1.0, {})
            if optional_q_var is None:
                optional_q_var = idx.q[req.id]
    if optional_q_var is not None:
        return (0.0, {optional_q_var: -1.0})
    return None


def _apply_connectivity_cuts(
    problem: MILPProblem,
    idx: SubproblemVarIndex,
    cuts: List[CutSet],
    requests: Tuple[Request, ...],
    mandatory_requests: Set[int],
) -> int:
    """Directed connectivity cuts. A DFJ cut sum_{i,j in S} x_ij <= |S|-1
    forbids a single node set, so a solver facing many similarly cheap
    isolated loops around the same forced-degree node rebuilds a different
    one each round. The connectivity cut forbids the whole class: an isolated
    component has zero entering arcs, which is < r(S) for any component that
    contains a served node, whatever arcs it uses inside. These cuts are added
    together with the DFJ cuts; both are valid.
    """
    added = 0
    for c in cuts:
        rhs_info = _connectivity_cut_rhs(c, requests, mandatory_requests, idx)
        if rhs_info is None:
            continue
        const_rhs, extra_coeffs = rhs_info
        cs = set(c)
        coeffs: Dict[int, float] = dict(extra_coeffs)
        for (i, j), var in idx.x.items():
            if i not in cs and j in cs:
                coeffs[var] = coeffs.get(var, 0.0) + 1.0
        if not coeffs:
            continue
        problem.add_constraint(
            f"connectivity_{min(cs)}_{len(cs)}_{problem.num_rows}",
            coeffs, const_rhs, 1e30,
        )
        added += 1
    return added


def solve_routing_subproblem(
    instance: Instance,
    vehicle: Vehicle,
    requests: Tuple[Request, ...],
    solver: SolverInterface,
    multipliers: Optional[Dict[int, float]] = None,
    cut_pool: Optional[List[CutSet]] = None,
    max_rounds: int = 50,
    wall_budget_s: float = 120.0,
    mandatory_requests: Optional[set] = None,
    use_precedence_separation: bool = False,
    use_gg_flow: bool = True,
    gg_steiner_demand: bool = True,
) -> SubproblemResult:
    """Solve one vehicle's routing+charging+SOC subproblem.

    use_gg_flow (default True): Gavish-Graves single-commodity flow makes
    connectivity structural - an isolated component containing any
    demand-positive node is flow-infeasible - so no separation loop is needed
    for connectivity. The round loop below then only runs a defensive
    find_subtours() check and, if requested, precedence separation. False
    falls back to cut-set separation.

    gg_steiner_demand (default True): the GG demand definition, which also
    fixes flow_cap (see the GG-flow block in src/model/formulation.py).
    True puts demand only at request pickup/delivery nodes with
    flow_cap = 2*len(requests), a much tighter LP relaxation. False puts
    demand at every visited node with flow_cap = n_nodes - 2. Both forbid
    isolated components containing a demand node; True also allows
    transit-only cycles, which cost distance and are never chosen.

    cut_pool: a fleet-wide list of previously discovered subtour node sets
    (only relevant when use_gg_flow=False). Since all vehicles share the
    same graph, start, end and battery (vehicles.csv), every such cut is
    valid for every vehicle and every multiplier vector, so the pool
    persists across vehicles and multiplier updates.

    use_precedence_separation: lazy precedence separation.
    When True, a subtour-free round is also checked for precedence
    violations (delivery visited before pickup); any violation adds a
    conditional cut and the loop continues instead of returning.
    """
    problem, idx = build_vehicle_subproblem(
        instance,
        vehicle,
        requests,
        multipliers=multipliers,
        include_mtz=False,
        include_soc=True,
        include_redundant_67=False,
        include_flow_conservation=True,
        include_gg_flow=use_gg_flow,
        gg_steiner_demand=gg_steiner_demand,
        mandatory_requests=mandatory_requests,
    )

    _add_reciprocal_pair_cuts(instance, idx, problem)

    mandatory_set: Set[int] = set(mandatory_requests) if mandatory_requests else set()
    pool = cut_pool if cut_pool is not None else []
    # The DFJ/connectivity cut machinery is redundant once GG flow is active,
    # so it is skipped in the normal case; find_subtours() below still runs
    # every round as a cheap defensive check regardless of use_gg_flow.
    if not use_gg_flow:
        _apply_cuts(problem, idx, pool)
        _apply_connectivity_cuts(problem, idx, pool, requests, mandatory_set)

    total_time = 0.0
    rounds = 0
    prev_obj: Optional[float] = None
    prev_optimal = False
    last_result = None
    subtours: List[CutSet] = []

    for rnd in range(max_rounds):
        result = solver.solve(problem)
        total_time += result.solve_time_s
        rounds = rnd + 1
        last_result = result

        if not result.col_values:
            # Infeasible or no incumbent. dual_value is the solver's bound if
            # available (still a valid LB); no primal.
            logger.warning(
                "%s round %d produced no solution (status=%s) - infeasibility "
                "or no incumbent; reporting as not-subtour-free",
                vehicle.id, rnd, result.status,
            )
            dual = result.mip_dual_bound if result.mip_dual_bound is not None else float("-inf")
            if dual == float("-inf"):
                logger.error(
                    "%s: no col_values AND no mip_dual_bound - genuinely no "
                    "valid lower bound available (status=%s); L(mu) built from "
                    "this iteration is NOT trustworthy for this vehicle",
                    vehicle.id, result.status,
                )
            return SubproblemResult(
                vehicle_id=vehicle.id,
                dual_value=dual,
                dual_is_proven_optimal=False,
                subtour_free=False,
                precedence_clean=False,
                solved_to_optimality=result.is_optimal,
                hit_round_cap=False,
                hit_wall_budget=False,
                rounds=rounds,
                total_solve_time_s=total_time,
                remaining_subtours=-1,
            )

        obj = result.objective_value
        # Monotonicity: each added cut shrinks the feasible region, so the
        # optimal objective is non-decreasing across rounds. Only compare two
        # rounds that both proved optimality: a time-limited incumbent can move
        # either way. A decrease between two optimal solves means a cut removed
        # integer-feasible points, which is an error.
        if prev_obj is not None and prev_optimal and result.is_optimal:
            if obj < prev_obj - 1e-6:
                raise AssertionError(
                    f"{vehicle.id} round {rnd}: monotonicity violated between "
                    f"two optimal solves ({prev_obj:.6f} -> {obj:.6f}); a cut is "
                    f"invalid (removed integer-feasible solutions)."
                )
        elif prev_obj is not None and not (prev_optimal and result.is_optimal):
            logger.info(
                "%s round %d: monotonicity check SKIPPED (a round was "
                "time-limited, not proven optimal): %.4f -> %.4f",
                vehicle.id, rnd, prev_obj, obj,
            )
        prev_obj = obj
        prev_optimal = result.is_optimal

        used = [k for k, v in idx.x.items() if result.col_values[v] > 0.5]
        # find_subtours() runs every round as a cheap defensive check. Under GG
        # flow it should always return nothing; a non-empty result is an error
        # and is logged.
        subtours = find_subtours(used)
        if subtours and use_gg_flow:
            logger.error(
                "%s round %d: %d subtours found WITH use_gg_flow=True - this "
                "should be structurally impossible. Investigate before trusting this "
                "result; falling back to DFJ+connectivity cuts defensively.",
                vehicle.id, rnd, len(subtours),
            )

        logger.info(
            "%s round %d: %.3fs obj=%.4f optimal=%s subtours=%d arcs=%d "
            "cutpool=%d cum=%.3fs",
            vehicle.id, rnd, result.solve_time_s, obj, result.is_optimal,
            len(subtours), len(used), len(pool), total_time,
        )

        if not subtours:
            if not use_precedence_separation:
                return _finalize(vehicle, idx, result, total_time, rounds,
                                 subtour_free=True, precedence_clean=True,
                                 hit_round_cap=False, hit_wall_budget=False,
                                 requests=requests)

            q_now = {rid: result.col_values[v] for rid, v in idx.q.items()}
            violations = find_precedence_violations(used, vehicle.start_node, requests, q_now)
            logger.info(
                "%s round %d: subtour-free, precedence violations=%d",
                vehicle.id, rnd, len(violations),
            )
            if not violations:
                return _finalize(vehicle, idx, result, total_time, rounds,
                                 subtour_free=True, precedence_clean=True,
                                 hit_round_cap=False, hit_wall_budget=False,
                                 requests=requests)
            _apply_precedence_cuts(problem, idx, violations)
            if total_time >= wall_budget_s:
                logger.warning(
                    "%s hit wall budget %.1fs after %d rounds; subtour-free but "
                    "%d precedence violations remain - primal NOT usable",
                    vehicle.id, wall_budget_s, rounds, len(violations),
                )
                return _finalize(vehicle, idx, result, total_time, rounds,
                                 subtour_free=True, precedence_clean=False,
                                 hit_round_cap=False, hit_wall_budget=True,
                                 requests=requests, remaining=0,
                                 remaining_precedence=len(violations))
            continue  # re-solve with the new precedence cuts in place

        # add new cuts to the fleet-wide pool and this problem
        new_cuts = [c for c in subtours if c not in set(pool)]
        for c in new_cuts:
            pool.append(c)
        _apply_cuts(problem, idx, subtours)  # apply this round's found cuts (DFJ, kept)
        n_conn = _apply_connectivity_cuts(problem, idx, subtours, requests, mandatory_set)
        logger.info(
            "%s round %d: connectivity cuts emitted=%d (of %d components)",
            vehicle.id, rnd, n_conn, len(subtours),
        )

        if total_time >= wall_budget_s:
            logger.warning(
                "%s hit wall budget %.1fs after %d rounds; %d subtours remain - "
                "returning dual bound only, primal NOT usable",
                vehicle.id, wall_budget_s, rounds, len(subtours),
            )
            return _finalize(vehicle, idx, result, total_time, rounds,
                             subtour_free=False, precedence_clean=False,
                             hit_round_cap=False, hit_wall_budget=True,
                             requests=requests, remaining=len(subtours))

    logger.warning(
        "%s hit round cap %d; %d subtours remain - dual bound only, primal NOT usable",
        vehicle.id, max_rounds, len(subtours),
    )
    return _finalize(vehicle, idx, last_result, total_time, rounds,
                     subtour_free=False, precedence_clean=False,
                     hit_round_cap=True, hit_wall_budget=False,
                     requests=requests, remaining=len(subtours))


def _add_reciprocal_pair_cuts(
    instance: Instance, idx: SubproblemVarIndex, problem: MILPProblem
) -> int:
    """Static cut x_ij + x_ji <= 1 for every reciprocal arc pair (both (i,j)
    and (j,i) exist). Valid unconditionally - a 2-node subtour uses both
    arcs of exactly such a pair, so this is the subtour cut for |C| = 2,
    added upfront instead of being found by separation. It costs O(|A|/2)
    rows and does not remove longer subtours.
    """
    added = 0
    seen: Set[Tuple[int, int]] = set()
    for (i, j) in idx.x:
        if i >= j:
            continue
        if (j, i) not in idx.x:
            continue
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        problem.add_constraint(
            f"reciprocal_{i}_{j}", {idx.x[(i, j)]: 1.0, idx.x[(j, i)]: 1.0}, -1e30, 1.0
        )
        added += 1
    return added


def _apply_cuts(problem: MILPProblem, idx: SubproblemVarIndex, cuts: List[CutSet]) -> None:
    for c in cuts:
        cs = set(c)
        coeffs = {idx.x[(i, j)]: 1.0 for (i, j) in idx.x if i in cs and j in cs}
        if coeffs:
            problem.add_constraint(
                f"cutset_{min(cs)}_{len(cs)}_{problem.num_rows}",
                coeffs, -1e30, float(len(cs) - 1),
            )


def _finalize(
    vehicle: Vehicle,
    idx: SubproblemVarIndex,
    result,
    total_time: float,
    rounds: int,
    subtour_free: bool,
    hit_round_cap: bool,
    hit_wall_budget: bool,
    requests: Tuple[Request, ...],
    remaining: int = 0,
    precedence_clean: bool = True,
    remaining_precedence: int = 0,
) -> SubproblemResult:
    cv = result.col_values
    x_values = {k: cv[v] for k, v in idx.x.items() if cv[v] > 0.5}
    c_values = {k: cv[v] for k, v in idx.c.items() if cv[v] > 1e-9}
    b_values = {n: cv[v] for n, v in idx.B.items()}
    q_values = {rid: cv[v] for rid, v in idx.q.items()}
    served = tuple(sorted(rid for rid, val in q_values.items() if val > 0.5))

    # dual_value (see SubproblemResult.dual_value): use the B&B bound whenever
    # the solver provides one; it is valid whether or not the search finished.
    # Fall back to the incumbent objective only when no bound is available and
    # the round proved optimal, in which case the incumbent is the optimum.
    if result.mip_dual_bound is not None:
        dual_value = result.mip_dual_bound
        dual_is_proven_optimal = result.is_optimal
    elif result.is_optimal:
        dual_value = result.objective_value
        dual_is_proven_optimal = True
    else:
        dual_value = float("-inf")
        dual_is_proven_optimal = False
        logger.error(
            "%s: no mip_dual_bound and round not proven optimal - no valid "
            "lower bound available; returning -inf so this is visible rather "
            "than silently using the incumbent as if it were one",
            vehicle.id,
        )

    return SubproblemResult(
        vehicle_id=vehicle.id,
        dual_value=dual_value,
        dual_is_proven_optimal=dual_is_proven_optimal,
        subtour_free=subtour_free,
        precedence_clean=precedence_clean,
        solved_to_optimality=result.is_optimal,
        hit_round_cap=hit_round_cap,
        hit_wall_budget=hit_wall_budget,
        rounds=rounds,
        total_solve_time_s=total_time,
        remaining_subtours=remaining,
        remaining_precedence_violations=remaining_precedence,
        x_values=x_values,
        c_values=c_values,
        b_values=b_values,
        q_values=q_values,
        served_requests=served,
    )
