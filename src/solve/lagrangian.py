from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.model.types import Instance
from src.solve.solver_interface import HighsSolver, SolverInterface
from src.solve.subproblem import CutSet, SubproblemResult, solve_routing_subproblem

logger = logging.getLogger(__name__)


@dataclass
class DualIterationRecord:
    iteration: int
    dual_value: float          # L(mu): valid lower bound on the original optimum
    best_dual_so_far: float
    best_primal_so_far: float  # +inf until a feasible primal is found
    certified_gap: Optional[float]  # (primal - dual)/|primal|, None if no primal yet
    subgradient_norm: float
    step_size: float
    num_assignment_violations: int  # count of requests not served exactly once
    all_subproblems_subtour_free: bool
    all_subproblems_precedence_clean: bool
    num_duals_from_proven_optimum: int  # subproblems whose dual_value is a proven optimum
    num_duals_from_unproven_bound: int  # a dual assembled mostly from these is weaker
    subproblem_time_s: float   # summed over all vehicles this iteration
    multiplier_snapshot: Dict[int, float] = field(default_factory=dict)


@dataclass
class LagrangianResult:
    best_dual: float
    best_primal: float
    certified_gap: Optional[float]
    history: List[DualIterationRecord]
    final_multipliers: Dict[int, float]
    cut_pool_size: int
    best_primal_solution: Optional[Dict[str, SubproblemResult]] = None
    duals_all_zero: bool = False


def _lagrangian_lower_bound(
    subproblem_results: Dict[str, SubproblemResult], multipliers: Dict[int, float]
) -> float:
    # L(mu) = sum_u mu_u + sum_k min_x[ cost_k - sum_u mu_u q_u^k ]
    # Each subproblem dual_value is a valid lower bound on that subproblem's
    # minimum: HiGHS's mip_dual_bound when available, else the proven
    # incumbent, never an unproven incumbent. A subproblem with no bound
    # returns -inf, which makes L(mu) = -inf for that iteration.
    return sum(multipliers.values()) + sum(r.dual_value for r in subproblem_results.values())


def _fleet_homogeneity_reason(instance: Instance) -> Optional[str]:
    """Returns None if the fleet is homogeneous enough for fleet-wide cut
    sharing to be valid: a cut derived from one vehicle's feasible set is
    valid for another vehicle only if both have the same start, end,
    battery, initial SOC and consumption rate. Otherwise returns a
    human-readable reason for the mismatch.
    """
    vehicles = instance.vehicles
    if len(vehicles) <= 1:
        return None
    ref = vehicles[0]

    def rate(v):
        r = v.max_energy_consumption_kWh_per_km
        return r if r is not None else instance.energy.consumption_kWh_per_km

    ref_rate = rate(ref)
    for v in vehicles[1:]:
        if v.start_node != ref.start_node:
            return f"vehicle {v.id} start_node={v.start_node} != {ref.id} start_node={ref.start_node}"
        if v.end_node != ref.end_node:
            return f"vehicle {v.id} end_node={v.end_node} != {ref.id} end_node={ref.end_node}"
        if v.battery_capacity_kWh != ref.battery_capacity_kWh:
            return (f"vehicle {v.id} battery_capacity_kWh={v.battery_capacity_kWh} != "
                    f"{ref.id} battery_capacity_kWh={ref.battery_capacity_kWh}")
        if v.initial_soc_kWh != ref.initial_soc_kWh:
            return (f"vehicle {v.id} initial_soc_kWh={v.initial_soc_kWh} != "
                    f"{ref.id} initial_soc_kWh={ref.initial_soc_kWh}")
        if rate(v) != ref_rate:
            return f"vehicle {v.id} consumption_rate={rate(v)} != {ref.id} consumption_rate={ref_rate}"
    # The arc set (instance.arcs) is shared by every vehicle, so it needs no
    # per-vehicle check.
    return None


def _assignment_counts(
    subproblem_results: Dict[str, SubproblemResult], request_ids: List[int]
) -> Dict[int, int]:
    counts = {u: 0 for u in request_ids}
    for r in subproblem_results.values():
        for u in r.served_requests:
            counts[u] = counts.get(u, 0) + 1
    return counts


def solve_lagrangian(
    instance: Instance,
    solver: Optional[SolverInterface] = None,
    max_iterations: int = 40,
    subproblem_max_rounds: int = 50,
    subproblem_wall_budget_s: float = 120.0,
    initial_step: float = 2.0,
    parallel: bool = False,
    repair_fn=None,
) -> LagrangianResult:
    """Lagrangian relaxation of the coupling constraint (10): sum_k q_u^k = 1.

    The relaxed problem separates into K per-vehicle routing subproblems
    (solve_routing_subproblem). Subgradient ascent on the multipliers mu_u.
    L(mu) is a valid lower bound each iteration (weak duality); the best feasible
    primal comes from repair_fn. Certified gap = (primal - dual)/|primal|.
    """
    if solver is None:
        solver = HighsSolver(time_limit_s=10.0, mip_rel_gap=0.0)

    request_ids = [r.id for r in instance.requests]
    multipliers: Dict[int, float] = {u: 0.0 for u in request_ids}

    hetero_reason = _fleet_homogeneity_reason(instance)
    per_vehicle_pools: Optional[Dict[str, List[CutSet]]] = None
    fleet_cut_pool: List[CutSet] = []
    if hetero_reason is not None:
        per_vehicle_pools = {v.id: [] for v in instance.vehicles}
        logger.warning(
            "cut pool: fleet-wide sharing DISABLED, falling back to "
            "per-vehicle pools - fleet is heterogeneous (%s); a cut valid "
            "for one vehicle's feasible region is not guaranteed valid for "
            "another's", hetero_reason,
        )
    else:
        logger.info("cut pool: fleet-wide sharing enabled (fleet verified homogeneous)")

    best_dual = float("-inf")
    best_primal = float("inf")
    best_primal_solution: Optional[Dict[str, SubproblemResult]] = None
    history: List[DualIterationRecord] = []

    for it in range(max_iterations):
        t0 = time.time()
        sub_results = _solve_all_subproblems(
            instance, solver, multipliers, fleet_cut_pool, per_vehicle_pools,
            subproblem_max_rounds, subproblem_wall_budget_s, parallel,
        )
        sub_time = sum(r.total_solve_time_s for r in sub_results.values())

        dual_value = _lagrangian_lower_bound(sub_results, multipliers)
        best_dual = max(best_dual, dual_value)

        counts = _assignment_counts(sub_results, request_ids)
        subgradient = {u: 1.0 - counts[u] for u in request_ids}
        sg_norm = sum(g * g for g in subgradient.values()) ** 0.5
        num_violations = sum(1 for u in request_ids if counts[u] != 1)
        all_subtour_free = all(r.subtour_free for r in sub_results.values())
        all_precedence_clean = all(r.precedence_clean for r in sub_results.values())
        all_clean = all_subtour_free and all_precedence_clean
        n_proven = sum(1 for r in sub_results.values() if r.dual_is_proven_optimal)
        n_unproven = len(sub_results) - n_proven
        if n_unproven > 0:
            logger.info(
                "iter %d: dual assembled from %d/%d proven-optimal subproblems, "
                "%d from unproven B&B bounds (still valid, but weaker)",
                it, n_proven, len(sub_results), n_unproven,
            )

        # primal repair (optional; produces a feasible original-model solution)
        primal_value = float("inf")
        if repair_fn is not None:
            repaired = repair_fn(instance, sub_results, solver)
            if repaired is not None:
                primal_value = repaired["objective"]
                if primal_value < best_primal:
                    best_primal = primal_value
                    best_primal_solution = repaired["solution"]

        cert_gap = None
        if best_primal < float("inf") and best_dual > float("-inf") and abs(best_primal) > 1e-12:
            cert_gap = (best_primal - best_dual) / abs(best_primal)

        # Subgradient step: Polyak-style when a primal exists, else a
        # diminishing 1/(t+1) step scaled by a fixed reference, the magnitude
        # of the iteration-0 (mu = 0) dual. The step is never scaled by the
        # current dual_value, which would feed back into divergence.
        first_dual_magnitude = max(abs(history[0].dual_value), 1.0) if history else max(abs(dual_value), 1.0)
        if best_primal < float("inf") and sg_norm > 1e-9:
            step = (best_primal - dual_value) / (sg_norm * sg_norm)
            step = max(step, 1e-6)
        else:
            step = (initial_step * first_dual_magnitude) / (sg_norm * sg_norm * (it + 1))

        for u in request_ids:
            multipliers[u] += step * subgradient[u]

        rec = DualIterationRecord(
            iteration=it,
            dual_value=dual_value,
            best_dual_so_far=best_dual,
            best_primal_so_far=best_primal,
            certified_gap=cert_gap,
            subgradient_norm=sg_norm,
            step_size=step,
            num_assignment_violations=num_violations,
            all_subproblems_subtour_free=all_subtour_free,
            all_subproblems_precedence_clean=all_precedence_clean,
            num_duals_from_proven_optimum=n_proven,
            num_duals_from_unproven_bound=n_unproven,
            subproblem_time_s=sub_time,
            multiplier_snapshot=dict(multipliers),
        )
        history.append(rec)
        pool_size = (
            len(fleet_cut_pool) if per_vehicle_pools is None
            else sum(len(p) for p in per_vehicle_pools.values())
        )
        logger.info(
            "iter %d: dual=%.4f best_dual=%.4f best_primal=%s gap=%s "
            "sgnorm=%.3f step=%.4f violations=%d clean=%s subtime=%.2fs pool=%d",
            it, dual_value, best_dual,
            f"{best_primal:.4f}" if best_primal < float("inf") else "inf",
            f"{cert_gap:.4f}" if cert_gap is not None else "n/a",
            sg_norm, step, num_violations, all_clean, sub_time, pool_size,
        )

        if sg_norm < 1e-9 and all_clean:
            logger.info("iter %d: subgradient zero and all clean - assignment satisfied", it)
            break

    duals_all_zero = all(abs(v) < 1e-12 for v in multipliers.values())
    final_gap = None
    if best_primal < float("inf") and best_dual > float("-inf") and abs(best_primal) > 1e-12:
        final_gap = (best_primal - best_dual) / abs(best_primal)

    final_pool_size = (
        len(fleet_cut_pool) if per_vehicle_pools is None
        else sum(len(p) for p in per_vehicle_pools.values())
    )
    return LagrangianResult(
        best_dual=best_dual,
        best_primal=best_primal,
        certified_gap=final_gap,
        history=history,
        final_multipliers=multipliers,
        cut_pool_size=final_pool_size,
        best_primal_solution=best_primal_solution,
        duals_all_zero=duals_all_zero,
    )


def _solve_all_subproblems(
    instance, solver, multipliers, fleet_cut_pool, per_vehicle_pools,
    max_rounds, wall_budget, parallel,
) -> Dict[str, SubproblemResult]:
    if parallel:
        return _solve_parallel(
            instance, solver, multipliers, fleet_cut_pool, per_vehicle_pools,
            max_rounds, wall_budget)
    results: Dict[str, SubproblemResult] = {}
    for veh in instance.vehicles:
        pool = per_vehicle_pools[veh.id] if per_vehicle_pools is not None else fleet_cut_pool
        res = solve_routing_subproblem(
            instance, veh, instance.requests, solver,
            multipliers=multipliers, cut_pool=fleet_cut_pool,
            max_rounds=max_rounds, wall_budget_s=wall_budget,
            use_precedence_separation=True,  # lazy precedence separation
        )
        results[veh.id] = res
    return results


def _subproblem_worker(args):
    (instance, veh, multipliers, pool_snapshot, max_rounds, wall_budget,
     time_limit, mip_gap) = args
    solver = HighsSolver(time_limit_s=time_limit, mip_rel_gap=mip_gap)
    local_pool = list(pool_snapshot)
    res = solve_routing_subproblem(
        instance, veh, instance.requests, solver,
        multipliers=multipliers, cut_pool=local_pool,
        max_rounds=max_rounds, wall_budget_s=wall_budget,
        use_precedence_separation=True,  # lazy precedence separation
    )
    # return the cuts this worker discovered beyond the snapshot, so the main
    # process can merge them into the fleet-wide pool
    new_cuts = local_pool[len(pool_snapshot):]
    return veh.id, res, new_cuts


def _solve_parallel(
    instance, solver, multipliers, fleet_cut_pool, per_vehicle_pools, max_rounds, wall_budget,
) -> Dict[str, SubproblemResult]:
    import multiprocessing as mp

    time_limit = getattr(solver, "time_limit_s", 10.0)
    mip_gap = getattr(solver, "mip_rel_gap", 0.0)
    if per_vehicle_pools is not None:
        snapshots = {vid: list(p) for vid, p in per_vehicle_pools.items()}
    else:
        shared_snapshot = list(fleet_cut_pool)
        snapshots = {v.id: shared_snapshot for v in instance.vehicles}
    args = [
        (instance, veh, multipliers, snapshots[veh.id], max_rounds, wall_budget, time_limit, mip_gap)
        for veh in instance.vehicles
    ]
    results: Dict[str, SubproblemResult] = {}
    with mp.Pool(processes=min(len(instance.vehicles), mp.cpu_count())) as pool:
        for veh_id, res, new_cuts in pool.map(_subproblem_worker, args):
            results[veh_id] = res
            if per_vehicle_pools is not None:
                target = per_vehicle_pools[veh_id]
                for c in new_cuts:
                    if c not in set(target):
                        target.append(c)
            else:
                for c in new_cuts:
                    if c not in set(fleet_cut_pool):
                        fleet_cut_pool.append(c)
    return results
