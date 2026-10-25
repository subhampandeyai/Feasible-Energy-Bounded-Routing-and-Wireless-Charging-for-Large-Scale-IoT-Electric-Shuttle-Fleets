from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.model.types import Instance
from src.solve.labeling import reconstruct, solve_labeling
from src.verify.checker import VehicleRoute, check_solution

logger = logging.getLogger(__name__)

# ADMM (Alternating Direction Method of Multipliers) for the fleet
# assignment coupling constraint, sum_k q_u^k = 1 (equation (10)).
#
# Unlike the plain Lagrangian relaxation in src/solve/lagrangian.py and
# src/solve/column_generation.py, which adds only the linear term
# -mu_u * q_u^k to each subproblem, ADMM also adds the augmented quadratic
# penalty (rho/2)*||q_k - z||^2 per vehicle.
#
# LINEARIZATION: q_u^k in {0,1}, z_u in [0,1]. For binary q_u^k,
# (q_u^k)^2 = q_u^k, so:
#   (q_u^k - z_u)^2 = (q_u^k)^2 - 2*q_u^k*z_u + z_u^2
#                    = q_u^k - 2*q_u^k*z_u + z_u^2
#                    = z_u^2 + q_u^k * (1 - 2*z_u)
# The z_u^2 term does not depend on vehicle k's decisions and drops out of
# its minimization; only q_u^k * (1 - 2*z_u) remains, which is linear in
# q_u^k. The penalty therefore fits a pricing subproblem that accepts linear
# per-request costs (solve_labeling's mu term).
#
# Combined with the dual term -mu_u * q_u^k, vehicle k's effective request
# price (subtracted from route cost when request u is delivered) is:
#   effective_price_u^k = mu_u - (rho/2) * (1 - 2*z_u)
# and is passed to solve_labeling as mu.


@dataclass
class AdmmVehicleResult:
    vehicle_id: str
    served: Tuple[int, ...]
    cost: float  # true, undiscounted route cost
    arcs: List[Tuple[int, int]]
    charge: Dict[Tuple[int, int], float]
    q: Dict[int, float]  # this vehicle's copy of the assignment indicator, one entry per request (0.0 or 1.0)
    solve_time_s: float


@dataclass
class AdmmIterationRecord:
    iteration: int
    primal_residual: float  # ||q_k - z|| aggregated (sqrt of sum of squares) across all vehicles/requests
    dual_residual: float  # rho * ||z_new - z_old|| (standard ADMM dual residual definition)
    fleet_objective: float  # sum of true route costs (NOT including penalty/dual terms) for the current x-update's routes
    unserved_requests: Tuple[int, ...]
    x_update_wall_s: float  # SEQUENTIAL wall time for this iteration's x-update (all vehicles solved one after another)
    x_update_wall_parallel_s: Optional[float]  # measured PARALLEL wall time, if parallel execution was used this iteration - None if not measured
    z_update_wall_s: float
    rho: float


@dataclass
class AdmmResult:
    vehicle_results: Dict[str, AdmmVehicleResult]
    z: Dict[int, float]
    history: List[AdmmIterationRecord]
    converged: bool  # True iff BOTH primal and dual residuals fell below their targets before max_iterations
    iterations_run: int
    total_wall_s: float
    fleet_objective: float
    unserved_requests: Tuple[int, ...]


def _true_route_cost(instance: Instance, arcs: List[Tuple[int, int]], charge: Dict[Tuple[int, int], float]) -> float:
    arc_dist = {(a.from_node, a.to_node): a.distance_km for a in instance.arcs}
    eps1 = instance.objective.epsilon1
    eps2 = instance.objective.epsilon2
    total = 0.0
    for (i, j) in arcs:
        total += eps1 * arc_dist.get((i, j), 0.0)
    for (i, j), amt in charge.items():
        price = instance.prices.base_price_per_node.get(i, 0.0)
        eta = 1.0 if instance.charging.arc_enabled(i, j) else 0.0
        total += eps2 * price * eta * amt
    return total


def _label_to_vehicle_result(instance: Instance, veh, request_ids: Tuple[int, ...], label, wall: float) -> AdmmVehicleResult:
    arcs, charge = reconstruct(label)
    true_cost = _true_route_cost(instance, arcs, charge)
    q = {u: (1.0 if u in label.delivered else 0.0) for u in request_ids}
    return AdmmVehicleResult(
        vehicle_id=veh.id, served=tuple(sorted(label.delivered)), cost=true_cost,
        arcs=arcs, charge=charge, q=q, solve_time_s=wall,
    )


def _empty_vehicle_result(veh, request_ids: Tuple[int, ...], wall: float) -> AdmmVehicleResult:
    # No complete label within budget: report an empty route.
    return AdmmVehicleResult(
        vehicle_id=veh.id, served=(), cost=0.0, arcs=[], charge={},
        q={u: 0.0 for u in request_ids}, solve_time_s=wall,
    )


def _x_update_one_vehicle(
    instance: Instance, veh, request_ids: Tuple[int, ...],
    mu: Dict[int, float], z: Dict[int, float], rho: float,
    label_budget: int, time_budget_s: float, top_k: int,
) -> AdmmVehicleResult:
    """One vehicle's ADMM x-update: solve its pricing subproblem with the
    combined linear price (dual term + linearized quadratic penalty, see the
    module comment). Uses the same labeling pricer as the Lagrangian and CG
    code (src/solve/labeling.py); the augmented penalty only changes the
    per-request price passed as mu.

    Always uses the rank-0 (best) completion, which is ADMM's x variable.
    For a rank-diversified variant used only to feed the ensemble's shared
    column pool, see _x_update_one_vehicle_topk."""
    t0 = time.time()
    effective_price = {u: mu.get(u, 0.0) - (rho / 2.0) * (1.0 - 2.0 * z.get(u, 0.0)) for u in request_ids}
    best, stats = solve_labeling(
        instance, veh, request_ids, mu=effective_price,
        time_budget_s=time_budget_s, label_budget=label_budget, top_k=top_k,
    )
    wall = time.time() - t0
    if best is None:
        return _empty_vehicle_result(veh, request_ids, wall)
    return _label_to_vehicle_result(instance, veh, request_ids, best, wall)


def _x_update_one_vehicle_topk(
    instance: Instance, veh, request_ids: Tuple[int, ...],
    mu: Dict[int, float], z: Dict[int, float], rho: float,
    label_budget: int, time_budget_s: float, top_k: int, rank: int,
) -> Tuple[AdmmVehicleResult, AdmmVehicleResult]:
    """Ensemble-only variant: returns both the rank-0 result (ADMM's x
    variable, as in _x_update_one_vehicle) and a rank-diversified result
    (completion min(rank, len(completions)-1)) for the shared column pool,
    so identical vehicles do not all contribute the same column. Falls back
    to the rank-0 result when there are no further completions."""
    t0 = time.time()
    effective_price = {u: mu.get(u, 0.0) - (rho / 2.0) * (1.0 - 2.0 * z.get(u, 0.0)) for u in request_ids}
    best, stats = solve_labeling(
        instance, veh, request_ids, mu=effective_price,
        time_budget_s=time_budget_s, label_budget=label_budget, top_k=top_k,
    )
    wall = time.time() - t0
    if best is None:
        empty = _empty_vehicle_result(veh, request_ids, wall)
        return empty, empty

    rank0_result = _label_to_vehicle_result(instance, veh, request_ids, best, wall)
    completions = stats.get("top_completions", [])
    if not completions:
        return rank0_result, rank0_result
    diversified_idx = min(rank, len(completions) - 1)
    diversified_label = completions[diversified_idx]
    diversified_result = _label_to_vehicle_result(instance, veh, request_ids, diversified_label, wall)
    return rank0_result, diversified_result


def _z_update(request_ids: Tuple[int, ...], vehicle_results: Dict[str, AdmmVehicleResult],
              vehicle_ids: List[str], rho: float) -> Dict[str, Dict[int, float]]:
    """z-update: projection of the vehicle copies onto sum_k z_u^k = 1.

    Plain consensus averaging (one shared z_u = mean_k q_u^k) has a fixed
    point at q_u^k = 0 for all k, z_u = 0, which satisfies consensus but not
    sum_k q_u^k = 1. The constraint is on the sum of the copies, so per
    request u the z-update solves:
        minimize_{z_u^1..z_u^K}  sum_k (rho/2)*(q_u^k - z_u^k)^2
        subject to               sum_k z_u^k = 1
    With L = sum_k (rho/2)(q_u^k - z_u^k)^2 - lambda*(sum_k z_u^k - 1),
    stationarity gives z_u^k = q_u^k + lambda/rho for every k, and the
    constraint gives lambda/rho = (1 - sum_k q_u^k) / K. So:
        z_u^k = q_u^k + (1 - sum_j q_u^j) / K   for every vehicle k
    At q = 0 for all k this gives z_u^k = 1/K, not 0."""
    num_vehicles = len(vehicle_ids)
    z: Dict[str, Dict[int, float]] = {vid: {} for vid in vehicle_ids}
    for u in request_ids:
        total_q = sum(vehicle_results[vid].q.get(u, 0.0) for vid in vehicle_ids)
        correction = (1.0 - total_q) / num_vehicles
        for vid in vehicle_ids:
            q_uk = vehicle_results[vid].q.get(u, 0.0)
            z[vid][u] = q_uk + correction
    return z


def solve_admm(
    instance: Instance,
    max_iterations: int = 50,
    rho: float = 5.0,
    initial_step: float = 2.0,
    label_budget: int = 100000,
    time_budget_s: float = 60.0,
    top_k: int = 20,
    primal_residual_target: float = 1e-3,
    dual_residual_target: float = 1e-3,
    parallel: bool = False,
    measure_parallel_timing: bool = True,
) -> AdmmResult:
    """ADMM on the assignment coupling constraint sum_k q_u^k = 1.

    x-update: per vehicle, solve_labeling with the combined dual +
    linearized-quadratic-penalty price (see _x_update_one_vehicle). Runs
    sequentially; with parallel=True it also times the same x-update under
    multiprocessing and reports both times.

    z-update: per-vehicle, per-request projection onto sum_k z_u^k = 1
    (see _z_update).

    dual update: mu_u += step * rho * (1 - sum_k q_u^k), the same
    subgradient direction as the Lagrangian loop in src/solve/lagrangian.py.
    """
    request_ids = tuple(r.id for r in instance.requests)
    vehicles = list(instance.vehicles)
    vehicle_ids = [v.id for v in vehicles]
    num_vehicles = len(vehicles)

    mu: Dict[int, float] = {u: 0.0 for u in request_ids}
    # z is per vehicle (see _z_update); seed it at 1/num_vehicles per
    # request so the first x-update does not start at the all-zero point.
    z: Dict[str, Dict[int, float]] = {
        vid: {u: 1.0 / num_vehicles for u in request_ids} for vid in vehicle_ids
    }

    history: List[AdmmIterationRecord] = []
    t0 = time.time()
    converged = False
    vehicle_results: Dict[str, AdmmVehicleResult] = {}
    it = 0

    for it in range(max_iterations):
        # x-update (sequential timing always measured; parallel timing
        # measured separately if requested, on the SAME work, so the two
        # are directly comparable)
        tx0 = time.time()
        vehicle_results = {}
        for veh in vehicles:
            vehicle_results[veh.id] = _x_update_one_vehicle(
                instance, veh, request_ids, mu, z[veh.id], rho, label_budget, time_budget_s, top_k,
            )
        x_wall_sequential = time.time() - tx0

        x_wall_parallel: Optional[float] = None
        if parallel and measure_parallel_timing:
            x_wall_parallel = _measure_parallel_x_update(
                instance, vehicles, request_ids, mu, z, rho, label_budget, time_budget_s, top_k,
            )

        # z-update
        tz0 = time.time()
        z_old = {vid: dict(z[vid]) for vid in vehicle_ids}
        z = _z_update(request_ids, vehicle_results, vehicle_ids, rho)
        z_wall = time.time() - tz0

        # residuals (per-vehicle z, see _z_update)
        primal_residual_sq = 0.0
        for u in request_ids:
            for vid, vr in vehicle_results.items():
                diff = vr.q.get(u, 0.0) - z[vid][u]
                primal_residual_sq += diff * diff
        primal_residual = primal_residual_sq ** 0.5

        dual_residual_sq = sum(
            (z[vid][u] - z_old[vid][u]) ** 2 for vid in vehicle_ids for u in request_ids
        )
        dual_residual = rho * (dual_residual_sq ** 0.5)

        # Dual update with subgradient 1 - total_q, the same sign as
        # src/solve/lagrangian.py: an under-served request (total_q = 0)
        # gets a larger mu, which makes it more attractive through
        # solve_labeling's cost -= mu[rid] term at delivery.
        for u in request_ids:
            total_q = sum(vr.q.get(u, 0.0) for vr in vehicle_results.values())
            step = initial_step / (it + 1)
            mu[u] += step * rho * (1.0 - total_q)

        fleet_obj = sum(vr.cost for vr in vehicle_results.values())
        served_by = {u: 0 for u in request_ids}
        for vr in vehicle_results.values():
            for u in vr.served:
                served_by[u] += 1
        unserved = tuple(sorted(u for u in request_ids if served_by[u] != 1))

        history.append(AdmmIterationRecord(
            iteration=it, primal_residual=primal_residual, dual_residual=dual_residual,
            fleet_objective=fleet_obj, unserved_requests=unserved,
            x_update_wall_s=x_wall_sequential, x_update_wall_parallel_s=x_wall_parallel,
            z_update_wall_s=z_wall, rho=rho,
        ))
        logger.info(
            "ADMM iter %d: primal_res=%.6f dual_res=%.6f fleet_obj=%.4f unserved=%d "
            "x_wall_seq=%.2fs x_wall_par=%s z_wall=%.4fs",
            it, primal_residual, dual_residual, fleet_obj, len(unserved),
            x_wall_sequential, f"{x_wall_parallel:.2f}s" if x_wall_parallel is not None else "N/A",
            z_wall,
        )

        if primal_residual < primal_residual_target and dual_residual < dual_residual_target:
            converged = True
            logger.info("ADMM iter %d: CONVERGED (primal_res < %.4g, dual_res < %.4g)",
                        it, primal_residual_target, dual_residual_target)
            break

    served_by_final = {u: 0 for u in request_ids}
    for vr in vehicle_results.values():
        for u in vr.served:
            served_by_final[u] += 1
    unserved_final = tuple(sorted(u for u in request_ids if served_by_final[u] != 1))
    fleet_obj_final = sum(vr.cost for vr in vehicle_results.values())

    return AdmmResult(
        vehicle_results=vehicle_results, z=z, history=history, converged=converged,
        iterations_run=it + 1, total_wall_s=time.time() - t0,
        fleet_objective=fleet_obj_final, unserved_requests=unserved_final,
    )


def _measure_parallel_x_update(
    instance: Instance, vehicles: List, request_ids: Tuple[int, ...],
    mu: Dict[int, float], z: Dict[str, Dict[int, float]], rho: float,
    label_budget: int, time_budget_s: float, top_k: int,
) -> float:
    """Wall-clock time of one x-update round under multiprocessing. This is
    measured separately from the sequential x-update, on the same work, so
    the two times are directly comparable."""
    import multiprocessing as mp

    # The worker is a module-level function with an explicit args tuple
    # because the spawn start method (Windows) cannot pickle local closures.
    args = [
        (instance, veh, request_ids, mu, z[veh.id], rho, label_budget, time_budget_s, top_k)
        for veh in vehicles
    ]
    t0 = time.time()
    with mp.Pool(processes=min(len(vehicles), mp.cpu_count())) as pool:
        pool.map(_admm_x_update_worker, args)
    return time.time() - t0


def _admm_x_update_worker(args) -> AdmmVehicleResult:
    instance, veh, request_ids, mu, z_veh, rho, label_budget, time_budget_s, top_k = args
    return _x_update_one_vehicle(instance, veh, request_ids, mu, z_veh, rho, label_budget, time_budget_s, top_k)


def to_vehicle_routes(vehicle_results: Dict[str, AdmmVehicleResult]) -> Dict[str, VehicleRoute]:
    return {
        vid: VehicleRoute(arcs=vr.arcs, charge=vr.charge, served=list(vr.served))
        for vid, vr in vehicle_results.items()
    }
