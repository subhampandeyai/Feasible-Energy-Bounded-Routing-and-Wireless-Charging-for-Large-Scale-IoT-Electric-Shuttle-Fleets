from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.model.types import Instance
from src.solve.admm import (
    AdmmVehicleResult,
    _admm_x_update_worker,
    _x_update_one_vehicle,
    _x_update_one_vehicle_topk,
    _z_update,
)
from src.solve.column_generation import (
    ARTIFICIAL_COST_PER_REQUEST,
    ARTIFICIAL_VEHICLE_ID,
    Column,
    _empty_route_cost,
    _group_vehicles_by_type,
    _run_bound_checkpoint,
    _select_columns_exact,
    _solve_master,
)
from src.solve.solver_interface import HighsSolver, SolverInterface
from src.solve.subproblem import find_subtours
from src.verify.checker import VehicleRoute

logger = logging.getLogger(__name__)

# Ensemble: a portfolio of three existing components.
#
#   1. Heuristic labeling pricer (src/solve/labeling.py): budget-limited
#      search that finds improving columns without exhaustive enumeration.
#   2. Column generation master (src/solve/column_generation.py): the
#      set-partitioning LP over a growing column pool, with a separate
#      bound/certificate step.
#   3. ADMM per-vehicle x-updates (src/solve/admm.py), which run in
#      parallel across vehicles.
#
# Each round runs the ADMM x-updates per vehicle with the labeling pricer.
# Every column a vehicle's pricer generates is added to one shared column
# pool for every vehicle of the same type (see _group_vehicles_by_type in
# src/solve/column_generation.py): a column valid for one vehicle is a
# candidate for every identical vehicle. The CG master over the shared pool
# gives a feasible fleet assignment, and a valid dual bound where exact
# pricing terminates. Each round reports the better of ADMM's consensus
# primal (when feasible) and the CG master's primal.


@dataclass
class EnsembleCheckpoint:
    round: int
    admm_primal: Optional[float]  # ADMM's fleet objective from this round's x-update if every request is served exactly once; None otherwise
    cg_primal: float  # CG master's LP objective over the shared pool (always defined - the master is always solvable via seeded empty-route + artificial columns)
    winner: str  # "admm" | "cg" | "cg (admm infeasible)"
    best_primal: float
    unserved_count: int
    shared_pool_size: int
    columns_added_this_round: int
    x_update_wall_sequential_s: float
    x_update_wall_parallel_s: Optional[float]
    cg_master_wall_s: float
    valid_dual_bound: float  # -inf if not computable this round
    valid_certified_gap: Optional[float]
    bound_undetermined_vehicles: Tuple[str, ...]


@dataclass
class EnsembleResult:
    columns: List[Column]
    history: List[EnsembleCheckpoint]
    best_primal: float
    best_primal_source: str
    unserved_requests: Tuple[int, ...]
    per_vehicle_selected: Dict[str, Optional[Column]]
    rounds: int
    wall_time_s: float
    stop_reason: str
    final_valid_dual_bound: float
    final_valid_certified_gap: Optional[float]


def solve_ensemble(
    instance: Instance,
    solver: Optional[SolverInterface] = None,
    max_rounds: int = 100,
    rho: float = 5.0,
    initial_step: float = 2.0,
    heuristic_label_budget: int = 100000,
    heuristic_label_budget_cap: int = 5000000,
    heuristic_time_budget_s: float = 30.0,
    top_k: int = 20,
    parallel: bool = True,
    bound_check_every_rounds: int = 5,
    bound_exact_time_budget_s: float = 30.0,
    stagnant_stop_rounds: int = 15,
    wall_budget_s: float = 3600.0,
) -> EnsembleResult:
    """ADMM + column generation ensemble (see the module comment).

    Escalating per-group label budget, as in column_generation.py: a fixed
    heuristic_label_budget stops the shared pool from growing once the
    remaining improving routes need a deeper search. Each vehicle group's
    budget doubles (up to heuristic_label_budget_cap) on a round where the
    master objective does not improve, and is never lowered.
    """
    if solver is None:
        solver = HighsSolver(time_limit_s=30.0, mip_rel_gap=0.0)

    request_ids = [r.id for r in instance.requests]
    request_id_tuple = tuple(request_ids)
    vehicles = list(instance.vehicles)
    vehicle_ids = [v.id for v in vehicles]
    num_vehicles = len(vehicles)
    vehicle_groups = _group_vehicles_by_type(instance, vehicles)
    logger.info("ensemble: %d vehicles, %d distinct type(s), %d requests",
                num_vehicles, len(vehicle_groups), len(request_ids))

    # shared column pool, seeded exactly like CG's own seeding
    columns: List[Column] = []
    for veh in vehicles:
        columns.append(Column(vehicle_id=veh.id, served=(), cost=_empty_route_cost(instance, veh), arcs=[], charge={}))
    for u in request_ids:
        columns.append(Column(vehicle_id=ARTIFICIAL_VEHICLE_ID, served=(u,), cost=ARTIFICIAL_COST_PER_REQUEST, arcs=[], charge={}))

    # ADMM state
    mu: Dict[int, float] = {u: 0.0 for u in request_ids}
    z: Dict[str, Dict[int, float]] = {vid: {u: 1.0 / num_vehicles for u in request_ids} for vid in vehicle_ids}

    # Rank-diversified pool feeding. Identical vehicles see the same mu and
    # the same z-correction (1 - total_q)/K, so they solve the same pricing
    # problem and return the same best route; feeding only rank 0 would add
    # K copies of one column. solve_labeling with top_k > 1 already returns
    # several distinct completions, so each vehicle feeds the pool its
    # completion at rank = vehicle index (see _x_update_one_vehicle_topk).
    # ADMM's own x variable still uses rank 0, and mu is not perturbed.
    vehicle_rank_offset: Dict[str, int] = {veh.id: i for i, veh in enumerate(vehicles)}

    # per-vehicle-group escalating label budget (never lowered, see the
    # docstring), keyed by the group's type signature as in
    # column_generation.py's grp_label_budget.
    grp_label_budget: Dict[Tuple, int] = {g: heuristic_label_budget for g in vehicle_groups}

    history: List[EnsembleCheckpoint] = []
    t0 = time.time()
    stop_reason = "max_rounds_exceeded"
    best_primal = float("inf")
    best_primal_source = "none"
    best_selected: Dict[str, Optional[Column]] = {vid: None for vid in vehicle_ids}
    best_unserved: Tuple[int, ...] = tuple(request_ids)
    stagnant_rounds = 0
    last_cg_primal: Optional[float] = None
    final_bound = float("-inf")
    final_gap: Optional[float] = None
    rnd = 0

    for rnd in range(max_rounds):
        if time.time() - t0 > wall_budget_s:
            stop_reason = "wall_budget_exceeded"
            logger.warning("ensemble: wall budget %.1fs exceeded after %d rounds", wall_budget_s, rnd)
            break

        # --- ADMM x-update, parallel per vehicle, feeding the shared pool ---
        # Each vehicle uses its group's current escalating label budget
        # (grp_label_budget), not the flat heuristic_label_budget.
        tx0 = time.time()
        vehicle_results: Dict[str, AdmmVehicleResult] = {}
        pool_feed_results: Dict[str, AdmmVehicleResult] = {}
        vehicle_to_group: Dict[str, Tuple] = {
            m.id: g for g, members in vehicle_groups.items() for m in members
        }
        for veh in vehicles:
            budget = grp_label_budget[vehicle_to_group[veh.id]]
            rank0, diversified = _x_update_one_vehicle_topk(
                instance, veh, request_id_tuple, mu, z[veh.id], rho,
                budget, heuristic_time_budget_s, top_k, vehicle_rank_offset[veh.id],
            )
            vehicle_results[veh.id] = rank0
            pool_feed_results[veh.id] = diversified
        x_wall_seq = time.time() - tx0

        x_wall_par: Optional[float] = None
        if parallel:
            # timing measurement only - uses the same shared mu as the
            # rank-0 x-update (matches what _x_update_one_vehicle_topk's
            # own rank-0 branch computes), NOT the rank-diversified pool
            # feed, since this call exists purely to measure sequential vs
            # parallel wall time on equivalent per-vehicle pricing work.
            mu_by_vehicle = {veh.id: mu for veh in vehicles}
            x_wall_par = _measure_parallel_ensemble_x_update(
                instance, vehicles, request_id_tuple, mu_by_vehicle, z, rho,
                grp_label_budget, vehicle_to_group, heuristic_time_budget_s, top_k,
            )

        # Feed each vehicle's x-update route into the shared pool for every
        # vehicle in its type group.
        #
        # Every route is checked with find_subtours before it enters the
        # pool; a non-elementary route is discarded. Pool feeding uses
        # pool_feed_results (rank-diversified); admm_all_elementary and
        # admm_feasible check vehicle_results (rank 0), since they gate
        # ADMM's own primal.
        columns_added = 0
        admm_all_elementary = True
        for veh in vehicles:
            if find_subtours(vehicle_results[veh.id].arcs):
                admm_all_elementary = False
            vr = pool_feed_results[veh.id]
            grp = vehicle_to_group[veh.id]
            if not vr.served:
                continue
            if find_subtours(vr.arcs):
                logger.warning("ensemble round %d: vehicle %s's DIVERSIFIED x-update route has "
                                "repeated nodes (non-elementary) - DISCARDED, not added to the "
                                "shared pool.", rnd, veh.id)
                continue
            for member in vehicle_groups[grp]:
                columns.append(Column(vehicle_id=member.id, served=vr.served, cost=vr.cost, arcs=vr.arcs, charge=vr.charge))
                columns_added += 1

        # --- ADMM consensus update (z, mu) ---
        z_old = {vid: dict(z[vid]) for vid in vehicle_ids}
        z = _z_update(request_id_tuple, vehicle_results, vehicle_ids, rho)

        # Dual step decays as 1/sqrt(rnd+1). A harmonic 1/(rnd+1) step decays
        # too fast for mu to reach the price at which an expensive but
        # feasible request becomes worth serving; a Polyak step is not used
        # because the ensemble has no per-round lower bound.
        subgradient = {u: 1.0 - sum(vr.q.get(u, 0.0) for vr in vehicle_results.values()) for u in request_ids}
        step = initial_step / ((rnd + 1) ** 0.5)
        for u in request_ids:
            mu[u] += step * rho * subgradient[u]

        # ADMM's own candidate primal: valid only if every request is served
        # by exactly one vehicle this round and every vehicle's route is
        # elementary (find_subtours check above); otherwise None.
        served_by = {u: 0 for u in request_ids}
        for vr in vehicle_results.values():
            for u in vr.served:
                served_by[u] += 1
        admm_feasible = admm_all_elementary and all(served_by[u] == 1 for u in request_ids)
        admm_primal = sum(vr.cost for vr in vehicle_results.values()) if admm_feasible else None

        # --- CG master over the shared pool ---
        tcg0 = time.time()
        cg_result, pi, sigma, _ = _solve_master(solver, request_ids, columns, vehicle_ids)
        cg_wall = time.time() - tcg0
        cg_primal = cg_result.objective_value if cg_result.objective_value is not None else float("inf")

        # Budget escalation is triggered by the master objective, not by the
        # column count: the diversified feed adds some column almost every
        # round (often a cheap route re-covering a served request), even
        # when the budget is too small to reach a completion that serves a
        # currently unserved request. As in column_generation.py, every
        # group's budget doubles whenever cg_primal does not improve, and is
        # never lowered.
        objective_improved = cg_primal < (last_cg_primal - 1e-6) if last_cg_primal is not None else True
        last_cg_primal = min(last_cg_primal, cg_primal) if last_cg_primal is not None else cg_primal
        if not objective_improved:
            for grp in vehicle_groups:
                if grp_label_budget[grp] < heuristic_label_budget_cap:
                    old_budget = grp_label_budget[grp]
                    grp_label_budget[grp] = min(old_budget * 2, heuristic_label_budget_cap)
                    logger.info("ensemble round %d: master objective did not improve "
                                "(cg_primal=%.4f) - escalating group %s from label_budget=%d to %d",
                                rnd, cg_primal, grp, old_budget, grp_label_budget[grp])

        # --- best-of selection ---
        if admm_primal is not None and admm_primal < cg_primal:
            winner = "admm"
            round_best = admm_primal
        elif admm_primal is None:
            winner = "cg (admm infeasible)"
            round_best = cg_primal
        else:
            winner = "cg"
            round_best = cg_primal

        if round_best < best_primal - 1e-9:
            # The LP relaxation's y_c can be fractional, so the LP value
            # (round_best / cg_primal) is only the trigger for re-extracting.
            # The reported best_primal, best_selected and best_unserved
            # always come from the integer re-solve's true cost
            # (_select_columns_exact in column_generation.py).
            if winner == "admm":
                candidate_selected = {vid: Column(vid, vr.served, vr.cost, vr.arcs, vr.charge) for vid, vr in vehicle_results.items() if vr.served}
                for vid in vehicle_ids:
                    if vid not in candidate_selected:
                        candidate_selected[vid] = None
                candidate_unserved: Tuple[int, ...] = ()
                candidate_primal = admm_primal
                candidate_source = winner
            else:
                sel, unserved, _ = _select_columns_exact(columns, solver, request_ids, vehicle_ids, vehicles)
                candidate_selected = sel
                candidate_unserved = unserved
                candidate_primal = (sum(c.cost for c in sel.values() if c is not None)
                                     + ARTIFICIAL_COST_PER_REQUEST * len(unserved))
                candidate_source = winner

            # The LP's round_best can be more optimistic than any integer
            # solution, so the re-solve's true cost is compared against the
            # previous best_primal (also an exact cost) before replacing it.
            if candidate_primal < best_primal - 1e-9:
                best_primal = candidate_primal
                best_primal_source = candidate_source
                best_selected = candidate_selected
                best_unserved = candidate_unserved
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1
        else:
            stagnant_rounds += 1

        # --- periodic bound checkpoint over the shared pool ---
        valid_dual_bound = float("-inf")
        valid_gap: Optional[float] = None
        undetermined: Tuple[str, ...] = ()
        if (rnd + 1) % bound_check_every_rounds == 0:
            cg_result2, pi2, sigma2, _ = _solve_master(solver, request_ids, columns, vehicle_ids)
            checkpoint, bc_new_cols, bc_exact_calls = _run_bound_checkpoint(
                instance, vehicle_groups, vehicle_ids, request_id_tuple, pi2, sigma2, columns,
                cg_result2.objective_value, bound_exact_time_budget_s, top_k, rnd,
            )
            valid_dual_bound = checkpoint.valid_dual_bound
            valid_gap = checkpoint.valid_certified_gap
            undetermined = checkpoint.undetermined_vehicles
            final_bound = valid_dual_bound
            final_gap = valid_gap
            if bc_new_cols > 0:
                stagnant_rounds = 0

        history.append(EnsembleCheckpoint(
            round=rnd, admm_primal=admm_primal, cg_primal=cg_primal, winner=winner,
            best_primal=best_primal, unserved_count=len(best_unserved),
            shared_pool_size=len(columns), columns_added_this_round=columns_added,
            x_update_wall_sequential_s=x_wall_seq, x_update_wall_parallel_s=x_wall_par,
            cg_master_wall_s=cg_wall, valid_dual_bound=valid_dual_bound,
            valid_certified_gap=valid_gap, bound_undetermined_vehicles=undetermined,
        ))
        logger.info(
            "ensemble round %d: admm_primal=%s cg_primal=%.4f winner=%s best=%.4f "
            "unserved=%d pool=%d (+%d) x_seq=%.2fs x_par=%s cg_wall=%.3fs bound=%s",
            rnd, f"{admm_primal:.4f}" if admm_primal is not None else "infeasible",
            cg_primal, winner, best_primal, len(best_unserved), len(columns), columns_added,
            x_wall_seq, f"{x_wall_par:.2f}s" if x_wall_par is not None else "N/A", cg_wall,
            f"{valid_dual_bound:.4f}" if valid_dual_bound > float("-inf") else "N/A",
        )

        if stagnant_rounds >= stagnant_stop_rounds:
            stop_reason = f"stagnant_for_{stagnant_stop_rounds}_rounds"
            logger.info("ensemble round %d: no improvement for %d rounds - stopping.", rnd, stagnant_rounds)
            break
    else:
        stop_reason = "max_rounds_exceeded"

    # Final re-selection: as in solve_column_generation, re-solve the exact
    # integer selection over the final pool and keep it if its true cost
    # beats the best found during the rounds. The LP objective is never used
    # for this comparison.
    final_sel, final_unserved, final_result = _select_columns_exact(columns, solver, request_ids, vehicle_ids, vehicles)
    final_primal = (sum(c.cost for c in final_sel.values() if c is not None)
                     + ARTIFICIAL_COST_PER_REQUEST * len(final_unserved))
    if final_primal < best_primal - 1e-9:
        best_primal = final_primal
        best_primal_source = "cg (final re-selection, exact)"
        best_selected = final_sel
        best_unserved = final_unserved

    return EnsembleResult(
        columns=columns, history=history, best_primal=best_primal,
        best_primal_source=best_primal_source, unserved_requests=best_unserved,
        per_vehicle_selected=best_selected, rounds=rnd + 1, wall_time_s=time.time() - t0,
        stop_reason=stop_reason, final_valid_dual_bound=final_bound, final_valid_certified_gap=final_gap,
    )


def _measure_parallel_ensemble_x_update(
    instance: Instance, vehicles: List, request_ids: Tuple[int, ...],
    mu_by_vehicle: Dict[str, Dict[int, float]], z: Dict[str, Dict[int, float]], rho: float,
    grp_label_budget: Dict[Tuple, int], vehicle_to_group: Dict[str, Tuple],
    time_budget_s: float, top_k: int,
) -> float:
    import multiprocessing as mp
    args = [
        (instance, veh, request_ids, mu_by_vehicle[veh.id], z[veh.id], rho,
         grp_label_budget[vehicle_to_group[veh.id]], time_budget_s, top_k)
        for veh in vehicles
    ]
    t0 = time.time()
    with mp.Pool(processes=min(len(vehicles), mp.cpu_count())) as pool:
        pool.map(_admm_x_update_worker, args)
    return time.time() - t0


def to_vehicle_routes(selected: Dict[str, Optional[Column]]) -> Dict[str, VehicleRoute]:
    out = {}
    for vid, col in selected.items():
        if col is not None:
            out[vid] = VehicleRoute(arcs=col.arcs, charge=col.charge, served=list(col.served))
    return out
