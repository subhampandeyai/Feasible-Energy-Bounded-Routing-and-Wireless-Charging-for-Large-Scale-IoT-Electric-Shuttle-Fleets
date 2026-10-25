from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.model.milp_ir import MILPProblem
from src.model.types import Instance
from src.solve.labeling import reconstruct, solve_labeling
from src.solve.solver_interface import HighsSolver, SolverInterface

logger = logging.getLogger(__name__)

# Column generation for the joint routing+charging problem, with the primal
# loop and the dual-bound computation deliberately DECOUPLED.
#
# The per-vehicle pricing subproblem is NP-hard, so exact ESPPRC
# certification over many free-choice requests does not terminate in a
# tolerable budget. The primal loop only needs SOME improving column each
# round, and any genuinely negative reduced cost is a real improving column
# regardless of whether the search that found it was exhaustive - so the
# primal loop runs fast and heuristic-only.
#
# A valid Dantzig-Wolfe bound, by contrast, is only trustworthy when it comes
# from a pricing call that genuinely TERMINATED (exhausted its search or
# proved optimality). A budget-limited pass can return some improving columns
# while missing a far better one, which would make the bound look falsely
# tight. The bound is therefore computed as a separate, occasional step
# (every bound_check_every_rounds rounds, plus once at the end) using exact
# pricing; any vehicle whose exact pricing does not terminate within that
# step's budget is reported as UNDETERMINED for that bound.


def _vehicle_type_signature(instance: Instance, vehicle) -> Tuple:
    rate = vehicle.max_energy_consumption_kWh_per_km
    if rate is None:
        rate = instance.energy.consumption_kWh_per_km
    return (vehicle.start_node, vehicle.end_node, vehicle.battery_capacity_kWh,
            vehicle.initial_soc_kWh, rate)


def _group_vehicles_by_type(instance: Instance, vehicles: List) -> Dict[Tuple, List]:
    """Group vehicles that are structurally IDENTICAL (same start, end,
    battery, initial SOC, consumption rate; src/solve/lagrangian.py's
    _fleet_homogeneity_reason uses the same test). At the same duals,
    identical vehicles have the same pricing subproblem, so each group is
    priced once."""
    groups: Dict[Tuple, List] = {}
    for v in vehicles:
        groups.setdefault(_vehicle_type_signature(instance, v), []).append(v)
    return groups


ARTIFICIAL_VEHICLE_ID = "__unserved_penalty__"
# Penalty charged per request left on an artificial column. Set well above any
# real per-vehicle route cost so leaving a request unserved is never attractive,
# but not so large that it distorts early-round pricing.
ARTIFICIAL_COST_PER_REQUEST = 1000.0


@dataclass
class Column:
    vehicle_id: str  # ARTIFICIAL_VEHICLE_ID for the penalty columns seeded below
    served: Tuple[int, ...]
    cost: float  # the route's TRUE, undiscounted cost (distance + charging) - NEVER the pricing subproblem's reduced cost
    arcs: List[Tuple[int, int]]
    charge: Dict[Tuple[int, int], float]


def _build_master_lp(
    request_ids: List[int], columns: List[Column], vehicle_ids: List[str]
) -> Tuple[MILPProblem, Dict[int, int], Dict[str, int]]:
    """Set-partitioning master LP relaxation:

      min sum(cost_c * y_c) s.t.
        for each request u: sum_{c: u in served(c)} y_c = 1   [dual pi_u]
          - the assignment coupling constraint, sum_k q_u^k = 1.
        for each REAL vehicle instance k: sum_{c: vehicle(c)=k} y_c = 1
          [dual sigma_k]
          - every vehicle drives exactly one path, possibly its empty
            start->end route, whose nonzero cost is counted. The row is
            `= 1`, not `<= 1`: `<= 1` would let a vehicle opt out at no
            cost and under-count the fleet cost.
        0 <= y_c <= 1 (LP relaxation; the LP dual - pi AND sigma together -
        is what prices columns and gives the valid lower bound).

    The artificial penalty columns (vehicle_id=ARTIFICIAL_VEHICLE_ID) do
    NOT get a capacity row - they must remain always-available so the
    master is feasible regardless of real-vehicle capacity.
    """
    problem = MILPProblem(name="cg_master")
    col_indices = []
    for c in columns:
        idx = problem.add_variable(f"y_{len(col_indices)}", lower=0.0, upper=1.0, integer=False, obj=c.cost)
        col_indices.append(idx)

    row_of_request: Dict[int, int] = {}
    for u in request_ids:
        coeffs = {}
        for ci, c in enumerate(columns):
            if u in c.served:
                coeffs[col_indices[ci]] = coeffs.get(col_indices[ci], 0.0) + 1.0
        row_of_request[u] = problem.add_constraint(f"cover_{u}", coeffs, 1.0, 1.0)

    row_of_vehicle: Dict[str, int] = {}
    for k in vehicle_ids:
        coeffs = {}
        for ci, c in enumerate(columns):
            if c.vehicle_id == k:
                coeffs[col_indices[ci]] = coeffs.get(col_indices[ci], 0.0) + 1.0
        row_of_vehicle[k] = problem.add_constraint(f"capacity_{k}", coeffs, 1.0, 1.0)

    return problem, row_of_request, row_of_vehicle


def _build_master_mip(
    request_ids: List[int], columns: List[Column], vehicle_ids: List[str]
) -> MILPProblem:
    """INTEGER version of _build_master_lp, same rows/coefficients, but
    y_c in {0,1} instead of the LP relaxation's 0<=y_c<=1.

    Pricing, duals and bounds need the fractional LP: valid duals exist only
    for the LP. Thresholding the LP's y_c at 0.5 to read off a final answer
    is not safe: a request covered by several columns that are each below
    0.5 is then neither selected nor reported unserved. For a final answer,
    the same master over the same pool is solved as a 0/1 integer program
    (_select_columns_exact), so "selected" and "unserved" are exact.
    """
    problem = MILPProblem(name="cg_master_mip")
    col_indices = []
    for c in columns:
        idx = problem.add_variable(f"y_{len(col_indices)}", lower=0.0, upper=1.0, integer=True, obj=c.cost)
        col_indices.append(idx)

    for u in request_ids:
        coeffs = {}
        for ci, c in enumerate(columns):
            if u in c.served:
                coeffs[col_indices[ci]] = coeffs.get(col_indices[ci], 0.0) + 1.0
        problem.add_constraint(f"cover_{u}", coeffs, 1.0, 1.0)

    for k in vehicle_ids:
        coeffs = {}
        for ci, c in enumerate(columns):
            if c.vehicle_id == k:
                coeffs[col_indices[ci]] = coeffs.get(col_indices[ci], 0.0) + 1.0
        problem.add_constraint(f"capacity_{k}", coeffs, 1.0, 1.0)

    return problem


def _select_columns_exact(
    columns: List[Column], solver: SolverInterface, request_ids: List[int], vehicle_ids: List[str], vehicles
) -> Tuple[Dict[str, Optional[Column]], Tuple[int, ...], "SolveResult"]:
    """Final-answer extraction via a genuine integer solve (see
    _build_master_mip's docstring for why an integer solve is required here)
    - use this instead of _select_columns whenever the result will be
    reported as the actual solution (checker input, "unserved" counts in
    user-facing output), not just an intermediate round's bookkeeping."""
    from src.solve.solver_interface import SolveResult  # local import: type hint only, avoid a module-level cycle

    mip = _build_master_mip(request_ids, columns, vehicle_ids)
    result = solver.solve(mip)
    selected: Dict[str, Optional[Column]] = {v.id: None for v in vehicles}
    unserved: List[int] = []
    if result.col_values:
        for ci, c in enumerate(columns):
            if result.col_values[ci] <= 0.5:
                continue
            if c.vehicle_id == ARTIFICIAL_VEHICLE_ID:
                unserved.append(c.served[0])
                continue
            selected[c.vehicle_id] = c
    else:
        # no integer-feasible incumbent within the solver's time budget -
        # Fallback: everyone unserved, no route claimed. This is a
        # real degradation to report, not to hide.
        logger.error("cg_master_mip: no integer-feasible incumbent found (status=%s) - "
                     "final selection is EMPTY, not a real solution", result.status)
        unserved = list(request_ids)
    if unserved:
        logger.warning("cg_master_mip: %d request(s) not covered by any selected real route "
                        "in the EXACT (integer) final master solution: %s", len(unserved), sorted(unserved))
    return selected, tuple(sorted(unserved)), result


def _get_row_duals(solver: SolverInterface, problem: MILPProblem) -> List[float]:
    # SolveResult does not carry row duals, so the master LP is solved
    # directly with highspy here to read them.
    import highspy
    import numpy as np

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    lp.num_col_ = problem.num_vars
    lp.num_row_ = problem.num_rows
    lp.col_cost_ = np.array(problem.obj_coeffs, dtype=np.float64)
    lp.col_lower_ = np.array(problem.var_lower, dtype=np.float64)
    lp.col_upper_ = np.array(problem.var_upper, dtype=np.float64)
    lp.row_lower_ = np.array(problem.row_lower, dtype=np.float64)
    lp.row_upper_ = np.array(problem.row_upper, dtype=np.float64)
    lp.offset_ = problem.offset
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.integrality_ = [highspy.HighsVarType.kContinuous] * problem.num_vars
    starts, indices, values = problem.to_csc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = starts
    lp.a_matrix_.index_ = indices
    lp.a_matrix_.value_ = values
    h.passModel(lp)
    h.run()
    sol = h.getSolution()
    return list(sol.row_dual)


def _empty_route_cost(instance: Instance, vehicle) -> float:
    from src.solve.labeling import _dist_to_target
    dist = _dist_to_target(instance, vehicle.end_node)
    d = dist.get(vehicle.start_node)
    if d is None:
        raise ValueError(f"vehicle {vehicle.id}: no path from start {vehicle.start_node} to end {vehicle.end_node}")
    return instance.objective.epsilon1 * d


def _true_route_cost(instance: Instance, arcs: List[Tuple[int, int]], charge: Dict[Tuple[int, int], float]) -> float:
    """A column's cost in the master LP must be the route's true, undiscounted
    cost - not the label's .cost, which is the pricing subproblem's reduced
    cost (cost minus mu for every delivered request). Using the reduced cost
    would discount every column and make the master objective meaningless."""
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


def _add_column(columns: List[Column], instance: Instance, veh, label) -> None:
    arcs, charge = reconstruct(label)
    true_cost = _true_route_cost(instance, arcs, charge)
    columns.append(Column(vehicle_id=veh.id, served=tuple(sorted(label.delivered)),
                           cost=true_cost, arcs=arcs, charge=charge))


@dataclass
class PrimalRoundRecord:
    round: int
    master_objective: float
    columns_total: int
    new_columns: int
    unserved_requests: Tuple[int, ...]


@dataclass
class BoundCheckpoint:
    round: int
    valid_dual_bound: float  # sum(pi) + sum(sigma) + correction; -inf if not computable (some vehicle undetermined at this checkpoint)
    best_primal: float
    valid_certified_gap: Optional[float]  # None if valid_dual_bound is -inf
    fully_certified: bool  # True iff EVERY vehicle's exact pricing terminated (bound is TIGHT, not just valid)
    undetermined_vehicles: Tuple[str, ...]
    exact_calls_this_checkpoint: int
    wall_time_s: float


@dataclass
class ColumnGenerationResult:
    columns: List[Column]
    primal_history: List[PrimalRoundRecord]
    bound_checkpoints: List[BoundCheckpoint]
    best_primal: float
    final_bound_checkpoint: Optional[BoundCheckpoint]
    rounds: int
    heuristic_pricing_calls: int
    exact_pricing_calls: int
    wall_time_s: float
    stop_reason: str  # "primal_stagnant_for_N_rounds" | "wall_budget_exceeded" | "max_rounds_exceeded"
    per_vehicle_selected: Dict[str, Optional[Column]] = field(default_factory=dict)
    unserved_requests: Tuple[int, ...] = ()


def _solve_master(solver: SolverInterface, request_ids: List[int], columns: List[Column], vehicle_ids: List[str]):
    master, row_of_request, row_of_vehicle = _build_master_lp(request_ids, columns, vehicle_ids)
    result = solver.solve(master)
    if result.objective_value is None:
        raise RuntimeError(f"CG master LP infeasible/unsolved: status={result.status}")
    row_duals = _get_row_duals(solver, master)
    pi = {u: row_duals[row_of_request[u]] for u in request_ids}
    sigma = {k: row_duals[row_of_vehicle[k]] for k in vehicle_ids}
    return result, pi, sigma, master


def _seed_columns(instance: Instance, vehicles: List, request_ids: List[int]) -> List[Column]:
    columns: List[Column] = []
    for veh in vehicles:
        columns.append(Column(vehicle_id=veh.id, served=(), cost=_empty_route_cost(instance, veh), arcs=[], charge={}))
    for u in request_ids:
        columns.append(Column(vehicle_id=ARTIFICIAL_VEHICLE_ID, served=(u,), cost=ARTIFICIAL_COST_PER_REQUEST, arcs=[], charge={}))
    return columns


def _run_primal_round(
    instance: Instance, vehicle_groups: Dict[Tuple, List], request_id_tuple: Tuple[int, ...],
    pi: Dict[int, float], columns: List[Column], grp_label_budget: Dict[Tuple, int], pricing_top_k: int,
    heuristic_time_budget_s: float, label_budget_cap: int,
) -> Tuple[int, int]:
    """Fast, heuristic-ONLY pricing round: add every negative-reduced-cost
    column found (against pi alone - see note below on why sigma is not
    needed here). Returns (new_columns_added, heuristic_calls_made).

    Escalating label budget per group: a fixed label_budget stops finding
    improving columns as the duals change. The primal loop only needs some
    improving column, not a proof, so a group whose budget runs out without
    finding one has its budget doubled for the next attempt, up to
    label_budget_cap. A doubled budget is kept: lowering it after a success
    loses exactly the search depth the next improvement needs.

    Sigma is not needed here: any route with cost - sum(pi) < 0 is a valid
    candidate column, and the master LP decides whether it is used. sigma
    only enters the bound's correction term.
    """
    new_columns = 0
    heuristic_calls = 0
    for grp_key, grp_vehicles in vehicle_groups.items():
        rep_veh = grp_vehicles[0]
        budget = grp_label_budget[grp_key]
        best, stats = solve_labeling(
            instance, rep_veh, request_id_tuple, mu=pi,
            time_budget_s=heuristic_time_budget_s, label_budget=budget,
            top_k=pricing_top_k,
        )
        heuristic_calls += 1
        found_improving = False
        for lab in stats.get("top_completions", []):
            if lab.cost < -1e-6:
                for veh in grp_vehicles:
                    _add_column(columns, instance, veh, lab)
                    new_columns += 1
                found_improving = True
        if not found_improving and budget < label_budget_cap:
            grp_label_budget[grp_key] = min(budget * 2, label_budget_cap)
            logger.info("primal pricing: group %s found nothing at label_budget=%d - escalating to %d",
                        grp_key, budget, grp_label_budget[grp_key])
    return new_columns, heuristic_calls


def _run_bound_checkpoint(
    instance: Instance, vehicle_groups: Dict[Tuple, List], vehicle_ids: List[str],
    request_id_tuple: Tuple[int, ...], pi: Dict[int, float], sigma: Dict[str, float],
    columns: List[Column], master_objective: float,
    exact_time_budget_s: float, pricing_top_k: int, round_num: int,
) -> Tuple[BoundCheckpoint, int, int]:
    """Occasional, EXPENSIVE step: run exact (not label-budget-limited)
    pricing per vehicle GROUP (shared across identical instances for the
    route search, but each instance's own sigma_k decides its true reduced
    cost, so the correction is not simply multiplied by the group size).
    A vehicle whose exact call does
    not terminate within exact_time_budget_s is reported UNDETERMINED for
    THIS checkpoint - never assumed to contribute a 0.0 correction.
    Returns (checkpoint, new_columns_added, exact_calls_made)."""
    t0 = time.time()
    new_columns = 0
    exact_calls = 0
    veh_best_reduced_cost: Dict[str, float] = {}
    undetermined: List[str] = []

    for grp_key, grp_vehicles in vehicle_groups.items():
        rep_veh = grp_vehicles[0]
        best_exact, stats_exact = solve_labeling(
            instance, rep_veh, request_id_tuple, mu=pi,
            time_budget_s=exact_time_budget_s, top_k=pricing_top_k,
        )
        exact_calls += 1
        terminated = not stats_exact["status"].startswith("TIME_BUDGET")
        if not terminated:
            undetermined.extend(v.id for v in grp_vehicles)
            continue
        completions = stats_exact.get("top_completions", [])
        for veh in grp_vehicles:
            best_rc = 0.0
            for lab in completions:
                true_reduced_cost = lab.cost - sigma[veh.id]
                best_rc = min(best_rc, true_reduced_cost)
                if true_reduced_cost < -1e-6:
                    _add_column(columns, instance, veh, lab)
                    new_columns += 1
            veh_best_reduced_cost[veh.id] = best_rc

    wall = time.time() - t0
    if undetermined:
        valid_dual_bound = float("-inf")
        valid_certified_gap = None
    else:
        correction = sum(min(0.0, rc) for rc in veh_best_reduced_cost.values())
        valid_dual_bound = sum(pi.values()) + sum(sigma.values()) + correction
        valid_certified_gap = None
        if master_objective > 1e-12:
            valid_certified_gap = (master_objective - valid_dual_bound) / master_objective

    checkpoint = BoundCheckpoint(
        round=round_num, valid_dual_bound=valid_dual_bound, best_primal=master_objective,
        valid_certified_gap=valid_certified_gap, fully_certified=(not undetermined and new_columns == 0),
        undetermined_vehicles=tuple(sorted(undetermined)), exact_calls_this_checkpoint=exact_calls,
        wall_time_s=wall,
    )
    return checkpoint, new_columns, exact_calls


def solve_column_generation(
    instance: Instance,
    solver: Optional[SolverInterface] = None,
    max_rounds: int = 200,
    heuristic_label_budget: int = 100000,  # starting budget for the per-group escalating budget, not a cap
    heuristic_label_budget_cap: int = 5000000,  # upper limit for the escalating budget
    heuristic_time_budget_s: float = 10.0,
    pricing_top_k: int = 8,
    bound_check_every_rounds: int = 5,
    bound_exact_time_budget_s: float = 30.0,
    primal_stagnant_stop_rounds: int = 10,
    wall_budget_s: float = 1800.0,
) -> ColumnGenerationResult:
    """Column generation with decoupled primal and bound steps (see the
    module comment). The primal loop runs every round, fast, heuristic-
    only. The BOUND is computed as a separate, occasional step (every
    bound_check_every_rounds rounds, plus once at the end) using exact
    pricing - each checkpoint is independently valid or explicitly marked
    undetermined-for-some-vehicles, never silently assumed.
    """
    if solver is None:
        solver = HighsSolver(time_limit_s=30.0, mip_rel_gap=0.0)

    request_ids = [r.id for r in instance.requests]
    request_id_tuple = tuple(request_ids)
    vehicles = list(instance.vehicles)
    vehicle_ids = [v.id for v in vehicles]
    vehicle_groups = _group_vehicles_by_type(instance, vehicles)
    logger.info("column generation: %d vehicles grouped into %d distinct type(s)",
                len(vehicles), len(vehicle_groups))

    columns = _seed_columns(instance, vehicles, request_ids)
    grp_label_budget: Dict[Tuple, int] = {g: heuristic_label_budget for g in vehicle_groups}

    primal_history: List[PrimalRoundRecord] = []
    bound_checkpoints: List[BoundCheckpoint] = []
    heuristic_calls_total = 0
    exact_calls_total = 0
    t0 = time.time()
    stagnant_rounds = 0
    stop_reason = "max_rounds_exceeded"
    rounds_completed = 0
    last_objective = float("inf")

    for rnd in range(max_rounds):
        rounds_completed = rnd + 1
        if time.time() - t0 > wall_budget_s:
            stop_reason = "wall_budget_exceeded"
            logger.warning("column generation: wall budget %.1fs exceeded after %d rounds", wall_budget_s, rnd)
            break

        result, pi, sigma, _ = _solve_master(solver, request_ids, columns, vehicle_ids)
        new_columns, h_calls = _run_primal_round(
            instance, vehicle_groups, request_id_tuple, pi, columns,
            grp_label_budget, pricing_top_k, heuristic_time_budget_s, heuristic_label_budget_cap,
        )
        heuristic_calls_total += h_calls

        # Re-solve after adding this round's columns, so master_objective and
        # the unserved count reflect the updated pool (the earlier result's
        # col_values is shorter than the new column list).
        result, pi, sigma, _ = _solve_master(solver, request_ids, columns, vehicle_ids)
        unserved = _unserved_requests_at(request_ids, columns, result, vehicle_ids)
        primal_history.append(PrimalRoundRecord(
            round=rnd, master_objective=result.objective_value, columns_total=len(columns),
            new_columns=new_columns, unserved_requests=unserved,
        ))
        logger.info("CG round %d: master_obj=%.4f columns=%d (+%d) unserved=%d heuristic_calls=%d",
                    rnd, result.objective_value, len(columns), new_columns, len(unserved), h_calls)

        # Stagnation is measured on the master objective, not on the number
        # of new columns: columns with negative reduced cost can keep
        # arriving while the objective stays the same.
        objective_improved = result.objective_value < last_objective - 1e-6
        stagnant_rounds = 0 if objective_improved else stagnant_rounds + 1
        last_objective = min(last_objective, result.objective_value)

        if (rnd + 1) % bound_check_every_rounds == 0:
            result, pi, sigma, _ = _solve_master(solver, request_ids, columns, vehicle_ids)
            checkpoint, bc_new_cols, bc_exact_calls = _run_bound_checkpoint(
                instance, vehicle_groups, vehicle_ids, request_id_tuple, pi, sigma, columns,
                result.objective_value, bound_exact_time_budget_s, pricing_top_k, rnd,
            )
            exact_calls_total += bc_exact_calls
            bound_checkpoints.append(checkpoint)
            logger.info("CG round %d BOUND CHECKPOINT: valid_dual_bound=%s gap=%s undetermined=%s "
                        "(+%d new columns from exact search)", rnd,
                        f"{checkpoint.valid_dual_bound:.4f}" if checkpoint.valid_dual_bound > float("-inf") else "N/A",
                        f"{checkpoint.valid_certified_gap:.4f}" if checkpoint.valid_certified_gap is not None else "N/A",
                        list(checkpoint.undetermined_vehicles), bc_new_cols)
            if bc_new_cols > 0:
                stagnant_rounds = 0

        if stagnant_rounds >= primal_stagnant_stop_rounds:
            stop_reason = f"primal_stagnant_for_{primal_stagnant_stop_rounds}_rounds"
            logger.info("CG round %d: no new column for %d consecutive rounds - stopping.", rnd, stagnant_rounds)
            break
    else:
        stop_reason = "max_rounds_exceeded"

    # final bound checkpoint, always, regardless of how the loop stopped
    result, pi, sigma, _ = _solve_master(solver, request_ids, columns, vehicle_ids)
    final_checkpoint, bc_new_cols, bc_exact_calls = _run_bound_checkpoint(
        instance, vehicle_groups, vehicle_ids, request_id_tuple, pi, sigma, columns,
        result.objective_value, bound_exact_time_budget_s, pricing_top_k, rounds_completed,
    )
    exact_calls_total += bc_exact_calls
    bound_checkpoints.append(final_checkpoint)
    if bc_new_cols > 0:
        # the final checkpoint found more improving columns - re-solve once more for the true final primal
        result, pi, sigma, _ = _solve_master(solver, request_ids, columns, vehicle_ids)

    selected, unserved_final = _select_columns(columns, result, solver, request_ids, vehicle_ids, vehicles)

    return ColumnGenerationResult(
        columns=columns,
        primal_history=primal_history,
        bound_checkpoints=bound_checkpoints,
        best_primal=result.objective_value,
        final_bound_checkpoint=final_checkpoint,
        rounds=rounds_completed,
        heuristic_pricing_calls=heuristic_calls_total,
        exact_pricing_calls=exact_calls_total,
        wall_time_s=time.time() - t0,
        stop_reason=stop_reason,
        per_vehicle_selected=selected,
        unserved_requests=unserved_final,
    )


def _unserved_requests_at(request_ids, columns, result, vehicle_ids) -> Tuple[int, ...]:
    if not result.col_values:
        return tuple(request_ids)
    unserved = []
    for u in request_ids:
        covered_by_real = any(
            result.col_values[ci] > 0.5 and c.vehicle_id != ARTIFICIAL_VEHICLE_ID and u in c.served
            for ci, c in enumerate(columns)
        )
        if not covered_by_real:
            unserved.append(u)
    return tuple(unserved)


def _select_columns(columns, result, solver, request_ids, vehicle_ids, vehicles):
    selected: Dict[str, Optional[Column]] = {v.id: None for v in vehicles}
    unserved: List[int] = []
    if result.col_values:
        for ci, c in enumerate(columns):
            if result.col_values[ci] <= 0.5:
                continue
            if c.vehicle_id == ARTIFICIAL_VEHICLE_ID:
                unserved.extend(c.served)
                continue
            if selected[c.vehicle_id] is None or c.cost < selected[c.vehicle_id].cost:  # type: ignore[union-attr]
                selected[c.vehicle_id] = c
    if unserved:
        logger.warning("column generation: %d request(s) not covered by any selected real route "
                        "in the final master solution: %s", len(unserved), sorted(unserved))
    return selected, tuple(sorted(unserved))
