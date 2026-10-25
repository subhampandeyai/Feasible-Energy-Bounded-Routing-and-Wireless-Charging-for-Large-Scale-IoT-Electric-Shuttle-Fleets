from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import highspy
import numpy as np

from src.model.milp_ir import MILPProblem

# The only module that imports highspy. Callers build a model.milp_ir.MILPProblem
# and see only SolverInterface.solve() and SolveResult, so changing the solver
# means changing this module alone.

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    problem_name: str
    status: str
    is_optimal: bool
    timed_out: bool
    objective_value: Optional[float]
    col_values: List[float]
    solve_time_s: float
    mip_gap: Optional[float]
    mip_dual_bound: Optional[float]


class SolverInterface:
    """Abstract boundary every caller in src/solve/ programs against."""

    def solve(self, problem: MILPProblem) -> SolveResult:
        raise NotImplementedError


class HighsSolver(SolverInterface):
    def __init__(
        self,
        time_limit_s: Optional[float] = None,
        mip_rel_gap: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        self.time_limit_s = time_limit_s
        self.mip_rel_gap = mip_rel_gap
        self.verbose = verbose

    def solve(self, problem: MILPProblem) -> SolveResult:
        h = highspy.Highs()
        h.setOptionValue("output_flag", self.verbose)
        if self.time_limit_s is not None:
            h.setOptionValue("time_limit", float(self.time_limit_s))
        if self.mip_rel_gap is not None:
            h.setOptionValue("mip_rel_gap", float(self.mip_rel_gap))

        lp = highspy.HighsLp()
        lp.num_col_ = problem.num_vars
        lp.num_row_ = problem.num_rows
        lp.col_cost_ = np.array(problem.obj_coeffs, dtype=np.float64)
        lp.col_lower_ = np.array(problem.var_lower, dtype=np.float64)
        lp.col_upper_ = np.array(problem.var_upper, dtype=np.float64)
        lp.row_lower_ = np.array(problem.row_lower, dtype=np.float64)
        lp.row_upper_ = np.array(problem.row_upper, dtype=np.float64)
        lp.offset_ = problem.offset
        lp.sense_ = highspy.ObjSense.kMinimize if problem.minimize else highspy.ObjSense.kMaximize
        lp.integrality_ = [
            highspy.HighsVarType.kInteger if is_int else highspy.HighsVarType.kContinuous
            for is_int in problem.var_integer
        ]

        starts, indices, values = problem.to_csc()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = starts
        lp.a_matrix_.index_ = indices
        lp.a_matrix_.value_ = values

        pass_status = h.passModel(lp)
        if pass_status != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS passModel failed for {problem.name}: {pass_status}")

        run_status = h.run()
        model_status = h.getModelStatus()
        status_str = h.modelStatusToString(model_status)
        solve_time = h.getRunTime()
        info = h.getInfo()

        is_optimal = model_status == highspy.HighsModelStatus.kOptimal
        timed_out = model_status == highspy.HighsModelStatus.kTimeLimit
        # A solve that hits the time limit without an integer-feasible
        # incumbent still reports kTimeLimit, and col_value then holds the
        # last LP point, not a solution. Only primal_solution_status ==
        # kSolutionStatusFeasible marks col_value as a genuine incumbent.
        has_incumbent = info.primal_solution_status == highspy.kSolutionStatusFeasible

        col_values: List[float] = []
        objective_value: Optional[float] = None
        if has_incumbent:
            sol = h.getSolution()
            col_values = list(sol.col_value)
            if col_values:
                objective_value = h.getObjectiveValue()

        result = SolveResult(
            problem_name=problem.name,
            status=status_str,
            is_optimal=is_optimal,
            timed_out=timed_out,
            objective_value=objective_value,
            col_values=col_values,
            solve_time_s=solve_time,
            mip_gap=getattr(info, "mip_gap", None),
            mip_dual_bound=getattr(info, "mip_dual_bound", None),
        )

        # Every solve is logged with status, time and gap. A timeout is
        # reported as such, never treated as a proven-optimal solution.
        logger.info(
            "solved %s: status=%s time=%.3fs gap=%s dual_bound=%s timed_out=%s",
            problem.name,
            status_str,
            solve_time,
            result.mip_gap,
            result.mip_dual_bound,
            timed_out,
        )
        if timed_out and has_incumbent:
            logger.warning(
                "%s hit the time limit (%.1fs) before proving optimality - "
                "gap=%s, using best incumbent found",
                problem.name,
                self.time_limit_s,
                result.mip_gap,
            )
        elif timed_out and not has_incumbent:
            logger.error(
                "%s hit the time limit (%.1fs) WITHOUT finding any integer-"
                "feasible incumbent - no usable solution from this solve, "
                "not just a weak one",
                problem.name,
                self.time_limit_s,
            )
        if not is_optimal and not timed_out:
            logger.error("%s did NOT solve to optimality: status=%s", problem.name, status_str)

        return result
