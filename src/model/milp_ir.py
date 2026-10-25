from __future__ import annotations

from typing import Dict, List, Tuple

# Solver-agnostic MILP representation: variables, bounds, objective and sparse
# constraint rows. src/solve/solver_interface.py translates it into the solver's
# API and is the only module that depends on the solver.


class MILPProblem:
    def __init__(self, name: str, minimize: bool = True) -> None:
        self.name = name
        self.minimize = minimize
        self.var_names: List[str] = []
        self.var_lower: List[float] = []
        self.var_upper: List[float] = []
        self.var_integer: List[bool] = []
        self.obj_coeffs: List[float] = []
        self.row_names: List[str] = []
        self.row_lower: List[float] = []
        self.row_upper: List[float] = []
        self._row_entries: List[List[Tuple[int, float]]] = []
        self.offset: float = 0.0

    def add_variable(
        self, name: str, lower: float, upper: float, integer: bool = False, obj: float = 0.0
    ) -> int:
        idx = len(self.var_names)
        self.var_names.append(name)
        self.var_lower.append(lower)
        self.var_upper.append(upper)
        self.var_integer.append(integer)
        self.obj_coeffs.append(obj)
        return idx

    def add_to_objective(self, var_idx: int, amount: float) -> None:
        self.obj_coeffs[var_idx] += amount

    def add_constraint(
        self, name: str, coeffs: Dict[int, float], lower: float, upper: float
    ) -> int:
        row_idx = len(self.row_names)
        self.row_names.append(name)
        self.row_lower.append(lower)
        self.row_upper.append(upper)
        self._row_entries.append(list(coeffs.items()))
        return row_idx

    @property
    def num_vars(self) -> int:
        return len(self.var_names)

    @property
    def num_rows(self) -> int:
        return len(self.row_names)

    def to_csc(self):
        """Column-major sparse matrix (starts, indices, values) - the format
        highspy's Highs.passModel expects for a_matrix."""
        col_entries: List[List[Tuple[int, float]]] = [[] for _ in range(self.num_vars)]
        for row_idx, entries in enumerate(self._row_entries):
            for var_idx, coeff in entries:
                col_entries[var_idx].append((row_idx, coeff))
        starts = [0]
        indices: List[int] = []
        values: List[float] = []
        for col in col_entries:
            for row_idx, coeff in col:
                indices.append(row_idx)
                values.append(coeff)
            starts.append(len(indices))
        return starts, indices, values
