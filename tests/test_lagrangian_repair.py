from __future__ import annotations

from src.solve.column_generation import ARTIFICIAL_COST_PER_REQUEST
from src.solve.lagrangian_repair import repair_from_subproblems
from src.solve.subproblem import SubproblemResult
from src.verify.checker import check_solution
from tests.mini_instance import build_mini_instance


def _sub(vid, arcs, q, subtour_free=True):
    return SubproblemResult(
        vehicle_id=vid, dual_value=0.0, dual_is_proven_optimal=True, subtour_free=subtour_free,
        precedence_clean=True, solved_to_optimality=True, hit_round_cap=False,
        hit_wall_budget=False, rounds=1, total_solve_time_s=0.0, remaining_subtours=0,
        x_values={a: 1.0 for a in arcs}, q_values=dict(q),
        served_requests=tuple(u for u, v in q.items() if v > 0.5))


def test_duplicate_service_kept_once_and_plan_is_checker_clean():
    inst = build_mini_instance(num_vehicles=2)
    subs = {
        "veh_1": _sub("veh_1", [(0, 2), (2, 3), (3, 1)], {1: 1.0}),
        "veh_2": _sub("veh_2", [(0, 2), (2, 3), (3, 1)], {1: 1.0}),   # also claims request 1
    }
    out = repair_from_subproblems(inst, subs)
    assert out is not None
    assert out["served"] == [1] and out["unserved"] == [2]
    assert set(out["solution"]) == {"veh_1"}                           # veh_2 kept nothing, so it does not drive
    rep = check_solution(inst, out["solution"])
    assert [v for v in rep.violations if "served by NO vehicle" not in v] == []
    assert out["objective"] > ARTIFICIAL_COST_PER_REQUEST                # one request unserved is penalised


def test_both_requests_recovered():
    inst = build_mini_instance(num_vehicles=2)
    subs = {
        "veh_1": _sub("veh_1", [(0, 2), (2, 3), (3, 1)], {1: 1.0}),
        "veh_2": _sub("veh_2", [(0, 4), (4, 5), (5, 1)], {2: 1.0}),
    }
    out = repair_from_subproblems(inst, subs)
    assert out["served"] == [1, 2] and out["unserved"] == []
    assert check_solution(inst, out["solution"]).passed
    assert out["objective"] < ARTIFICIAL_COST_PER_REQUEST


def test_subtour_and_reversed_precedence_are_not_recovered():
    inst = build_mini_instance(num_vehicles=2)
    subs = {
        "veh_1": _sub("veh_1", [(0, 2), (2, 3), (3, 1)], {1: 1.0}, subtour_free=False),
        # path 0->4->3->1 visits request 2's pickup (4) but never its delivery (5)
        "veh_2": _sub("veh_2", [(0, 4), (4, 3), (3, 1)], {2: 1.0}),
    }
    assert repair_from_subproblems(inst, subs) is None
