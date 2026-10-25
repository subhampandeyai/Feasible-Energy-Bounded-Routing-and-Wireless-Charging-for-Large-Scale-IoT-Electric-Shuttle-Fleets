from __future__ import annotations

from src.solve.subproblem import find_subtours


def test_simple_path_has_no_subtours():
    assert find_subtours([(0, 2), (2, 3), (3, 1)]) == []


def test_single_two_cycle_detected():
    subs = find_subtours([(0, 1), (1, 0)])
    assert subs == [(0, 1)]


def test_two_disjoint_cycles_detected():
    subs = find_subtours([(0, 1), (1, 0), (2, 3), (3, 4), (4, 2)])
    assert sorted(subs) == [(0, 1), (2, 3, 4)]


def test_path_plus_cycle():
    # a valid s-e path 10->11->12 plus a disjoint 3-cycle among 2,3,4
    subs = find_subtours([(10, 11), (11, 12), (2, 3), (3, 4), (4, 2)])
    assert subs == [(2, 3, 4)]


def test_self_loop_detected():
    subs = find_subtours([(5, 5), (0, 1)])
    assert subs == [(5,)]


def test_large_graph_no_recursion_limit():
    # a long path should not blow a recursion limit (Tarjan is iterative here)
    arcs = [(i, i + 1) for i in range(2000)]
    assert find_subtours(arcs) == []
