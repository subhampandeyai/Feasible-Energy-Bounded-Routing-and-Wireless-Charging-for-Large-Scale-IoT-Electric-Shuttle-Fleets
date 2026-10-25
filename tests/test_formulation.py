from __future__ import annotations

import os

from src.model.formulation import build_vehicle_subproblem
from src.model.readers.dataset_reader import load_dataset_instance

DATASET = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def test_start_end_arcs_pinned_to_zero():
    inst = load_dataset_instance(DATASET, 30)
    veh = inst.vehicles[0]
    problem, idx = build_vehicle_subproblem(inst, veh, inst.requests)
    # every arc into start(0) or out of end(1) must have upper bound 0
    for (i, j), vidx in idx.x.items():
        if j == veh.start_node or i == veh.end_node:
            assert problem.var_upper[vidx] == 0.0, f"arc ({i},{j}) not pinned"
        else:
            assert problem.var_upper[vidx] == 1.0


def test_mandatory_request_fixes_q_lower_bound():
    inst = load_dataset_instance(DATASET, 30)
    veh = inst.vehicles[0]
    problem, idx = build_vehicle_subproblem(
        inst, veh, inst.requests, mandatory_requests={1, 2})
    assert problem.var_lower[idx.q[1]] == 1.0
    assert problem.var_lower[idx.q[2]] == 1.0
    assert problem.var_lower[idx.q[3]] == 0.0


def test_variable_counts():
    inst = load_dataset_instance(DATASET, 30)
    veh = inst.vehicles[0]
    problem, idx = build_vehicle_subproblem(inst, veh, inst.requests)
    assert len(idx.x) == 3632
    assert len(idx.c) == 3632
    assert len(idx.B) == 602
    assert len(idx.q) == 30
    # t (MTZ position) vars exist but MTZ constraints are off by default
    assert len(idx.t) == 601


def test_gg_flow_cap_values():
    # flow_cap must match the demand definition in use. Under the
    # non-Steiner definition every visited node carries demand, so a route's
    # total node count - transit included - can exceed 2*len(requests),
    # making that value an invalid cap; the valid cap is n_nodes - 2. Under
    # the Steiner definition only pickup/delivery nodes carry demand, so
    # 2*len(requests) is valid and much tighter.
    #
    # This runs on the full 602-node/30-request data/ instance: the 6-node
    # mini instance never approaches either bound and cannot detect a
    # mismatched cap.
    inst = load_dataset_instance(DATASET, 30)
    veh = inst.vehicles[0]
    mandatory = {1, 2, 3}

    # non-Steiner: flow_cap must bound n_nodes - 2 (max possible route
    # length in nodes, since every visited node - transit included - counts)
    problem, idx = build_vehicle_subproblem(
        inst, veh, inst.requests, include_gg_flow=True,
        gg_steiner_demand=False, mandatory_requests=mandatory,
    )
    some_f = next(iter(idx.f.values()))
    assert problem.var_upper[some_f] == len(inst.nodes) - 2 == 600.0

    # Steiner: flow_cap must bound 2*len(requests) - only pickup/delivery
    # nodes of candidate requests can carry positive demand, transit nodes
    # cannot, regardless of how many transit nodes the route passes through
    problem2, idx2 = build_vehicle_subproblem(
        inst, veh, inst.requests, include_gg_flow=True,
        gg_steiner_demand=True, mandatory_requests=mandatory,
    )
    some_f2 = next(iter(idx2.f.values()))
    assert problem2.var_upper[some_f2] == 2 * len(inst.requests) == 60.0
