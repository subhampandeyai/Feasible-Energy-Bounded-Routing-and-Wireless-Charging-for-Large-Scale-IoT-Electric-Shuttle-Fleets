from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import json, logging, sys, time

from src.model.readers.dataset_reader import load_dataset_instance
from src.solve.labeling import solve_labeling, reconstruct
from src.solve.subproblem import find_subtours, solve_routing_subproblem
from src.solve.solver_interface import HighsSolver
from src.solve.insertion_router import route_cluster_insertion
from src.verify.checker import check_solution, VehicleRoute
import math

# Per-vehicle engine comparison. Three engines are run on the same feasible
# spatial clusters of increasing size k, the same vehicle (veh_1) and the same
# instance (R=100 request pool, 602-node graph):
#   1. ESPPRC labeling  (exact / proves optimality)
#   2. GG-flow single-vehicle MILP (elementarity via flow)
#   3. insertion router (the proposed feasibility engine)
# Every cell gets a status, label count where applicable, and wall time.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)

R = 100
inst = load_dataset_instance("data", R)
veh = inst.vehicles[0]
solo = json.load(open(f"results/coverage/solo_R{R}.json"))
feas = sorted(set(solo["feasible"]) | set(solo.get("undetermined", [])))


def mid(rid):
    r = inst.request_by_id(rid)
    p = inst.nodes[r.pickup_node]
    d = inst.nodes[r.delivery_node]
    return ((p.x_km + d.x_km) / 2, (p.y_km + d.y_km) / 2)


seed = feas[0]
order = sorted(feas, key=lambda r: math.hypot(mid(seed)[0] - mid(r)[0],
                                              mid(seed)[1] - mid(r)[1]))
KS = [2, 3, 4, 5, 6, 8, 10, 12]
LABEL_BUDGET_S = 120.0
MILP_BUDGET_S = 120.0
rows = []

print("=" * 78)
print("ENGINE 1: ESPPRC labeling (exact, mandatory = whole cluster)")
print("=" * 78)
for k in KS:
    cl = tuple(sorted(order[:k]))
    t0 = time.time()
    best, st = solve_labeling(inst, veh, cl, mu={}, time_budget_s=LABEL_BUDGET_S,
                              mandatory_ids=frozenset(cl), label_budget=None, top_k=1)
    w = time.time() - t0
    if best is not None and frozenset(cl).issubset(best.delivered):
        arcs, ch = reconstruct(best)
        elem = not find_subtours(arcs)
        status = f"{st['status']}(served all {k})" + ("" if elem else " NON-ELEMENTARY")
    elif best is None and st["status"] == "INFEASIBLE":
        status = "INFEASIBLE(search exhausted - no elementary route for this SET)"
    else:
        status = f"{st['status']}(no complete route)"
    print(f"  k={k:2d}: {status}  labels={st['labels_created']}  wall={w:.2f}s", flush=True)
    rows.append({"engine": "labeling_ESPPRC", "k": k, "status": status,
                 "labels_created": st["labels_created"], "wall_s": round(w, 2),
                 "log_file": "engine_wall.log"})

print()
print("=" * 78)
print("ENGINE 2: GG-flow single-vehicle MILP (mandatory = whole cluster)")
print("=" * 78)
for k in [2, 3, 4, 5, 6]:
    cl = tuple(sorted(order[:k]))
    reqs = tuple(inst.request_by_id(r) for r in cl)
    solver = HighsSolver(time_limit_s=MILP_BUDGET_S, mip_rel_gap=0.0)
    t0 = time.time()
    try:
        res = solve_routing_subproblem(inst, veh, reqs, solver, multipliers=None,
                                       mandatory_requests=set(cl), max_rounds=1,
                                       wall_budget_s=MILP_BUDGET_S,
                                       use_precedence_separation=True,
                                       use_gg_flow=True, gg_steiner_demand=True)
        w = time.time() - t0
        served = set(res.served_requests)
        if res.subtour_free and res.precedence_clean and set(cl).issubset(served):
            status = f"FEASIBLE(served all {k}, optimal={res.solved_to_optimality})"
        else:
            status = (f"NO_USABLE_SOLUTION(subtour_free={res.subtour_free} "
                      f"prec_clean={res.precedence_clean} served={len(served)})")
    except Exception as e:
        w = time.time() - t0
        status = f"ERROR({type(e).__name__})"
    print(f"  k={k:2d}: {status}  wall={w:.2f}s", flush=True)
    rows.append({"engine": "GGflow_MILP", "k": k, "status": status,
                 "labels_created": "", "wall_s": round(w, 2),
                 "log_file": "engine_wall.log"})

print()
print("=" * 78)
print("ENGINE 3: insertion router (proposed - feasibility, not optimality)")
print("=" * 78)
for k in KS + [15, 20]:
    cl = list(order[:k])
    t0 = time.time()
    served, arcs, charge = route_cluster_insertion(inst, veh, cl, n_candidates=14)
    w = time.time() - t0
    rep = check_solution(inst, {veh.id: VehicleRoute(arcs=arcs, charge=charge,
                                                     served=sorted(served))})
    struct = [v for v in rep.violations if "served by NO vehicle" not in v]
    status = f"SERVED {len(served)}/{k}" + (" checker-clean" if not struct
                                            else f" STRUCT_VIOL={len(struct)}")
    print(f"  k={k:2d}: {status}  wall={w:.2f}s", flush=True)
    rows.append({"engine": "insertion_router", "k": k, "status": status,
                 "labels_created": "", "wall_s": round(w, 2),
                 "log_file": "engine_wall.log"})

json.dump(rows, open("results/scaling/engine_wall_results.json", "w"), indent=2)
print("\nwrote results/scaling/engine_wall_results.json")
