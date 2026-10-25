from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import csv, itertools, json, logging, os, sys

from src.solve.cluster_route import _balanced_spatial_clusters, _route_cost
from src.solve.insertion_router import route_cluster_insertion
from src.solve.labeling import solve_labeling
from src.verify.checker import VehicleRoute, check_solution
from tests.mini_instance import build_mini_instance

# Optimality gap of the insertion router against an exact reference.
#
# The router is a feasibility heuristic and produces no dual bound, so a gap
# can only be quoted where an exact reference optimum is genuinely computable.
# On the small hand-built instance the reference optimum is obtained by
# enumerating every assignment of requests to vehicles and solving each
# vehicle's route to proven optimality, so the comparison is exact.
#
# A reference that does not terminate with proven optimality is reported as
# UNDETERMINED and never converted into a gap.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)
OUT = "results/csv"
os.makedirs(OUT, exist_ok=True)
os.makedirs("results/scaling", exist_ok=True)


def exact_optimum(inst, budget_s=60.0):
    """Brute-force fleet optimum: enumerate assignments, solve each vehicle's
    route to proven optimality. Returns (cost, status)."""
    rids = [r.id for r in inst.requests]
    best = None
    for assign in itertools.product(range(len(inst.vehicles)), repeat=len(rids)):
        total, ok = 0.0, True
        for k, veh in enumerate(inst.vehicles):
            mand = frozenset(rids[i] for i in range(len(rids)) if assign[i] == k)
            lab, st = solve_labeling(inst, veh, tuple(rids), mu={},
                                     time_budget_s=budget_s,
                                     mandatory_ids=mand, top_k=1)
            if lab is None or st["status"] != "OPTIMAL":
                ok = False
                break
            total += lab.cost
        if ok and (best is None or total < best):
            best = total
    if best is None:
        return None, "UNDETERMINED (no assignment proved optimal in budget)"
    return best, "OPTIMAL"


def main():
    inst = build_mini_instance(2)
    rids = [r.id for r in inst.requests]

    exact, status = exact_optimum(inst)

    clusters = _balanced_spatial_clusters(inst, rids, len(inst.vehicles))
    total, served_all, routes = 0.0, [], {}
    for veh, cl in zip(inst.vehicles, clusters):
        served, arcs, charge = route_cluster_insertion(inst, veh, list(cl),
                                                       n_candidates=24)
        total += _route_cost(inst, arcs, charge)
        served_all.extend(served)
        routes[veh.id] = VehicleRoute(arcs=arcs, charge=charge, served=sorted(served))
    rep = check_solution(inst, routes)
    struct = [v for v in rep.violations if "served by NO vehicle" not in v]
    full = set(served_all) == set(rids)

    if exact is not None and full and not struct:
        gap = 100.0 * (total - exact) / exact
        print(f"exact optimum   = {exact:.6f}  ({status})")
        print(f"insertion cost  = {total:.6f}  served={len(served_all)}/{len(rids)}"
              f"  structural_violations={len(struct)}")
        print(f"GAP             = {gap:+.2f}%")
    else:
        gap = None
        print(f"exact optimum   = {exact} ({status})")
        print(f"insertion cost  = {total:.6f} served={len(served_all)}/{len(rids)}"
              f" full_coverage={full} structural_violations={len(struct)}")
        print("GAP             = UNDETERMINED")

    row = {"instance": "mini_2veh_2req", "n_requests": len(rids),
           "exact_method": "assignment enumeration + exact labeling",
           "exact_cost": (round(exact, 6) if exact is not None else ""),
           "insertion_cost": round(total, 6),
           "gap_percent": (round(gap, 2) if gap is not None else ""),
           "exact_status": status,
           "structural_violations": len(struct),
           "log_file": "optimality_gap.log"}
    with open(os.path.join(OUT, "optimality_gap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    json.dump([row], open("results/scaling/optimality_gap_results.json", "w"), indent=2)
    print(f"\nwrote {OUT}/optimality_gap.csv")


if __name__ == "__main__":
    main()
