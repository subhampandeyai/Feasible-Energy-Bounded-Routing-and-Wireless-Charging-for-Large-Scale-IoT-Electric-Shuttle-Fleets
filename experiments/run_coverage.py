from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import csv, json, os, sys, time

from src.model.readers.dataset_reader import load_dataset_instance
from src.solve.cluster_route import solve_cluster_route, to_vehicle_routes, _route_cost
from src.verify.checker import check_solution

# Computes headline coverage for each request scale.
#
# Every request is offered to the method, and every unserved request counts
# against coverage: nothing is removed from the denominator.

OUT = "results/csv"
os.makedirs(OUT, exist_ok=True)
rows = []

for R, nc, pp in ((30, 14, 3), (100, 14, 3), (200, 14, 3), (300, 24, 5)):
    cls = json.load(open(f"results/coverage/classification_R{R}.json"))
    inst = load_dataset_instance("data", R)
    ids = [r.id for r in inst.requests]
    infeasible = set(cls["infeasible_ids"])
    undetermined = set(cls["undetermined_ids"])
    # everything not proven infeasible is offered to the method
    feasible = [i for i in ids if i not in infeasible]
    ceiling = len(ids) - len(infeasible)

    t0 = time.time()
    res = solve_cluster_route(inst, engine="insertion", n_candidates=nc,
                              reinsert_passes=pp,
                              precomputed_solo=(feasible, sorted(infeasible), []))
    wall = time.time() - t0
    routes = to_vehicle_routes(res.per_vehicle)
    rep = check_solution(inst, routes)
    struct = [v for v in rep.violations if "served by NO vehicle" not in v]
    served = len(res.served)
    cost = sum(_route_cost(inst, pl.arcs, pl.charge) for pl in res.per_vehicle.values())
    unserved = sorted(set(ids) - set(res.served))

    print(f"R={R}: served={served}/{len(ids)}  feasibility ceilings={ceiling}  "
          f"struct_viol={len(struct)}  wall={wall:.2f}s  cost={cost:.4f}", flush=True)
    print(f"      unserved ({len(unserved)}): {unserved}", flush=True)
    print(f"      of those, UNDETERMINED-feasibility: "
          f"{sorted(set(unserved) & undetermined)}", flush=True)
    rows.append({"R": R, "served": served, "feasible_ceiling": ceiling,
                 "total_requests": len(ids), "proven_infeasible": len(infeasible),
                 "undetermined": len(undetermined),
                 "structural_violations": len(struct), "wall_time_s": round(wall, 2),
                 "routing_cost": round(cost, 4),
                 "coverage_pct_of_total": round(100.0 * served / len(ids), 2),
                 "unserved_ids": ";".join(str(u) for u in unserved),
                 "log_file": "coverage.log"})

with open(f"{OUT}/coverage.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
json.dump(rows, open("results/coverage/coverage_results.json", "w"), indent=2)
print(f"\nwrote {OUT}/coverage.csv")

print("\n=== COVERAGE SUMMARY ===")
for r in rows:
    print(f"R={r['R']}: {r['served']}/{r['feasible_ceiling']} served "
          f"({r['coverage_pct_of_total']}% of all {r['total_requests']}); "
          f"proven-infeasible {r['proven_infeasible']}, "
          f"undetermined {r['undetermined']}")
