from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import csv, dataclasses, json, os, random, sys, time

from src.model.readers.dataset_reader import load_dataset_instance
from src.solve.cluster_route import solve_cluster_route, to_vehicle_routes, solo_feasibility
from src.verify.checker import check_solution

# Multi-instance robustness.
#
# Builds several independent request subsets of equal size from the largest
# request file, on the same graph and fleet, and runs the full pipeline on each
# with default parameters. Seeds are fixed so the subsets reproduce exactly.

SEEDS = [11, 22, 33, 44, 55]
OUT = "results/csv"
os.makedirs(OUT, exist_ok=True)

base = load_dataset_instance("data", 300)
all_ids = [r.id for r in base.requests]

# 3 disjoint partitions from a single seeded shuffle
rng = random.Random(SEEDS[0])
shuffled = list(all_ids)
rng.shuffle(shuffled)
subsets = {
    "I1_disjoint_a": sorted(shuffled[0:100]),
    "I2_disjoint_b": sorted(shuffled[100:200]),
    "I3_disjoint_c": sorted(shuffled[200:300]),
}
# 2 further independent random subsets (overlap with the above is unavoidable)
for s in SEEDS[3:]:
    r2 = random.Random(s)
    subsets[f"I{len(subsets)+1}_seed{s}"] = sorted(r2.sample(all_ids, 100))

print("EXPERIMENT 1 - MULTI-INSTANCE ROBUSTNESS", flush=True)
print(f"5 instances of 100 requests each, sampled from the 300-request file.", flush=True)
print(f"  I1/I2/I3 are mutually DISJOINT (a seeded shuffle of all 300 split into thirds, seed={SEEDS[0]})", flush=True)
print(f"  I4/I5 are independent random subsets (seeds {SEEDS[3]}, {SEEDS[4]}) and DO overlap the others", flush=True)
print(f"  knobs: DEFAULT (n_candidates=14, reinsert_passes=3)", flush=True)

rows = []
for name, ids in subsets.items():
    inst = dataclasses.replace(base, requests=tuple(r for r in base.requests if r.id in ids))
    t0 = time.time()
    feas, infeas, undet = solo_feasibility(inst, inst.vehicles[0], 8.0)
    t_solo = time.time() - t0
    t1 = time.time()
    res = solve_cluster_route(inst, engine="insertion", n_candidates=14,
                              reinsert_passes=3,
                              precomputed_solo=(feas, infeas, undet))
    t_solve = time.time() - t1
    routes = to_vehicle_routes(res.per_vehicle)
    rep = check_solution(inst, routes)
    struct = [v for v in rep.violations if "served by NO vehicle" not in v]
    ceiling = len(ids) - len(infeas)
    served = len(res.served)
    full = (served == ceiling) and not struct
    print(f"\n{name}: ceiling={ceiling}/{len(ids)} (proven_infeasible={len(infeas)}, "
          f"undetermined={len(undet)})", flush=True)
    print(f"  served={served}  struct_viol={len(struct)}  "
          f"solo={t_solo:.1f}s solve={t_solve:.1f}s  FULL_CEILING={full}", flush=True)
    if not full:
        print(f"  NOT AT CEILING - unplaced feasible: {list(res.unserved_feasible)}", flush=True)
    rows.append({"instance": name, "n_requests": len(ids), "feasible_ceiling": ceiling,
                 "served": served, "proven_infeasible": len(infeas),
                 "undetermined": len(undet), "structural_violations": len(struct),
                 "solo_wall_s": round(t_solo, 2), "solve_wall_s": round(t_solve, 2),
                 "reached_full_ceiling": full, "log_file": "multi_instance.log"})

print("\n" + "=" * 66)
print("ACROSS THE 5 INSTANCES")
print("=" * 66)
ratios = [r["served"] / r["feasible_ceiling"] for r in rows]
walls = [r["solve_wall_s"] for r in rows]
nfull = sum(1 for r in rows if r["reached_full_ceiling"])
print(f"  coverage vs ceiling: mean={sum(ratios)/len(ratios)*100:.2f}%  "
      f"min={min(ratios)*100:.2f}%  max={max(ratios)*100:.2f}%")
print(f"  solve wall time:     mean={sum(walls)/len(walls):.1f}s  "
      f"min={min(walls):.1f}s  max={max(walls):.1f}s")
print(f"  reached FULL ceiling: {nfull}/5")
print(f"  total structural violations across all 5: {sum(r['structural_violations'] for r in rows)}")

with open(f"{OUT}/multi_instance.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
json.dump(rows, open("results/scaling/multi_instance_results.json", "w"), indent=2)
print(f"\nwrote {OUT}/multi_instance.csv and results/scaling/multi_instance_results.json")
