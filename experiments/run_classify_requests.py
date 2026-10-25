from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import argparse, csv, json, math, os, sys, time

from src.model.readers.dataset_reader import load_dataset_instance
from src.solve.insertion_router import route_cluster_insertion
from src.solve.labeling import solve_labeling, reconstruct
from src.solve.subproblem import find_subtours
from src.verify.checker import check_solution, VehicleRoute

# Classifies every request into one of three buckets.
#
#   FEASIBLE     - a checker-validated route serving the request exists, either
#                  harvested from an existing solution (results/coverage/witness_feasible.json)
#                  or constructed here. Requires no trust in any solver.
#   INFEASIBLE   - the labeling search exhausts with no route AND no witness
#                  exists anywhere. Label count and wall time are recorded so a
#                  budget timeout can never be mistaken for a proof.
#   UNDETERMINED - everything else. Undetermined requests count against
#                  coverage; the ceiling is total minus INFEASIBLE.

OUT = "results/csv"
os.makedirs(OUT, exist_ok=True)


def midpoint(inst, rid):
    r = inst.request_by_id(rid)
    p = inst.nodes[r.pickup_node]
    d = inst.nodes[r.delivery_node]
    return ((p.x_km + d.x_km) / 2, (p.y_km + d.y_km) / 2)


def try_witness(inst, rid, budget_s):
    """Constructive witness: insertion router with the request alone, then with
    1-3 nearest companions (a request can be servable in company even when a
    solo attempt fails). Every candidate validated by the independent checker."""
    veh = inst.vehicles[0]
    others = sorted((r.id for r in inst.requests if r.id != rid),
                    key=lambda o: math.hypot(midpoint(inst, rid)[0] - midpoint(inst, o)[0],
                                             midpoint(inst, rid)[1] - midpoint(inst, o)[1]))
    trials = [[rid]] + [[rid] + others[:k] for k in (1, 2, 3)]
    for cl in trials:
        served, arcs, charge = route_cluster_insertion(inst, veh, list(cl), n_candidates=24)
        if rid not in served:
            continue
        rep = check_solution(inst, {veh.id: VehicleRoute(arcs=arcs, charge=charge,
                                                         served=sorted(served))})
        own = [v for v in rep.violations if "served by NO vehicle" not in v]
        if not own:
            return {"how": f"insertion_router(cluster={sorted(cl)})",
                    "served_set": sorted(served), "n_arcs": len(arcs)}
    # last resort: exact labeling with this request mandatory
    best, st = solve_labeling(inst, veh, (rid,), mu={}, time_budget_s=budget_s,
                              mandatory_ids=frozenset({rid}), top_k=1)
    if best is not None and rid in best.delivered:
        arcs, charge = reconstruct(best)
        if not find_subtours(arcs):
            rep = check_solution(inst, {veh.id: VehicleRoute(arcs=arcs, charge=charge,
                                                             served=sorted(best.delivered))})
            own = [v for v in rep.violations if "served by NO vehicle" not in v]
            if not own:
                return {"how": "labeling(mandatory=solo)", "served_set": sorted(best.delivered),
                        "n_arcs": len(arcs)}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, required=True)
    ap.add_argument("--budget-s", type=float, default=120.0)
    args = ap.parse_args()

    inst = load_dataset_instance("data", args.requests)
    ids = [r.id for r in inst.requests]
    harvested = {}
    if os.path.exists("results/coverage/witness_feasible.json"):
        harvested = {int(k): v for k, v in json.load(open("results/coverage/witness_feasible.json")).items()}

    print(f"CLASSIFY R={args.requests}: {len(ids)} requests, "
          f"{sum(1 for i in ids if i in harvested)} already have a harvested witness", flush=True)

    rows = []
    counts = {"FEASIBLE": 0, "INFEASIBLE": 0, "UNDETERMINED": 0}
    for rid in ids:
        if rid in harvested:
            w = harvested[rid]
            rows.append({"request_id": rid, "R_scale": args.requests, "bucket": "FEASIBLE",
                         "evidence": f"harvested witness: {w['log_file']} {w['vehicle']} "
                                     f"served=({w['served_set']})",
                         "labels_created": "", "wall_s": "", "log_file": "witness_scan.log"})
            counts["FEASIBLE"] += 1
            continue
        t0 = time.time()
        w = try_witness(inst, rid, args.budget_s)
        tw = time.time() - t0
        if w is not None:
            rows.append({"request_id": rid, "R_scale": args.requests, "bucket": "FEASIBLE",
                         "evidence": f"constructed witness: {w['how']} served={w['served_set']}",
                         "labels_created": "", "wall_s": round(tw, 2),
                         "log_file": f"classify_R{args.requests}.log"})
            counts["FEASIBLE"] += 1
            print(f"  req {rid}: FEASIBLE (constructed) via {w['how']} [{tw:.1f}s]", flush=True)
            continue
        t1 = time.time()
        best, st = solve_labeling(inst, inst.vehicles[0], (rid,), mu={},
                                  time_budget_s=args.budget_s,
                                  mandatory_ids=frozenset({rid}), top_k=1)
        tl = time.time() - t1
        if best is None and st["status"] == "INFEASIBLE":
            bucket = "INFEASIBLE"
            ev = (f"labeling exhausted search, no route "
                  f"(labels={st['labels_created']}, wall={tl:.2f}s < budget {args.budget_s}s)")
        else:
            bucket = "UNDETERMINED"
            ev = f"no witness; labeling status={st['status']} (labels={st['labels_created']})"
        counts[bucket] += 1
        rows.append({"request_id": rid, "R_scale": args.requests, "bucket": bucket,
                     "evidence": ev, "labels_created": st["labels_created"],
                     "wall_s": round(tl, 2), "log_file": f"classify_R{args.requests}.log"})
        print(f"  req {rid}: {bucket} - {ev}", flush=True)

    total = len(ids)
    ceiling = total - counts["INFEASIBLE"]
    print(f"\n=== R={args.requests} CLASSIFICATION ===")
    print(f"  FEASIBLE     = {counts['FEASIBLE']}")
    print(f"  INFEASIBLE   = {counts['INFEASIBLE']}")
    print(f"  UNDETERMINED = {counts['UNDETERMINED']}")
    print(f"  FEASIBLE CEILING = total - INFEASIBLE = {total} - {counts['INFEASIBLE']} = {ceiling}/{total}")
    print(f"  (UNDETERMINED counts AGAINST coverage, stays in the denominator)")

    with open(f"{OUT}/classification_R{args.requests}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["request_id", "R_scale", "bucket", "evidence",
                                          "labels_created", "wall_s", "log_file"])
        w.writeheader()
        w.writerows(rows)
    json.dump({"R": args.requests, "counts": counts, "ceiling": ceiling, "total": total,
               "infeasible_ids": [r["request_id"] for r in rows if r["bucket"] == "INFEASIBLE"],
               "undetermined_ids": [r["request_id"] for r in rows if r["bucket"] == "UNDETERMINED"]},
              open(f"results/coverage/classification_R{args.requests}.json", "w"), indent=2)
    print(f"\nwrote results/coverage/classification_R{args.requests}.json "
          f"and {OUT}/classification_R{args.requests}.csv")


if __name__ == "__main__":
    main()
