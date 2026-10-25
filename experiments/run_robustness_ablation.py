from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import csv, dataclasses, json, math, os, random, sys, time

from src.model.readers.dataset_reader import load_dataset_instance
from src.solve.cluster_route import solve_cluster_route, to_vehicle_routes, _route_cost
from src.solve.insertion_router import try_add_one
from src.verify.checker import check_solution, VehicleRoute

# Two robustness experiments:
#   1. a five-instance sweep over independently sampled 100-request subsets
#   2. a cheap single-pass greedy compared against the full method
#
# Every request is offered to the method and every unserved request counts
# against coverage.

OUT = "results/csv"
os.makedirs(OUT, exist_ok=True)
SEEDS = [11, 22, 33, 44, 55]


# ---------------------------------------------------------------- B1
def b1():
    base = load_dataset_instance("data", 300)
    all_ids = [r.id for r in base.requests]
    rng = random.Random(SEEDS[0])
    shuffled = list(all_ids)
    rng.shuffle(shuffled)
    subsets = {
        "I1_disjoint_a": sorted(shuffled[0:100]),
        "I2_disjoint_b": sorted(shuffled[100:200]),
        "I3_disjoint_c": sorted(shuffled[200:300]),
    }
    for s in SEEDS[3:]:
        r2 = random.Random(s)
        subsets[f"I{len(subsets)+1}_seed{s}"] = sorted(r2.sample(all_ids, 100))

    print("Multi-instance robustness", flush=True)
    print("  feasibility classification: 0 proven-infeasible, so ceiling = 100/100", flush=True)
    print("  every request offered; unserved counts AGAINST coverage", flush=True)
    rows = []
    for name, ids in subsets.items():
        inst = dataclasses.replace(base, requests=tuple(r for r in base.requests if r.id in ids))
        t0 = time.time()
        res = solve_cluster_route(inst, engine="insertion", n_candidates=14,
                                  reinsert_passes=3, precomputed_solo=(ids, [], []))
        wall = time.time() - t0
        rep = check_solution(inst, to_vehicle_routes(res.per_vehicle))
        struct = [v for v in rep.violations if "served by NO vehicle" not in v]
        served = len(res.served)
        unserved = sorted(set(ids) - set(res.served))
        cost = sum(_route_cost(inst, pl.arcs, pl.charge) for pl in res.per_vehicle.values())
        print(f"  {name}: served={served}/100  struct_viol={len(struct)}  "
              f"wall={wall:.1f}s  cost={cost:.2f}  unserved={unserved}", flush=True)
        rows.append({"instance": name, "n_requests": len(ids), "feasible_ceiling": len(ids),
                     "served": served, "structural_violations": len(struct),
                     "solve_wall_s": round(wall, 2), "routing_cost": round(cost, 4),
                     "coverage_pct": round(100.0 * served / len(ids), 2),
                     "unserved_ids": ";".join(map(str, unserved)),
                     "log_file": "robustness_ablation.log"})
    cov = [r["coverage_pct"] for r in rows]
    walls = [r["solve_wall_s"] for r in rows]
    print(f"\n  coverage: mean={sum(cov)/len(cov):.2f}%  min={min(cov):.2f}%  max={max(cov):.2f}%")
    print(f"  wall:     mean={sum(walls)/len(walls):.1f}s  min={min(walls):.1f}s  max={max(walls):.1f}s")
    print(f"  total structural violations across all 5: "
          f"{sum(r['structural_violations'] for r in rows)}")
    with open(f"{OUT}/multi_instance_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


# ---------------------------------------------------------------- B2
def midpoint(inst, rid):
    r = inst.request_by_id(rid)
    p = inst.nodes[r.pickup_node]; d = inst.nodes[r.delivery_node]
    return ((p.x_km + d.x_km) / 2, (p.y_km + d.y_km) / 2)


def cheap_greedy(inst, placeable, n_candidates=14):
    """Single pass, first-fit, no restarts, no reinsertion, no clustering -
    same splice primitive as the full method."""
    vehicles = list(inst.vehicles)
    seqs = {v.id: [v.start_node, v.end_node] for v in vehicles}
    served = {v.id: [] for v in vehicles}
    arcs_of = {v.id: [] for v in vehicles}
    charge_of = {v.id: {} for v in vehicles}
    order = sorted(placeable, key=lambda r: midpoint(inst, r)[0] ** 2 + midpoint(inst, r)[1] ** 2)
    for rid in order:
        m = midpoint(inst, rid)
        def vkey(v):
            if served[v.id]:
                cx = sum(midpoint(inst, r)[0] for r in served[v.id]) / len(served[v.id])
                cy = sum(midpoint(inst, r)[1] for r in served[v.id]) / len(served[v.id])
            else:
                s = inst.nodes[v.start_node]; cx, cy = s.x_km, s.y_km
            return math.hypot(m[0] - cx, m[1] - cy)
        for v in sorted(vehicles, key=vkey):
            res = try_add_one(inst, v, seqs[v.id], served[v.id], rid, n_candidates=n_candidates)
            if res is not None:
                ns, arcs, charge = res
                seqs[v.id] = ns; served[v.id] = sorted(served[v.id] + [rid])
                arcs_of[v.id] = arcs; charge_of[v.id] = charge
                break
    routes = {v.id: VehicleRoute(arcs=arcs_of[v.id], charge=charge_of[v.id],
                                 served=served[v.id]) for v in vehicles}
    cost = sum(_route_cost(inst, arcs_of[v.id], charge_of[v.id]) for v in vehicles)
    return routes, served, cost


def b2():
    print("\nCheap greedy versus full method", flush=True)
    rows = []
    for R in (30, 100):
        inst = load_dataset_instance("data", R)
        ids = [r.id for r in inst.requests]

        t0 = time.time()
        routes, served, cost = cheap_greedy(inst, ids)
        w_cheap = time.time() - t0
        rep = check_solution(inst, routes)
        struct = [v for v in rep.violations if "served by NO vehicle" not in v]
        n_cheap = sum(len(s) for s in served.values())
        print(f"  R={R} cheap greedy : served={n_cheap}/{len(ids)} struct={len(struct)} "
              f"wall={w_cheap:.2f}s cost={cost:.4f}", flush=True)
        rows.append({"R": R, "method": "greedy (cheap, single-pass)", "served": n_cheap,
                     "feasible_ceiling": len(ids), "structural_violations": len(struct),
                     "wall_time_s": round(w_cheap, 2), "routing_cost": round(cost, 4),
                     "log_file": "robustness_ablation.log"})

        t1 = time.time()
        res = solve_cluster_route(inst, engine="insertion", n_candidates=14,
                                  reinsert_passes=3, precomputed_solo=(ids, [], []))
        w_full = time.time() - t1
        rep2 = check_solution(inst, to_vehicle_routes(res.per_vehicle))
        struct2 = [v for v in rep2.violations if "served by NO vehicle" not in v]
        cost2 = sum(_route_cost(inst, pl.arcs, pl.charge) for pl in res.per_vehicle.values())
        print(f"  R={R} full method  : served={len(res.served)}/{len(ids)} struct={len(struct2)} "
              f"wall={w_full:.2f}s cost={cost2:.4f}", flush=True)
        rows.append({"R": R, "method": "cluster-first + insertion (full)", "served": len(res.served),
                     "feasible_ceiling": len(ids), "structural_violations": len(struct2),
                     "wall_time_s": round(w_full, 2), "routing_cost": round(cost2, 4),
                     "log_file": "robustness_ablation.log"})

        tie = (n_cheap == len(res.served))
        gap = 100.0 * (cost - cost2) / cost2 if cost2 else 0.0
        print(f"    coverage tie holds: {tie}   cheap cost is {gap:+.2f}% vs full", flush=True)

    with open(f"{OUT}/cheap_greedy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    r1 = b1()
    r2 = b2()
    json.dump({"b1": r1, "b2": r2}, open("results/scaling/robustness_ablation_results.json", "w"), indent=2)
    print("\nwrote results/scaling/robustness_ablation_results.json", flush=True)
