from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import csv, dataclasses, json, os, sys, time

from src.model.readers.dataset_reader import load_dataset_instance
from src.solve.cluster_route import solve_cluster_route, to_vehicle_routes, _route_cost
from src.verify.checker import check_solution

# Fleet-size scalability: K vehicles at a fixed request count.
#
# Vehicles are cloned from the dataset's own vehicle template, so the fleet
# stays homogeneous and only its size changes. A larger fleet also carries more
# battery stock, so coverage is expected to rise toward the energy bound.

OUT = "results/csv"
os.makedirs(OUT, exist_ok=True)
R = 300
KNOBS = (24, 5)


def with_k_vehicles(base, K):
    v0 = base.vehicles[0]
    vs = tuple(dataclasses.replace(v0, id=f"veh_{i+1}") for i in range(K))
    return dataclasses.replace(base, vehicles=vs)


def main():
    base = load_dataset_instance("data", R)
    ids = [r.id for r in base.requests]
    cons = base.vehicles[0].max_energy_consumption_kWh_per_km
    cap = base.vehicles[0].battery_capacity_kWh
    nc, pp = KNOBS
    Ks = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                           else ["10", "20", "50", "100"])]
    rows = []

    print(f"FLEET SCALABILITY at R={R} (knobs n_candidates={nc}, reinsert_passes={pp})",
          flush=True)
    print(f"  vehicles cloned from the dataset template: {cap} kWh, "
          f"{cons:.5f} kWh/km, range {cap/cons:.0f} km", flush=True)

    for K in Ks:
        inst = with_k_vehicles(base, K)
        t0 = time.time()
        res = solve_cluster_route(inst, engine="insertion", n_candidates=nc,
                                  reinsert_passes=pp, precomputed_solo=(ids, [], []))
        wall = time.time() - t0
        rep = check_solution(inst, to_vehicle_routes(res.per_vehicle))
        struct = [v for v in rep.violations if "served by NO vehicle" not in v]
        served = len(res.served)
        cost = sum(_route_cost(inst, pl.arcs, pl.charge) for pl in res.per_vehicle.values())
        used = sum(1 for pl in res.per_vehicle.values() if pl.served)
        km = 0.0
        arcdist = {(a.from_node, a.to_node): a.distance_km for a in inst.arcs}
        for pl in res.per_vehicle.values():
            km += sum(arcdist.get(a, 0.0) for a in pl.arcs)
        battery_only_km = K * cap / cons

        print(f"\n  K={K:3d}: served={served}/{len(ids)}  struct_viol={len(struct)}  "
              f"wall={wall:.1f}s  cost={cost:.2f}", flush=True)
        print(f"        vehicles used={used}/{K}  km driven={km:.1f}  "
              f"battery-only fleet range={battery_only_km:.0f} km  "
              f"ratio={km/battery_only_km:.2f}x", flush=True)
        for v in struct[:3]:
            print(f"        STRUCT: {v}", flush=True)

        rows.append({
            "K_vehicles": K, "R": R, "served": served, "total_requests": len(ids),
            "coverage_pct": round(100.0 * served / len(ids), 2),
            "structural_violations": len(struct),
            "vehicles_used": used, "wall_time_s": round(wall, 2),
            "routing_cost": round(cost, 4), "km_driven": round(km, 1),
            "battery_only_fleet_range_km": round(battery_only_km, 1),
            "km_over_battery_only": round(km / battery_only_km, 3),
            "unserved_ids": ";".join(str(u) for u in sorted(set(ids) - set(res.served))),
            "log_file": "fleet_scaling.log",
        })
        json.dump(rows, open("results/scaling/fleet_scaling_results.json", "w"), indent=2)
        with open(f"{OUT}/fleet_scaling.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    print(f"\nwrote {OUT}/fleet_scaling.csv", flush=True)


if __name__ == "__main__":
    main()
