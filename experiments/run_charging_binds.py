from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import csv, json

from src.model.readers.dataset_reader import load_dataset_instance

# Does charging bind on the R=300 solution? For each vehicle's checker-verified
# route (results/geometry/solution_geometry_R300.json, written by
# run_geometry_export.py - run it first), replay the energy profile from the raw
# data twice:
#   1. with NO charging: at what distance does SOC first reach zero, and what
#      would SOC be at the end of the route;
#   2. with the route's own charging events (the reactive rule): the minimum SOC
#      along the route.
# Distances and consumption come from data/, not from the geometry file; the
# exported SOC profile is only used as a cross-check of replay 2.

R = 300
GEOMETRY = f"results/geometry/solution_geometry_R{R}.json"
OUT_DIR = "results/charging"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("results/csv", exist_ok=True)


def main():
    if not os.path.exists(GEOMETRY):
        raise SystemExit(f"{GEOMETRY} not found - run experiments/run_geometry_export.py first")
    inst = load_dataset_instance("data", R)
    arcdist = {(a.from_node, a.to_node): a.distance_km for a in inst.arcs}
    veh_by_id = {v.id: v for v in inst.vehicles}
    g = json.load(open(GEOMETRY))

    charge_by_vehicle = {}
    for ev in g["charging_events"]:
        charge_by_vehicle.setdefault(ev["vehicle"], {})[tuple(ev["arc"])] = ev["amount_kWh"]

    rows = []
    print(f"CHARGING NECESSITY on the R={R} solution ({g['meta']['served']}/{g['meta']['total']} served)")
    print(f"  {'vehicle':8s} {'route km':>9s} {'empty at km':>12s} {'end SOC, no charging':>21s} "
          f"{'min SOC, reactive':>18s}")
    for vid in sorted(g["routes"], key=lambda v: int(v.split("_")[1])):
        seq = g["routes"][vid]["node_sequence"]
        veh = veh_by_id[vid]
        cons = veh.max_energy_consumption_kWh_per_km or inst.energy.consumption_kWh_per_km
        charge = charge_by_vehicle.get(vid, {})

        soc_nc, soc_ch, km, empty_at = veh.initial_soc_kWh, veh.initial_soc_kWh, 0.0, None
        min_ch = soc_ch
        for a, b in zip(seq, seq[1:]):
            d = arcdist[(a, b)]
            e = cons * d
            if empty_at is None and soc_nc - e < 0:
                empty_at = km + soc_nc / cons          # exact point on the arc where SOC hits 0
            soc_nc -= e
            eta = 1.0 if inst.charging.arc_enabled(a, b) else 0.0
            soc_ch = soc_ch + eta * charge.get((a, b), 0.0) - e
            min_ch = min(min_ch, soc_ch)
            km += d

        exported = min(p["soc_kWh"] for p in g["soc_profile"][vid])
        if abs(exported - min_ch) > 1e-4:
            raise SystemExit(f"{vid}: replayed min SOC {min_ch:.6f} != exported {exported:.6f}")
        print(f"  {vid:8s} {km:9.1f} {('%.1f' % empty_at) if empty_at is not None else 'never':>12s} "
              f"{soc_nc:21.2f} {min_ch:18.4f}")
        rows.append({"vehicle": vid, "route_km": round(km, 3),
                     "km_at_empty_without_charging": round(empty_at, 3) if empty_at is not None else "",
                     "end_soc_without_charging_kWh": round(soc_nc, 3),
                     "min_soc_reactive_kWh": round(min_ch, 6),
                     "charging_events": len(charge)})

    binds = [r for r in rows if r["km_at_empty_without_charging"] != ""]
    ends = [r["end_soc_without_charging_kWh"] for r in rows]
    empties = [r["km_at_empty_without_charging"] for r in binds]
    print()
    print(f"  charging binds for {len(binds)}/{len(rows)} vehicles")
    if empties:
        print(f"  without charging, SOC reaches zero at {min(empties):.1f} to {max(empties):.1f} km")
    print(f"  end-of-route SOC without charging: {min(ends):.2f} to {max(ends):.2f} kWh")
    print(f"  fleet minimum SOC under the reactive rule: {min(r['min_soc_reactive_kWh'] for r in rows):.6f} kWh")

    with open("results/csv/charging_binds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump(rows, open(f"{OUT_DIR}/charging_binds.json", "w"), indent=2)
    print("\nwrote results/csv/charging_binds.csv")


if __name__ == "__main__":
    main()
