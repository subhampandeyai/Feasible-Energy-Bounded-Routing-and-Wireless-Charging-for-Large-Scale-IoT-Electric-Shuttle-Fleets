from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import csv, dataclasses, json, os, random, sys, time

from src.model.readers.dataset_reader import load_dataset_instance
from src.model.types import ChargingConfig, ChargingSite
from src.solve.cluster_route import solve_cluster_route, to_vehicle_routes, _route_cost
from src.verify.checker import check_solution

# Charger-availability experiment.
#
# Three charger-availability regimes on the same graph and the same fleet.
# Availability is changed only through the data model's own switch,
# ChargingSite.is_enabled, which ChargingConfig.node_enabled reads and from
# which ChargingConfig.arc_enabled derives arc availability. The router itself
# is not modified.
#
# Note that an arc is charging-capable only when BOTH of its endpoints are
# enabled, so arc availability falls quadratically as nodes are disabled.

SEED = 20260727           # fixed, stated, reproducible
REGIMES = [("dense", 1.00), ("medium", 0.30), ("sparse", 0.10)]
R = 300
KNOBS = (24, 5)           # --n-candidates 24 --reinsert-passes 5 at R=300
OUT_CSV = "results/csv/charging_regimes.csv"
os.makedirs("results/csv", exist_ok=True)
os.makedirs("results/charging", exist_ok=True)


def make_regime(base, frac, seed):
    """Return (instance, enabled_node_ids). Disables stations via
    ChargingSite.is_enabled only. Depot start/end nodes stay enabled."""
    depots = {v.start_node for v in base.vehicles} | {v.end_node for v in base.vehicles}
    all_nodes = sorted(base.nodes)
    if frac >= 1.0:
        enabled = set(all_nodes)
    else:
        k = max(len(depots), int(round(frac * len(all_nodes))))
        pool = [n for n in all_nodes if n not in depots]
        rng = random.Random(seed)
        enabled = set(depots) | set(rng.sample(pool, k - len(depots)))
    sites = {}
    for nid, site in base.charging.sites_by_node.items():
        sites[nid] = dataclasses.replace(site, is_enabled=(nid in enabled))
    cfg = ChargingConfig(sites_by_node=sites,
                         electrified_arc_fraction=base.charging.electrified_arc_fraction)
    return dataclasses.replace(base, charging=cfg), sorted(enabled)


def main():
    base = load_dataset_instance("data", R)
    ids = [r.id for r in base.requests]
    nc, pp = KNOBS
    rows = []
    served_sets = {}

    print(f"Charger availability experiment, R={R}, seed={SEED}", flush=True)
    print(f"  availability changed ONLY via ChargingSite.is_enabled; router untouched", flush=True)
    print(f"  amount is still the CHARGE_LEVELS constant; trigger still reactive soc-need<0",
          flush=True)

    for name, frac in REGIMES:
        inst, enabled = make_regime(base, frac, SEED)
        n_nodes_on = sum(1 for n in inst.nodes if inst.charging.node_enabled(n))
        n_arcs_on = sum(1 for a in inst.arcs if inst.charging.arc_enabled(a.from_node, a.to_node))
        json.dump({"regime": name, "target_fraction": frac, "seed": SEED,
                   "enabled_nodes": enabled},
                  open(f"results/charging/charger_regime_{name}_nodes.json", "w"))

        t0 = time.time()
        res = solve_cluster_route(inst, engine="insertion", n_candidates=nc,
                                  reinsert_passes=pp, precomputed_solo=(ids, [], []))
        wall = time.time() - t0
        routes = to_vehicle_routes(res.per_vehicle)
        rep = check_solution(inst, routes)
        struct = [v for v in rep.violations if "served by NO vehicle" not in v]
        served = sorted(res.served)
        served_sets[name] = set(served)
        cost = sum(_route_cost(inst, pl.arcs, pl.charge) for pl in res.per_vehicle.values())

        # charging events on the REAL routes
        ev = []
        for vid, pl in res.per_vehicle.items():
            for (i, j), amt in pl.charge.items():
                if amt > 1e-9 and inst.charging.arc_enabled(i, j):
                    ev.append((vid, i, j, amt, inst.prices.base_price_per_node.get(i, 0.0)))
        amounts = sorted({e[3] for e in ev})
        prices = [e[4] for e in ev]
        nodes_used = sorted({e[1] for e in ev})
        vehicles_charging = {e[0] for e in ev}

        # served requests whose serving vehicle charged at least once
        req_needs_charge = 0
        for vid, pl in res.per_vehicle.items():
            if vid in vehicles_charging:
                req_needs_charge += len(pl.served)

        print(f"\n--- regime={name} (target {frac:.0%}) ---", flush=True)
        print(f"  chargers: {n_nodes_on}/{len(inst.nodes)} nodes enabled, "
              f"{n_arcs_on}/{len(inst.arcs)} arcs enabled", flush=True)
        print(f"  CHECKER: served={len(served)}/{len(ids)}  structural_violations={len(struct)}"
              f"  wall={wall:.2f}s  cost={cost:.4f}", flush=True)
        for v in struct[:5]:
            print(f"    STRUCT: {v}", flush=True)
        print(f"  charging events={len(ev)}  distinct nodes used={len(nodes_used)}  "
              f"distinct amounts={amounts}  "
              f"price range={(f'{min(prices):.3f}-{max(prices):.3f}' if prices else 'n/a')}",
              flush=True)
        print(f"  requests served by a vehicle that charged at least once: "
              f"{req_needs_charge}/{len(served)}", flush=True)

        rows.append({
            "regime": name, "target_fraction": frac, "seed": SEED, "R": R,
            "nodes_charging_enabled": n_nodes_on, "nodes_total": len(inst.nodes),
            "arcs_charging_enabled": n_arcs_on, "arcs_total": len(inst.arcs),
            "served": len(served), "total_requests": len(ids),
            "structural_violations": len(struct),
            "wall_time_s": round(wall, 2), "routing_cost": round(cost, 4),
            "charging_events": len(ev), "distinct_charging_nodes": len(nodes_used),
            "distinct_amounts_kWh": ";".join(str(a) for a in amounts),
            "price_min": (round(min(prices), 4) if prices else ""),
            "price_max": (round(max(prices), 4) if prices else ""),
            "requests_on_charging_vehicles": req_needs_charge,
            "unserved_ids": ";".join(str(u) for u in sorted(set(ids) - set(served))),
            "log_file": "charging_sparsity_experiment.log",
        })

    # feasibility drop vs the dense control
    print("\n=== FEASIBILITY DROP vs dense control ===", flush=True)
    dense = served_sets["dense"]
    for name, _ in REGIMES:
        if name == "dense":
            print(f"  dense: {len(dense)}/{len(ids)} (control)", flush=True)
            continue
        s = served_sets[name]
        lost = sorted(dense - s)
        gained = sorted(s - dense)
        print(f"  {name}: {len(s)}/{len(ids)}  delta={len(s)-len(dense):+d}", flush=True)
        print(f"     lost vs dense ({len(lost)}): {lost}", flush=True)
        if gained:
            print(f"     gained vs dense ({len(gained)}): {gained}", flush=True)
        for r in rows:
            if r["regime"] == name:
                r["coverage_delta_vs_dense"] = len(s) - len(dense)
                r["lost_vs_dense_ids"] = ";".join(map(str, lost))
    for r in rows:
        r.setdefault("coverage_delta_vs_dense", 0)
        r.setdefault("lost_vs_dense_ids", "")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open("results/charging/charging_regimes_results.json", "w"), indent=2)
    print(f"\nwrote {OUT_CSV} and results/charging/charging_regimes_results.json", flush=True)

    allclean = all(r["structural_violations"] == 0 for r in rows)
    print(f"\nAll regimes checker-clean (0 structural violations): {allclean}", flush=True)


if __name__ == "__main__":
    main()
