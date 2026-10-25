from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import json, os, sys, time

from src.model.readers.dataset_reader import load_dataset_instance
from src.solve.cluster_route import solve_cluster_route, to_vehicle_routes, _route_cost
from src.verify.checker import check_solution

# Exports solved routes as plain JSON for independent plotting and animation.
#
# Two gates are enforced before any file is written:
#   1. every route is validated with src/verify/check_solution
#   2. served counts and unserved ids must match the recorded coverage results
# Either gate failing aborts that scale without writing a file.
#
# No interpolation, no smoothing and no synthetic timestamps: the model has no
# time windows, so no arrival times are invented. Coordinates are the dataset's
# own, not re-projected, re-scaled or re-centred.

KNOBS = {30: (14, 3), 100: (14, 3), 200: (14, 3), 300: (24, 5)}
EXPECTED = {r["R"]: r for r in json.load(open("results/coverage/coverage_results.json"))}


def node_sequence_from_arcs(veh, arcs):
    """Walk the route from the start node, returning every intermediate node."""
    if not arcs:
        return []
    succ = {i: j for (i, j) in arcs}
    seq = [veh.start_node]
    cur = veh.start_node
    seen = {cur}
    while cur in succ:
        cur = succ[cur]
        if cur in seen:            # cycle - checker would already have failed it
            break
        seq.append(cur)
        seen.add(cur)
    return seq


def export(R, logf):
    nc, pp = KNOBS[R]
    inst = load_dataset_instance("data", R)
    ids = [r.id for r in inst.requests]
    exp = EXPECTED[R]

    t0 = time.time()
    res = solve_cluster_route(inst, engine="insertion", n_candidates=nc,
                              reinsert_passes=pp,
                              precomputed_solo=(ids, [], []))
    wall = time.time() - t0

    routes = to_vehicle_routes(res.per_vehicle)
    rep = check_solution(inst, routes)
    struct = [v for v in rep.violations if "served by NO vehicle" not in v]
    served_ids = sorted(res.served)
    unserved_ids = sorted(set(ids) - set(served_ids))

    # ---- GATE 1: checker ----
    if struct:
        print(f"R={R}: ABORT - {len(struct)} structural violations", file=sys.stderr)
        for v in struct[:5]:
            print(f"   {v}", file=sys.stderr)
        logf.write(f"R={R}: ABORTED - {len(struct)} structural violations\n")
        return None

    exp_unserved = [int(x) for x in exp["unserved_ids"].split(";") if x]
    if len(served_ids) != exp["served"] or unserved_ids != exp_unserved:
        msg = (f"R={R}: ABORT - MISMATCH vs coverage.log. "
               f"served {len(served_ids)} (expected {exp['served']}); "
               f"unserved {unserved_ids} (expected {exp_unserved})")
        print(msg, file=sys.stderr)
        logf.write(msg + "\n")
        return None

    cons = inst.vehicles[0].max_energy_consumption_kWh_per_km or inst.energy.consumption_kWh_per_km
    cap = inst.vehicles[0].battery_capacity_kWh
    arcdist = {(a.from_node, a.to_node): a.distance_km for a in inst.arcs}

    xs = [n.x_km for n in inst.nodes.values()]
    ys = [n.y_km for n in inst.nodes.values()]

    out = {
        "meta": {
            "R": R,
            "served": len(served_ids),
            "total": len(ids),
            "structural_violations": len(struct),
            "wall_time_s": round(wall, 2),
            "routing_cost": round(sum(_route_cost(inst, pl.arcs, pl.charge)
                                      for pl in res.per_vehicle.values()), 4),
            "checker_verdict": ("PASS (0 structural violations; "
                                f"{len(unserved_ids)} request(s) unserved)"),
            "source_log": "coverage.log",
            "battery_capacity_kWh": cap,
            "consumption_kWh_per_km": cons,
            "region_bbox": {"x_min": min(xs), "x_max": max(xs),
                            "y_min": min(ys), "y_max": max(ys)},
        },
        "nodes": {str(nid): {"x": n.x_km, "y": n.y_km,
                             "charging_enabled": inst.charging.node_enabled(nid)}
                  for nid, n in inst.nodes.items()},
        "edges": [[a.from_node, a.to_node, a.distance_km] for a in inst.arcs],
        "depot": {"start": inst.vehicles[0].start_node,
                  "end": inst.vehicles[0].end_node},
        "requests": {},
        "routes": {},
        "soc_profile": {},
        "charging_events": [],
    }

    served_by = {}
    for vid, pl in res.per_vehicle.items():
        for rid in pl.served:
            served_by[rid] = vid
    for r in inst.requests:
        out["requests"][str(r.id)] = {
            "pickup_node": r.pickup_node,
            "delivery_node": r.delivery_node,
            "served_by": served_by.get(r.id),
            "status": "SERVED" if r.id in served_by else "UNSERVED",
        }

    veh_by_id = {v.id: v for v in inst.vehicles}
    for vid, pl in res.per_vehicle.items():
        veh = veh_by_id[vid]
        seq = node_sequence_from_arcs(veh, pl.arcs)
        total_km = sum(arcdist[(seq[i], seq[i + 1])] for i in range(len(seq) - 1))
        out["routes"][vid] = {
            "node_sequence": seq,
            "served_requests": sorted(pl.served),
            "cost": round(_route_cost(inst, pl.arcs, pl.charge), 6),
            "total_km": round(total_km, 6),
        }

        # SOC walk computed from the raw arcs exactly as the checker does
        # (no clamping, no smoothing).
        prof = []
        soc = veh.initial_soc_kWh
        cum = 0.0
        if seq:
            prof.append({"node": seq[0], "cumulative_km": 0.0, "soc_kWh": round(soc, 6)})
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            d = arcdist[(a, b)]
            amt = pl.charge.get((a, b), 0.0)
            eta = 1.0 if inst.charging.arc_enabled(a, b) else 0.0
            soc_before = soc
            soc = soc + eta * amt - cons * d
            cum += d
            prof.append({"node": b, "cumulative_km": round(cum, 6),
                         "soc_kWh": round(soc, 6)})
            if amt > 1e-9 and eta > 0:
                out["charging_events"].append({
                    "vehicle": vid,
                    "at_node": a,
                    "arc": [a, b],
                    "price_per_kWh": inst.prices.base_price_per_node.get(a, 0.0),
                    "amount_kWh": amt,
                    "soc_before": round(soc_before, 6),
                    "soc_after": round(soc_before + eta * amt, 6),
                    "cumulative_km": round(cum - d, 6),
                })
        out["soc_profile"][vid] = prof

    path = f"results/geometry/solution_geometry_R{R}.json"
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    size = os.path.getsize(path)
    line = (f"R={R}: WROTE {path} ({size/1024:.1f} KB) - checker PASS "
            f"(0 structural violations), served={len(served_ids)}/{len(ids)} "
            f"matches coverage.log, unserved={unserved_ids}, "
            f"charging_events={len(out['charging_events'])}, "
            f"nodes={len(out['nodes'])}, edges={len(out['edges'])}, wall={wall:.2f}s")
    print(line, flush=True)
    logf.write(line + "\n")
    return out["meta"]


def main():
    metas = {}
    os.makedirs("results/geometry", exist_ok=True)
    with open("results/geometry/geometry_export.log", "w") as logf:
        logf.write("Geometry export\n")
        logf.write("gates: (1) src/verify/check_solution on every route; "
                   "(2) served count + unserved ids must match "
                   "coverage.log exactly\n\n")
        for R in (30, 100, 200, 300):
            m = export(R, logf)
            if m is None:
                logf.write("EXPORT ABORTED - see message above\n")
                print("EXPORT ABORTED", file=sys.stderr)
                sys.exit(1)
            metas[R] = m
        logf.write("\nall four scales exported, all checker verdicts PASS\n")
    print("\n=== META BLOCKS ===")
    for R, m in metas.items():
        print(f"\n--- solution_geometry_R{R}.json ---")
        print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
