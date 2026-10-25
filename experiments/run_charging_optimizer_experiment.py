from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import csv, dataclasses, json, os, random, statistics, sys, time

from src.model.readers.dataset_reader import load_dataset_instance
from src.model.types import ChargingConfig
from src.solve.cluster_route import solve_cluster_route, to_vehicle_routes, _route_cost
from src.solve.charging_optimizer import optimize_charging, plan_is_soc_valid
from src.verify.checker import check_solution, VehicleRoute

# Compares two charging plans on identical routes:
#   BASELINE  - the reactive constant-amount repair inside the router
#   OPTIMISED - the per-route charging optimiser, with a reserve-SOC floor
#
# Reported for each: whether charge amounts vary with price, total electricity
# cost, the wall time of the charging decision itself, coverage and structural
# violations, and the minimum SOC reached.

SEED = 20260727
R = 300
KNOBS = (24, 5)
RESERVE_KWH = 3.0
OUT = "results/csv"
os.makedirs(OUT, exist_ok=True)
os.makedirs("results/charging", exist_ok=True)


def make_regime(base, frac, seed):
    depots = {v.start_node for v in base.vehicles} | {v.end_node for v in base.vehicles}
    all_nodes = sorted(base.nodes)
    if frac >= 1.0:
        enabled = set(all_nodes)
    else:
        k = max(len(depots), int(round(frac * len(all_nodes))))
        pool = [n for n in all_nodes if n not in depots]
        rng = random.Random(seed)
        enabled = set(depots) | set(rng.sample(pool, k - len(depots)))
    sites = {nid: dataclasses.replace(s, is_enabled=(nid in enabled))
             for nid, s in base.charging.sites_by_node.items()}
    return dataclasses.replace(
        base, charging=ChargingConfig(sites_by_node=sites,
                                      electrified_arc_fraction=base.charging.electrified_arc_fraction))


def node_seq(veh, arcs):
    if not arcs:
        return []
    succ = {i: j for (i, j) in arcs}
    seq, cur, seen = [veh.start_node], veh.start_node, {veh.start_node}
    while cur in succ:
        cur = succ[cur]
        if cur in seen:
            break
        seq.append(cur); seen.add(cur)
    return seq


def energy_cost_of(instance, charge):
    tot = 0.0
    for (i, j), amt in charge.items():
        if instance.charging.arc_enabled(i, j):
            tot += instance.prices.base_price_per_node.get(i, 0.0) * amt
    return tot


def events_of(instance, per_vehicle_charge):
    ev = []
    for vid, ch in per_vehicle_charge.items():
        for (i, j), amt in ch.items():
            if amt > 1e-9 and instance.charging.arc_enabled(i, j):
                ev.append((vid, i, amt, instance.prices.base_price_per_node.get(i, 0.0)))
    return ev


def corr(xs, ys):
    if len(xs) < 2:
        return None
    sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
    if sx < 1e-12 or sy < 1e-12:
        return None          # undefined: one variable has zero variance
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (len(xs) * sx * sy)


def main():
    base = load_dataset_instance("data", R)
    ids = [r.id for r in base.requests]
    nc, pp = KNOBS
    arcdist = {(a.from_node, a.to_node): a.distance_km for a in base.arcs}
    rows = []

    print("CHARGING OPTIMIZER EXPERIMENT", flush=True)
    print(f"  R={R}, seed={SEED}, reserve_SOC={RESERVE_KWH} kWh", flush=True)
    print("  routes are FIXED by the router first; charging decided afterwards", flush=True)

    for name, frac in (("dense", 1.00), ("medium", 0.30), ("sparse", 0.10)):
        inst = make_regime(base, frac, SEED)
        n_on = sum(1 for n in inst.nodes if inst.charging.node_enabled(n))

        t0 = time.time()
        res = solve_cluster_route(inst, engine="insertion", n_candidates=nc,
                                  reinsert_passes=pp, precomputed_solo=(ids, [], []))
        t_route = time.time() - t0

        base_charge = {vid: dict(pl.charge) for vid, pl in res.per_vehicle.items()}
        base_routes = to_vehicle_routes(res.per_vehicle)
        rep_b = check_solution(inst, base_routes)
        struct_b = [v for v in rep_b.violations if "served by NO vehicle" not in v]
        served = len(res.served)

        # ---- optimised charging on the SAME routes ----
        t1 = time.time()
        opt_charge, infeasible_routes = {}, []
        for vid, pl in res.per_vehicle.items():
            veh = next(v for v in inst.vehicles if v.id == vid)
            seq = node_seq(veh, pl.arcs)
            if len(seq) < 2:
                opt_charge[vid] = {}
                continue
            plan = optimize_charging(inst, veh, seq, arcdist, reserve_kWh=RESERVE_KWH)
            if not plan.feasible:
                # reserve floor may be unreachable on a route built without it
                plan = optimize_charging(inst, veh, seq, arcdist, reserve_kWh=0.0)
                if not plan.feasible:
                    infeasible_routes.append(vid)
                    opt_charge[vid] = dict(pl.charge)     # fall back, reported
                    continue
                infeasible_routes.append(f"{vid}(reserve_relaxed)")
            assert plan_is_soc_valid(plan, veh, 0.0), f"{vid}: reconstructed plan invalid"
            opt_charge[vid] = plan.charge
        t_charge = time.time() - t1

        opt_routes = {vid: VehicleRoute(arcs=pl.arcs, charge=opt_charge[vid],
                                        served=list(pl.served))
                      for vid, pl in res.per_vehicle.items()}
        rep_o = check_solution(inst, opt_routes)
        struct_o = [v for v in rep_o.violations if "served by NO vehicle" not in v]

        ev_b, ev_o = events_of(inst, base_charge), events_of(inst, opt_charge)
        cost_b, cost_o = (sum(energy_cost_of(inst, c) for c in base_charge.values()),
                          sum(energy_cost_of(inst, c) for c in opt_charge.values()))
        amts_b = sorted({e[2] for e in ev_b})
        amts_o = sorted({e[2] for e in ev_o})
        c_o = corr([e[3] for e in ev_o], [e[2] for e in ev_o])

        # min SOC under the optimised plan
        min_soc = None
        for vid, pl in res.per_vehicle.items():
            veh = next(v for v in inst.vehicles if v.id == vid)
            seq = node_seq(veh, pl.arcs)
            if len(seq) < 2:
                continue
            cons = veh.max_energy_consumption_kWh_per_km or inst.energy.consumption_kWh_per_km
            soc = veh.initial_soc_kWh
            for i in range(len(seq) - 1):
                u, v = seq[i], seq[i + 1]
                eta = 1.0 if inst.charging.arc_enabled(u, v) else 0.0
                soc += eta * opt_charge[vid].get((u, v), 0.0) - cons * arcdist[(u, v)]
                min_soc = soc if min_soc is None else min(min_soc, soc)

        print(f"\n--- regime={name} ({n_on}/{len(inst.nodes)} nodes enabled) ---", flush=True)
        print(f"  routing: served={served}/{len(ids)} struct={len(struct_b)} wall={t_route:.1f}s",
              flush=True)
        print(f"  BASELINE  (reactive): events={len(ev_b)} amounts={amts_b} "
              f"energy_cost={cost_b:.4f}", flush=True)
        print(f"  OPTIMISED (DP)      : events={len(ev_o)} amounts={amts_o} "
              f"energy_cost={cost_o:.4f}  charging_wall={t_charge:.3f}s", flush=True)
        if cost_b > 1e-9:
            print(f"  energy cost change: {100.0*(cost_o-cost_b)/cost_b:+.2f}%", flush=True)
        print(f"  price-vs-amount correlation (optimised): "
              f"{'UNDEFINED (zero amount variance)' if c_o is None else f'{c_o:+.4f}'}", flush=True)
        print(f"  optimised checker: struct_viol={len(struct_o)}  min_SOC={min_soc:.4f} kWh"
              f" (reserve {RESERVE_KWH})", flush=True)
        if infeasible_routes:
            print(f"  routes needing reserve relaxation / infeasible: {infeasible_routes}",
                  flush=True)

        rows.append({
            "regime": name, "nodes_enabled": n_on, "R": R, "reserve_kWh": RESERVE_KWH,
            "served": served, "total_requests": len(ids),
            "routing_wall_s": round(t_route, 2), "charging_wall_s": round(t_charge, 3),
            "struct_viol_baseline": len(struct_b), "struct_viol_optimised": len(struct_o),
            "events_baseline": len(ev_b), "events_optimised": len(ev_o),
            "amounts_baseline": ";".join(str(a) for a in amts_b),
            "amounts_optimised": ";".join(str(a) for a in amts_o),
            "energy_cost_baseline": round(cost_b, 4),
            "energy_cost_optimised": round(cost_o, 4),
            "energy_cost_change_pct": (round(100.0*(cost_o-cost_b)/cost_b, 2) if cost_b > 1e-9 else ""),
            "price_amount_correlation": ("UNDEFINED" if c_o is None else round(c_o, 4)),
            "min_soc_kWh": (round(min_soc, 4) if min_soc is not None else ""),
            "routes_reserve_relaxed": ";".join(map(str, infeasible_routes)),
            "log_file": "charging_optimizer_experiment.log",
        })
        json.dump(rows, open("results/charging/charging_optimizer_experiment_results.json", "w"), indent=2)

    with open(f"{OUT}/charging_optimizer_experiment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}/charging_optimizer_experiment.csv", flush=True)


if __name__ == "__main__":
    main()
