from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import argparse
import json
import logging
import sys
import time

from src.model.readers.dataset_reader import load_dataset_instance
from src.solve.cluster_route import solve_cluster_route, to_vehicle_routes
from src.verify.checker import check_solution


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=30, choices=[30, 100, 200, 300])
    ap.add_argument("--solo-budget-s", type=float, default=8.0)
    ap.add_argument("--engine", type=str, default="insertion", choices=["insertion", "labeling"])
    ap.add_argument("--n-candidates", type=int, default=14)
    ap.add_argument("--reinsert-passes", type=int, default=3)
    ap.add_argument("--cluster-solve-budget-s", type=float, default=20.0)
    ap.add_argument("--greedy-step-budget-s", type=float, default=8.0)
    ap.add_argument("--label-budget", type=int, default=400000)
    ap.add_argument("--solo-json", type=str, default="")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        stream=sys.stdout, force=True)

    print(f"Loading R={args.requests}...", flush=True)
    inst = load_dataset_instance("data", args.requests)
    print(f"vehicles={len(inst.vehicles)} requests={len(inst.requests)} "
          f"nodes={len(inst.nodes)} arcs={len(inst.arcs)}", flush=True)

    precomputed = None
    if args.solo_json:
        d = json.load(open(args.solo_json))
        precomputed = (d["feasible"], d["infeasible"], d["undetermined"])
        print(f"loaded precomputed solo: {len(d['feasible'])} feasible, "
              f"infeasible={d['infeasible']}, undetermined={d['undetermined']}", flush=True)

    t0 = time.time()
    res = solve_cluster_route(
        inst,
        solo_budget_s=args.solo_budget_s,
        engine=args.engine,
        n_candidates=args.n_candidates,
        reinsert_passes=args.reinsert_passes,
        cluster_solve_budget_s=args.cluster_solve_budget_s,
        greedy_step_budget_s=args.greedy_step_budget_s,
        label_budget=args.label_budget,
        precomputed_solo=precomputed,
    )
    wall = time.time() - t0

    print(f"\n=== CLUSTER-ROUTE RESULT R={args.requests}, wall={wall:.2f}s ===", flush=True)
    print(f"served ({len(res.served)}): {list(res.served)}", flush=True)
    print(f"unserved_feasible ({len(res.unserved_feasible)}): {list(res.unserved_feasible)} "
          f"[placeable but not placed - NOT a proof of infeasibility]", flush=True)
    print(f"proven_infeasible ({len(res.proven_infeasible)}): {list(res.proven_infeasible)} "
          f"[solo search exhausted, no route for any identical vehicle]", flush=True)
    print(f"undetermined ({len(res.undetermined_feasibility)}): {list(res.undetermined_feasibility)} "
          f"[solo search hit budget - status unknown, NOT a proof]", flush=True)
    print("\nper-vehicle:", flush=True)
    for vid, pl in res.per_vehicle.items():
        print(f"  {vid}: served={pl.served} cost={pl.cost:.3f} arcs={len(pl.arcs)}", flush=True)

    routes = to_vehicle_routes(res.per_vehicle)
    rep = check_solution(inst, routes)
    print(f"\n=== INDEPENDENT CHECKER (R={args.requests}) ===", flush=True)
    # coverage of PLACEABLE requests, treating proven-infeasible separately
    n = len(inst.requests)
    covered = len(res.served)
    print(f"checker_full_summary_first_line: {rep.summary().splitlines()[0]}", flush=True)
    structural = [v for v in rep.violations if "served by NO vehicle" not in v]
    coverage_viol = [v for v in rep.violations if "served by NO vehicle" in v]
    print(f"STRUCTURAL violations (energy/precedence/path/subtour): {len(structural)}", flush=True)
    for v in structural[:20]:
        print(f"   STRUCT: {v}", flush=True)
    print(f"coverage violations (served by NO vehicle): {len(coverage_viol)}", flush=True)
    print(f"CHECKER_COVERAGE: {covered}/{n} requests served", flush=True)
    print(f"CHECKER_STRUCTURAL_PASS: {len(structural) == 0}", flush=True)
    proven = len(res.proven_infeasible)
    print(f"COVERAGE_VS_FEASIBLE_CEILING: {covered}/{n - proven} "
          f"(ceiling excludes {proven} proven-infeasible)", flush=True)
    # A run is a full success iff structural pass AND every non-proven-infeasible
    # request is served.
    full = (len(structural) == 0) and (covered == n - proven)
    print(f"FULL_FEASIBLE_COVERAGE_MODULO_PROVEN_INFEASIBLE: {full}", flush=True)


if __name__ == "__main__":
    main()
