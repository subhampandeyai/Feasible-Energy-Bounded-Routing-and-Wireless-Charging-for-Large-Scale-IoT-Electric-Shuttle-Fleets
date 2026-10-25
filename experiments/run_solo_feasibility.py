from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import argparse
import json
import sys

from src.model.readers.dataset_reader import load_dataset_instance
from src.solve.cluster_route import solo_feasibility

# Solo-feasibility scan: classify every request as feasible / proven-infeasible
# / undetermined by asking whether ONE identical vehicle can serve it alone.
# Since all vehicles are identical and the checker enforces no inter-vehicle
# coupling, a request no single vehicle can serve is infeasible for the whole
# fleet. A search that TERMINATES with no route is a proof; one that hits the
# time budget is UNDETERMINED, never a proof. Writes a JSON consumed by
# run_cluster_route.py --solo-json.


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=30, choices=[30, 100, 200, 300])
    ap.add_argument("--budget-s", type=float, default=8.0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    inst = load_dataset_instance("data", args.requests)
    veh = inst.vehicles[0]
    feasible, infeasible, undetermined = solo_feasibility(inst, veh, args.budget_s)
    print(f"R={args.requests}: {len(feasible)} feasible, "
          f"{len(infeasible)} proven-infeasible={sorted(infeasible)}, "
          f"{len(undetermined)} undetermined={sorted(undetermined)}", flush=True)
    print(f"feasible ceiling = {len(inst.requests) - len(infeasible)}/{len(inst.requests)}", flush=True)
    out = args.out or f"results/coverage/solo_R{args.requests}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"R": args.requests, "feasible": sorted(feasible),
               "infeasible": sorted(infeasible), "undetermined": sorted(undetermined)},
              open(out, "w"), indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
