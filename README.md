# Feasible Energy-Bounded Routing and Wireless Charging for Large Scale IoT Electric Shuttle Fleets

A Python solver for routing and charging a fleet of electric vehicles that serve
pickup-and-delivery requests on a road network with dynamic wireless charging.

## Overview

Each vehicle travels one elementary path from its start node to its end node,
picks up every assigned request before delivering it, and keeps its state of
charge (SOC) between 0 and its battery capacity. Energy can be bought only on
charging-enabled arcs, in discrete amounts.

The solver works cluster-first, route-second:

1. Each request is tested for single-vehicle feasibility and classified as
   feasible, infeasible or undetermined.
2. Requests are grouped into balanced spatial clusters, one per vehicle.
3. Each vehicle's route is built by node-disjoint insertion under several
   request orderings. Where the SOC would fall below zero, the smallest allowed
   charging amount is added on a charging arc.
4. Requests left unplaced are offered to the other vehicles in reinsertion
   passes.
5. The result is validated by an independent checker.

The routing procedure is deterministic.

## Key functionality

- Cluster-first insertion router (`src/solve/cluster_route.py`,
  `src/solve/insertion_router.py`)
- Independent solution checker for SOC bounds, elementary and connected routes,
  flow conservation, pickup-before-delivery precedence and single service of
  each request (`src/verify/checker.py`)
- Exact dynamic program that re-optimises charging cost on a fixed route,
  subject to a reserve level (`src/solve/charging_optimizer.py`)
- Comparison methods: regret-2 greedy insertion, column generation with ESPPRC
  labeling, Lagrangian relaxation with Gavish–Graves flow subproblems, a
  monolithic MILP, ADMM, and an ADMM + column-generation ensemble
  (`src/solve/`)
- MILP components solved with HiGHS through `highspy`

## Project structure

```
data/              Input instance (CSV and JSON)
  witness_logs/    Solution logs read by run_witness_scan.py
experiments/       Command-line experiment scripts (run_*.py)
src/
  model/           Problem types, formulation, MILP builder, dataset reader
  solve/           Router, charging optimiser and comparison methods
  verify/          Solution checker
results/           Output folders written by the experiments
tests/             pytest test suite
requirements.txt
LICENSE
```

## Requirements and installation

- Python 3.9–3.12
- `highspy` 1.15.1, `numpy` 2.0.2, `pytest` 8.4.2
- On Windows only: `psutil` 7.0.0 and `pywin32` 311, used by
  `experiments/run_baseline_r30.py`

The Windows-only packages are selected by environment markers in
`requirements.txt`.

```bash
git clone <repository-url>
cd <repository-directory>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Quick start

Solve the 30-request scenario and print the per-vehicle routes and the checker
report:

```bash
python experiments/run_cluster_route.py --requests 30
```

`--requests` accepts `30`, `100`, `200` or `300`. Run with `--help` to list the
other options: routing engine, candidate count, reinsertion passes and time
budgets. This command prints to the console and writes no files.

## Running experiments

The experiment scripts set the working directory to the repository root, so you
can run them from any location. They write their outputs under `results/`.
Some scripts read files written by earlier ones, so run the groups in the order
shown.

Request feasibility and coverage:

```bash
python experiments/run_solo_feasibility.py --requests 100
python experiments/run_witness_scan.py
python experiments/run_classify_requests.py --requests 30
python experiments/run_classify_requests.py --requests 100
python experiments/run_classify_requests.py --requests 200
python experiments/run_classify_requests.py --requests 300
python experiments/run_coverage.py
```

Solution quality and scaling:

```bash
python experiments/run_optimality_gap.py
python experiments/run_multi_instance.py
python experiments/run_robustness_ablation.py
python experiments/run_fleet_scaling.py 10,20,50,100
python experiments/run_engine_wall.py
```

Charging analysis:

```bash
python experiments/run_geometry_export.py
python experiments/run_charging_regimes.py
python experiments/run_energy_bound.py
python experiments/run_charging_binds.py
python experiments/run_charging_optimizer_experiment.py
```

Method comparison at 30 requests (Windows only):

```bash
python experiments/run_baseline_r30.py
```

`run_baseline_r30.py` runs each method in a separate process and terminates it
at a hard wall-clock cap using a Windows Job Object. It also checks the Windows
event log to confirm the machine did not sleep during the run. The options are:

- `--cap`: the cap in seconds (default 1800)
- `--methods`: a comma-separated subset of `cluster`, `greedy`, `cg`,
  `monolithic`, `ggflow`, `admm` and `ensemble`
- `--tag`: a label for the output files

It requires `results/coverage/classification_R30.json`.

### Reproducibility

Random choices in the experiments use fixed seeds defined in the scripts. For
example, the charger-availability scripts use `SEED = 20260727`, and
`run_multi_instance.py` uses seeds 11–55. Served counts, costs, charge amounts
and SOC values are therefore deterministic. Wall-clock times, and the number of
labels generated within a fixed time budget, depend on the hardware.

## Tests

```bash
python -m pytest tests
```

## Data and outputs

Included input data (`data/`):

- a synthetic road network of 602 nodes and 3,632 directed arcs on a
  60 × 60 km area, with distances, charging stations and a time-of-use price
  schedule
- 10 vehicles, with model parameters
- request files for 30, 100, 200 and 300 pickup-and-delivery pairs
  (`requests_30.csv` … `requests_300.csv`)
- solution logs in `data/witness_logs/`

`data/README.md` describes the file formats and parameter values.

Generated outputs (`results/`, shipped as empty folders):

| Folder | Contents |
|---|---|
| `results/coverage/` | Feasibility, classification and coverage JSON |
| `results/csv/` | Tabular summaries of each experiment |
| `results/scaling/` | Scaling, robustness, optimality-gap and engine results |
| `results/geometry/` | Exported route geometry |
| `results/charging/` | Charging-regime, energy and optimiser results |
| `results/baseline/` | Logs and CSV from `run_baseline_r30.py` |

## License

MIT License. See `LICENSE`.
