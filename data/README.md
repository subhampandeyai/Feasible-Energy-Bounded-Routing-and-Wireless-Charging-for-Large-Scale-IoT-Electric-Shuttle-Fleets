# IoEV Simulation Dataset (Paper-Matched Synthetic Instance)

This package is a **synthetic, ready-to-run dataset** constructed to match the simulation setup explicitly reported in the paper:

- Road network size: **60 × 60 km²**
- IoEV battery capacity: **30 kWh**
- Maximum mileage: **260 km**
- Allowable charging amounts: **5 / 15 / 20 kWh**
- Charging efficiency: **90%**
- Weighting parameters: **ε1 = ε2 = ε3 = 1/3**
- Big-M parameter: **M = 1000**
- Dynamic charging price range: **$0.5–$1.0 per kWh**
- Evaluation scenarios: **30, 100, 200, 300 pickup–delivery pairs**

## Files

- `nodes.csv`
- `edges_undirected.csv`
- `edges_directed.csv`
- `distance_matrix.csv`
- `vehicles.csv`
- `charging_stations.csv`
- `price_schedule.csv`
- `parameters.csv`
- `model_config.json`

## Scenarios

- `requests_30.csv`
- `requests_100.csv`
- `requests_200.csv`
- `requests_300.csv`

## How to use

1. Load `nodes.csv`, `edges_directed.csv`, and `distance_matrix.csv` for routing.
2. Load `vehicles.csv` for the IoEV battery model.
3. Load `charging_stations.csv` and `price_schedule.csv` for charging decisions.
4. Select one of the request files depending on the experiment size.

## Notes

The paper does not publish a raw benchmark dataset. This package is a reproducible synthetic reconstruction that matches the paper’s stated simulation parameters.
