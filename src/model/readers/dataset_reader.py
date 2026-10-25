from __future__ import annotations

import csv
import os
from typing import Dict, Tuple

from src.model.types import (
    Arc,
    ChargingConfig,
    ChargingSite,
    DegradationParams,
    EnergyModel,
    Instance,
    Node,
    ObjectiveWeights,
    PriceSchedule,
    Request,
    Vehicle,
)

# Reads the instance files in data/ into an Instance. Request files exist for
# R = 30, 100, 200 and 300; all sizes share the same 602-node graph.

_VALID_REQUEST_SIZES = (30, 100, 200, 300)


def _read_parameters(dataset_dir: str) -> Dict[str, float]:
    path = os.path.join(dataset_dir, "parameters.csv")
    params: Dict[str, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            params[row["parameter"]] = float(row["value"])
    return params


def _read_nodes(dataset_dir: str) -> Dict[int, Node]:
    path = os.path.join(dataset_dir, "nodes.csv")
    nodes: Dict[int, Node] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            node_id = int(row["node_id"])
            nodes[node_id] = Node(
                id=node_id,
                x_km=float(row["x_km"]),
                y_km=float(row["y_km"]),
                node_type=row["node_type"],
            )
    return nodes


def _read_arcs(dataset_dir: str) -> Tuple[Arc, ...]:
    # travel_time_min is derived under a unit-speed convention
    # (travel_time_min = distance_km); data/ has no speed or travel-time column.
    path = os.path.join(dataset_dir, "edges_directed.csv")
    arcs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            distance_km = float(row["distance_km"])
            arcs.append(
                Arc(
                    from_node=int(row["from_node"]),
                    to_node=int(row["to_node"]),
                    distance_km=distance_km,
                    travel_time_min=distance_km,
                    travel_time_is_derived=True,
                )
            )
    return tuple(arcs)


def _read_requests(dataset_dir: str, num_requests: int) -> Tuple[Request, ...]:
    if num_requests not in _VALID_REQUEST_SIZES:
        raise ValueError(
            f"num_requests={num_requests} has no matching file; "
            f"data/ only has {_VALID_REQUEST_SIZES}"
        )
    path = os.path.join(dataset_dir, f"requests_{num_requests}.csv")
    requests = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            requests.append(
                Request(
                    id=int(row["request_id"]),
                    pickup_node=int(row["pickup_node_id"]),
                    delivery_node=int(row["delivery_node_id"]),
                    load=None,  # not in data/
                    service_time_min=None,  # not in data/
                    release_time_min=float(row["departure_time_min"]),
                    time_window=None,  # not in data/; departure_time_min is a release time only
                    max_ride_time_min=None,  # not in data/
                )
            )
    return tuple(requests)


def _read_vehicles(dataset_dir: str) -> Tuple[Vehicle, ...]:
    path = os.path.join(dataset_dir, "vehicles.csv")
    vehicles = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            vehicles.append(
                Vehicle(
                    id=row["vehicle_id"],
                    start_node=int(row["start_node_id"]),
                    end_node=int(row["end_node_id"]),
                    battery_capacity_kWh=float(row["battery_capacity_kWh"]),
                    initial_soc_kWh=float(row["initial_battery_kWh"]),
                    capacity=None,  # not in data/; the model has no load capacity
                    min_soc_kWh=None,  # SOC bounds are applied by the model, not stored here
                    max_soc_kWh=None,
                    min_soc_at_destination_kWh=None,  # not in data/
                    max_energy_consumption_kWh_per_km=float(
                        row["max_energy_consumption_kWh_per_km"]
                    ),
                    charging_efficiency=float(row["charging_efficiency"]),
                )
            )
    return tuple(vehicles)


def _read_charging(dataset_dir: str) -> ChargingConfig:
    path = os.path.join(dataset_dir, "charging_stations.csv")
    sites: Dict[int, ChargingSite] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            node_id = int(row["node_id"])
            sites[node_id] = ChargingSite(
                node_id=node_id,
                small_kWh=float(row["small_allowable_kWh"]),
                medium_kWh=float(row["medium_allowable_kWh"]),
                large_kWh=float(row["large_allowable_kWh"]),
                efficiency=float(row["charging_efficiency"]),
                recharge_rate_kW=None,  # not in data/; charging is an amount per arc, not a rate
                is_enabled=bool(int(row["is_charging_enabled"])),
            )
    # Every arc between two charging-enabled nodes can charge.
    return ChargingConfig(sites_by_node=sites, electrified_arc_fraction=1.0)


def _read_prices(dataset_dir: str) -> PriceSchedule:
    price_path = os.path.join(dataset_dir, "price_schedule.csv")
    multipliers = [0.0] * 24
    with open(price_path, newline="") as f:
        for row in csv.DictReader(f):
            multipliers[int(row["hour"])] = float(row["price_multiplier"])
    if any(m == 0.0 for m in multipliers):
        raise ValueError("price_schedule.csv did not cover all 24 hours")

    station_path = os.path.join(dataset_dir, "charging_stations.csv")
    base_price: Dict[int, float] = {}
    with open(station_path, newline="") as f:
        for row in csv.DictReader(f):
            base_price[int(row["node_id"])] = float(row["base_price_per_kWh"])

    return PriceSchedule(hourly_multiplier=tuple(multipliers), base_price_per_node=base_price)


def load_dataset_instance(dataset_dir: str, num_requests: int) -> Instance:
    """Load one data/ scenario (30/100/200/300 requests) into an Instance.

    The graph, fleet and charging sites are the same for every num_requests;
    only the request set changes.
    """
    params = _read_parameters(dataset_dir)
    nodes = _read_nodes(dataset_dir)
    arcs = _read_arcs(dataset_dir)
    requests = _read_requests(dataset_dir, num_requests)
    vehicles = _read_vehicles(dataset_dir)
    charging = _read_charging(dataset_dir)
    prices = _read_prices(dataset_dir)

    energy = EnergyModel(
        # Default consumption rate, used only when a vehicle has no rate of its
        # own. The congestion terms kappa, iota and road_capacity_Cr are not in data/.
        consumption_kWh_per_km=params["battery_capacity_kWh"] / params["max_mileage_km"],
        kappa=None,
        iota=None,
        road_capacity_Cr=None,
    )

    objective = ObjectiveWeights(
        epsilon1=params["epsilon1"],
        epsilon2=params["epsilon2"],
        epsilon3=params["epsilon3"],
    )

    # data/ has no battery-degradation parameters, so DegradationParams stays
    # empty and the objective uses its two terms: distance and energy cost.
    degradation = DegradationParams(zeta=None, P_b=None, V=None)

    return Instance(
        name=f"dataset_{num_requests}",
        nodes=nodes,
        arcs=arcs,
        requests=requests,
        vehicles=vehicles,
        charging=charging,
        prices=prices,
        energy=energy,
        objective=objective,
        degradation=degradation,
        big_M=params["BigM"],
    )
