from __future__ import annotations

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

# A tiny hand-built instance for fast end-to-end pipeline validation. 6 nodes,
# 2 vehicles, 2 requests. SOC never binds (short arcs, full battery). Distinct
# from the data/ reader path - this exercises the solver stack, not I/O.


def build_mini_instance(num_vehicles: int = 2) -> Instance:
    coords = {
        0: (0.0, 0.0), 1: (10.0, 0.0),
        2: (2.0, 1.0), 3: (4.0, 1.0),
        4: (6.0, 1.0), 5: (8.0, 1.0),
    }
    types = {0: "start", 1: "end", 2: "pickup", 3: "delivery", 4: "pickup", 5: "delivery"}
    nodes = {n: Node(n, x, y, types[n]) for n, (x, y) in coords.items()}

    raw_arcs = [
        (0, 2), (0, 4), (0, 1),
        (2, 3), (2, 4), (2, 5),
        (3, 1), (3, 4), (3, 5),
        (4, 5), (4, 3),
        (5, 1), (5, 3),
    ]

    def dist(a, b):
        (x1, y1), (x2, y2) = coords[a], coords[b]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    arcs = tuple(
        Arc(a, b, dist(a, b), dist(a, b), travel_time_is_derived=True) for a, b in raw_arcs
    )

    requests = (
        Request(id=1, pickup_node=2, delivery_node=3, release_time_min=0.0),
        Request(id=2, pickup_node=4, delivery_node=5, release_time_min=0.0),
    )

    vehicles = tuple(
        Vehicle(
            id=f"veh_{k+1}", start_node=0, end_node=1,
            battery_capacity_kWh=30.0, initial_soc_kWh=30.0,
            max_energy_consumption_kWh_per_km=0.1, charging_efficiency=0.9,
        )
        for k in range(num_vehicles)
    )

    sites = {
        n: ChargingSite(n, 5.0, 15.0, 20.0, 0.9, is_enabled=True) for n in nodes
    }
    charging = ChargingConfig(sites_by_node=sites, electrified_arc_fraction=1.0)

    prices = PriceSchedule(
        hourly_multiplier=tuple([1.0] * 24),
        base_price_per_node={n: 0.9 for n in nodes},
    )

    return Instance(
        name="mini",
        nodes=nodes,
        arcs=arcs,
        requests=requests,
        vehicles=vehicles,
        charging=charging,
        prices=prices,
        energy=EnergyModel(consumption_kWh_per_km=0.1),
        objective=ObjectiveWeights(epsilon1=1.0 / 3, epsilon2=1.0 / 3, epsilon3=1.0 / 3),
        degradation=DegradationParams(),
        big_M=1000.0,
    )
