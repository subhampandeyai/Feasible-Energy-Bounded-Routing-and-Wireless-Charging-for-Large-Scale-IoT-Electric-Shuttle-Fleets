from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Node:
    id: int
    x_km: float
    y_km: float
    node_type: str  # "start" | "end" | "pickup" | "delivery"


@dataclass(frozen=True)
class Arc:
    from_node: int
    to_node: int
    distance_km: float
    travel_time_min: float
    travel_time_is_derived: bool


@dataclass(frozen=True)
class Request:
    id: int
    pickup_node: int
    delivery_node: int
    load: Optional[float] = None
    service_time_min: Optional[float] = None
    release_time_min: Optional[float] = None
    time_window: Optional[Tuple[float, float]] = None
    max_ride_time_min: Optional[float] = None


@dataclass(frozen=True)
class Vehicle:
    id: str
    start_node: int
    end_node: int
    battery_capacity_kWh: float
    initial_soc_kWh: float
    capacity: Optional[float] = None
    min_soc_kWh: Optional[float] = None
    max_soc_kWh: Optional[float] = None
    min_soc_at_destination_kWh: Optional[float] = None
    max_energy_consumption_kWh_per_km: Optional[float] = None
    charging_efficiency: Optional[float] = None


@dataclass(frozen=True)
class ChargingSite:
    node_id: int
    small_kWh: Optional[float]
    medium_kWh: Optional[float]
    large_kWh: Optional[float]
    efficiency: Optional[float]
    recharge_rate_kW: Optional[float] = None
    is_enabled: bool = True
    # The price per kWh lives in PriceSchedule. charging_stations.csv carries
    # both, and the reader splits them.


@dataclass(frozen=True)
class ChargingConfig:
    sites_by_node: Dict[int, ChargingSite]
    electrified_arc_fraction: float = 1.0

    def node_enabled(self, node_id: int) -> bool:
        site = self.sites_by_node.get(node_id)
        return site is not None and site.is_enabled

    def arc_enabled(self, from_node: int, to_node: int) -> bool:
        # An arc can charge if both endpoints are charging-enabled nodes and the
        # arc is kept by the electrified_arc_fraction selection. The default
        # fraction 1.0 keeps every such arc.
        if not (self.node_enabled(from_node) and self.node_enabled(to_node)):
            return False
        if self.electrified_arc_fraction >= 1.0:
            return True
        if self.electrified_arc_fraction <= 0.0:
            return False
        # Deterministic selection from a stable hash of the endpoints (not
        # Python's salted hash()), so a given fraction always keeps the same arcs.
        bucket = (from_node * 1_000_003 + to_node) % 1000
        return bucket < self.electrified_arc_fraction * 1000


@dataclass(frozen=True)
class PriceSchedule:
    hourly_multiplier: Tuple[float, ...]  # length 24, index = hour of day
    base_price_per_node: Dict[int, float]

    def price_at(self, node_id: int, hour: int) -> float:
        return self.base_price_per_node[node_id] * self.hourly_multiplier[hour % 24]


@dataclass(frozen=True)
class EnergyModel:
    consumption_kWh_per_km: float
    kappa: Optional[float] = None
    iota: Optional[float] = None
    road_capacity_Cr: Optional[float] = None


@dataclass(frozen=True)
class DegradationParams:
    zeta: Optional[Tuple[float, float, float, float, float, float, float]] = None
    P_b: Optional[float] = None
    V: Optional[float] = None

    @property
    def is_populated(self) -> bool:
        return self.zeta is not None and self.P_b is not None and self.V is not None


@dataclass(frozen=True)
class ObjectiveWeights:
    epsilon1: float
    epsilon2: float
    epsilon3: Optional[float]
    # epsilon3 is stored as read; the degradation term is built only when
    # DegradationParams.is_populated.


@dataclass(frozen=True)
class Instance:
    name: str
    nodes: Dict[int, Node]
    arcs: Tuple[Arc, ...]
    requests: Tuple[Request, ...]
    vehicles: Tuple[Vehicle, ...]
    charging: ChargingConfig
    prices: PriceSchedule
    energy: EnergyModel
    objective: ObjectiveWeights
    degradation: DegradationParams
    big_M: float
    out_arcs: Dict[int, Tuple[Arc, ...]] = field(init=False, repr=False)
    in_arcs: Dict[int, Tuple[Arc, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        out_map: Dict[int, List[Arc]] = {n: [] for n in self.nodes}
        in_map: Dict[int, List[Arc]] = {n: [] for n in self.nodes}
        for arc in self.arcs:
            out_map.setdefault(arc.from_node, []).append(arc)
            in_map.setdefault(arc.to_node, []).append(arc)
        object.__setattr__(self, "out_arcs", {k: tuple(v) for k, v in out_map.items()})
        object.__setattr__(self, "in_arcs", {k: tuple(v) for k, v in in_map.items()})

    def request_by_id(self, request_id: int) -> Request:
        for r in self.requests:
            if r.id == request_id:
                return r
        raise KeyError(f"request {request_id} not found in instance {self.name}")
