from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from src.model.types import Instance, Vehicle

# Per-route charging optimiser.
#
# Given a route that is ALREADY FIXED (the node sequence chosen by the router),
# decide where to stop and how much energy to buy at each stop, minimising
#
#     min  sum_i price(node_i) * amount_i
#
# subject to
#     soc_0     = initial_soc
#     soc_{i+1} = soc_i + eta_i*amount_i - consumption*dist(i, i+1)
#     reserve <= soc_i <= capacity            for every i      (SOC window)
#     amount_i = 0 unless the arc leaving node i is charging-enabled
#     amount_i in ALLOWED_LEVELS
#
# SOLVED EXACTLY. With the route fixed this is a one-dimensional resource
# problem over an ordered sequence, solved by dynamic programming: stage =
# position on the route, state = cumulative energy purchased so far, action =
# amount bought at this node. The DP returns the globally minimum-cost charging
# plan for that route, or proves that none exists.
#
# The optimiser runs ONCE PER FINISHED ROUTE, outside the insertion search, so
# the routing loop never pays the cost of a charging decision.
#
# RESERVE SOC. The `reserve_kWh` floor keeps a usable margin in the battery
# rather than allowing a route to arrive at ~0 kWh. It is a modelling choice
# that makes solutions operationally meaningful and gives the optimiser a
# genuine trade-off against price.

# Discrete levels the instance actually offers (data/parameters.csv:
# small/medium/large allowable kWh). Zero is always available (do not charge).
ALLOWED_LEVELS = (0.0, 5.0, 15.0, 20.0)


class ChargingPlan:
    """Result of optimising charging over one fixed route."""

    def __init__(self, feasible: bool, charge: Dict[Tuple[int, int], float],
                 energy_cost: float, soc_profile: List[float],
                 n_events: int, reason: str = ""):
        self.feasible = feasible
        self.charge = charge                 # {(from,to): kWh} - router's format
        self.energy_cost = energy_cost       # sum price*amount
        self.soc_profile = soc_profile
        self.n_events = n_events
        self.reason = reason


def optimize_charging(
    instance: Instance,
    vehicle: Vehicle,
    node_seq: List[int],
    arcdist: Dict[Tuple[int, int], float],
    reserve_kWh: float = 0.0,
    soc_step_kWh: float = 0.5,
    levels: Tuple[float, ...] = ALLOWED_LEVELS,
) -> ChargingPlan:
    """Minimum-electricity-cost charging plan for a FIXED route.

    Charging is credited on the arc leaving a node (matching the checker's
    convention: `soc += eta*c_ij - E_ij` on arc (i,j), with eta determined by
    `charging.arc_enabled(i, j)`), so the decision at position i is "how much
    to buy before traversing arc (i, i+1)".

    Returns a ChargingPlan; `feasible=False` means no assignment of the allowed
    amounts keeps SOC within [reserve, capacity] along this route.
    """
    if len(node_seq) < 2:
        return ChargingPlan(True, {}, 0.0, [vehicle.initial_soc_kWh], 0)

    cons = vehicle.max_energy_consumption_kWh_per_km or instance.energy.consumption_kWh_per_km
    cap = vehicle.battery_capacity_kWh
    lo = max(0.0, reserve_kWh)

    arcs = [(node_seq[i], node_seq[i + 1]) for i in range(len(node_seq) - 1)]
    if any(a not in arcdist for a in arcs):
        return ChargingPlan(False, {}, 0.0, [], 0, "route uses a non-existent arc")

    n = len(arcs)
    # EXACT SOC STATES - DO NOT DISCRETISE.
    #
    # SOC must be carried exactly. Bucketing it onto a grid and rounding (in
    # either direction) makes the DP's notion of the battery disagree with the
    # true SOC: rounding down loses energy at every stage, so the reconstructed
    # plan buys energy the real battery cannot absorb and the route overcharges
    # past capacity; rounding up invents energy the vehicle never had. Either
    # way the emitted plan fails the independent checker.
    #
    # Exact states are also cheap here. Every reachable SOC equals
    #     initial_soc + (sum of chosen amounts) - (cumulative distance * cons),
    # and the amounts come from a small fixed set, so at stage i the reachable
    # SOCs are determined entirely by the SUM of amounts chosen so far. Those
    # distinct sums form a small set (bounded by capacity), so states are keyed
    # on the exact cumulative-charge total and SOC is carried exactly.
    INF = float("inf")
    # states[i] = {cumulative_charge_total: (min_cost, prev_total, amount_used)}
    states: List[Dict[float, Tuple[float, float, float]]] = [dict() for _ in range(n + 1)]
    states[0][0.0] = (0.0, 0.0, 0.0)
    cum_dist = 0.0

    def soc_at(total_charged: float, dist_so_far: float) -> float:
        return vehicle.initial_soc_kWh + total_charged - cons * dist_so_far

    for i, (u, v) in enumerate(arcs):
        d = arcdist[(u, v)]
        eta = 1.0 if instance.charging.arc_enabled(u, v) else 0.0
        price = instance.prices.base_price_per_node.get(u, 0.0)
        opts = levels if eta > 0 else (0.0,)
        site = instance.charging.sites_by_node.get(u)
        if eta > 0 and site is not None and site.large_kWh is not None:
            opts = tuple(c for c in opts if c <= site.large_kWh + 1e-9)
        nxt = states[i + 1]
        for total, (c_here, _pt, _pa) in states[i].items():
            soc_here = soc_at(total, cum_dist)
            for amt in opts:
                after = soc_here + eta * amt
                if after > cap + 1e-9:          # would overcharge - forbidden
                    continue
                nsoc = after - cons * d
                if nsoc < lo - 1e-9:            # would breach the reserve floor
                    continue
                ntotal = round(total + eta * amt, 9)
                ncost = c_here + price * amt
                cur = nxt.get(ntotal)
                if cur is None or ncost < cur[0] - 1e-12:
                    nxt[ntotal] = (ncost, total, amt)
        cum_dist += d
        if not nxt:
            return ChargingPlan(False, {}, 0.0, [], 0,
                                f"no charging plan keeps SOC in [{lo}, {cap}] on this route "
                                f"(dead end at position {i + 1}/{n})")

    best_total, best_c = None, INF
    for total, (c, _p, _a) in states[n].items():
        if c < best_c:
            best_c, best_total = c, total
    if best_total is None:
        return ChargingPlan(False, {}, 0.0, [], 0,
                            f"no charging plan keeps SOC in [{lo}, {cap}] on this route")

    # reconstruct
    charge: Dict[Tuple[int, int], float] = {}
    amounts = [0.0] * n
    total = best_total
    for i in range(n, 0, -1):
        _c, prev_total, amt = states[i][total]
        amounts[i - 1] = amt
        total = prev_total
    for i, (u, v) in enumerate(arcs):
        if amounts[i] > 1e-9:
            charge[(u, v)] = amounts[i]

    # exact SOC walk for reporting
    prof = [vehicle.initial_soc_kWh]
    soc = vehicle.initial_soc_kWh
    for i, (u, v) in enumerate(arcs):
        eta = 1.0 if instance.charging.arc_enabled(u, v) else 0.0
        soc = soc + eta * amounts[i] - cons * arcdist[(u, v)]
        prof.append(soc)

    return ChargingPlan(True, charge, best_c, prof,
                        sum(1 for a in amounts if a > 1e-9))


def plan_is_soc_valid(plan: ChargingPlan, vehicle: Vehicle,
                      reserve_kWh: float = 0.0, tol: float = 1e-6) -> bool:
    """Independent re-check of the reconstructed plan's exact SOC walk."""
    if not plan.feasible:
        return False
    lo = max(0.0, reserve_kWh) - tol
    hi = vehicle.battery_capacity_kWh + tol
    return all(lo <= s <= hi for s in plan.soc_profile)
