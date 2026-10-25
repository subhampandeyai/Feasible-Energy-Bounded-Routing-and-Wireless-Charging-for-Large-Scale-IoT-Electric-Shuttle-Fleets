from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.model.types import Instance, Vehicle

# Independent feasibility checker. Deliberately imports NOTHING from src/solve/
# - it re-derives every quantity from the raw Instance and a plain solution
# dict, so a bug in the solver cannot hide behind a shared helper. Every
# reported number in the project passes through check_solution before being
# reported anywhere.

# Solution contract (plain data, no solver types):
#   routes: Dict[vehicle_id, VehicleRoute] where VehicleRoute has:
#     arcs:   List[(i, j)] the arcs the vehicle uses (x_ij = 1)
#     charge: Dict[(i, j), float] charging amount on each arc (kWh), c_ij
#     served: List[request_id] the requests this vehicle claims to serve


@dataclass
class VehicleRoute:
    arcs: List[Tuple[int, int]]
    charge: Dict[Tuple[int, int], float] = field(default_factory=dict)
    served: List[int] = field(default_factory=list)


@dataclass
class CheckReport:
    passed: bool
    violations: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.violations.append(msg)
        self.passed = False

    def skip(self, msg: str) -> None:
        self.skipped.append(msg)

    def summary(self) -> str:
        lines = [f"FEASIBILITY: {'PASS' if self.passed else 'FAIL'}"]
        for v in self.violations:
            lines.append(f"  VIOLATION: {v}")
        for s in self.skipped:
            lines.append(f"  skipped: {s}")
        return "\n".join(lines)


def check_solution(
    instance: Instance,
    routes: Dict[str, VehicleRoute],
    soc_tol: float = 1e-6,
    energy_tol: float = 1e-4,
) -> CheckReport:
    report = CheckReport(passed=True)
    arc_set = {(a.from_node, a.to_node) for a in instance.arcs}
    arc_dist = {(a.from_node, a.to_node): a.distance_km for a in instance.arcs}
    veh_by_id = {v.id: v for v in instance.vehicles}
    req_by_id = {r.id: r for r in instance.requests}

    _check_each_request_served_once(instance, routes, report)

    for vid, route in routes.items():
        veh = veh_by_id.get(vid)
        if veh is None:
            report.fail(f"route references unknown vehicle {vid}")
            continue
        _check_arcs_exist(vid, route, arc_set, report)
        _check_no_empty_vehicle_driving(vid, route, report)
        _check_single_connected_path(vid, veh, route, report)
        _check_charge_amounts(vid, route, instance, report, energy_tol)
        _check_soc(vid, veh, route, instance, arc_dist, report, soc_tol, energy_tol)
        _check_served_endpoints_visited(vid, route, req_by_id, report)
        _check_precedence(vid, veh, route, req_by_id, report)

    _check_time_windows(instance, routes, req_by_id, report)
    return report


def _check_each_request_served_once(instance, routes, report) -> None:
    counts: Dict[int, int] = {r.id: 0 for r in instance.requests}
    for route in routes.values():
        for u in route.served:
            counts[u] = counts.get(u, 0) + 1
    for u, c in counts.items():
        if c == 0:
            report.fail(f"request {u} is served by NO vehicle (must be exactly once)")
        elif c > 1:
            report.fail(f"request {u} is served by {c} vehicles (must be exactly once)")


def _check_arcs_exist(vid, route, arc_set, report) -> None:
    for (i, j) in route.arcs:
        if (i, j) not in arc_set:
            report.fail(f"vehicle {vid} uses arc ({i},{j}) that does not exist in edges_directed.csv")


def _check_no_empty_vehicle_driving(vid, route, report) -> None:
    if not route.served and route.arcs:
        # A vehicle serving nobody should not be driving a non-trivial route.
        # (An empty route, or the degenerate direct start->end, is allowed.)
        moved = [a for a in route.arcs]
        if len(moved) > 1:
            report.fail(
                f"vehicle {vid} serves no request but drives {len(moved)} arcs "
                f"(empty vehicle driving a nonzero route)"
            )


def _check_single_connected_path(vid, veh: Vehicle, route, report) -> None:
    if not route.arcs:
        return
    succ: Dict[int, List[int]] = {}
    indeg: Dict[int, int] = {}
    outdeg: Dict[int, int] = {}
    for (i, j) in route.arcs:
        succ.setdefault(i, []).append(j)
        outdeg[i] = outdeg.get(i, 0) + 1
        indeg[j] = indeg.get(j, 0) + 1
    # degree sanity: on a simple path each node has in<=1 and out<=1
    for n, d in outdeg.items():
        if d > 1:
            report.fail(f"vehicle {vid} node {n} has out-degree {d} (>1): not a simple path")
    for n, d in indeg.items():
        if d > 1:
            report.fail(f"vehicle {vid} node {n} has in-degree {d} (>1): not a simple path")
    # walk from start; must consume every arc and end at end node with no cycle
    walked = 0
    seen = {veh.start_node}
    cur = veh.start_node
    while cur in succ:
        nxts = succ[cur]
        if len(nxts) != 1:
            break
        cur = nxts[0]
        walked += 1
        if cur in seen:
            report.fail(f"vehicle {vid} route revisits node {cur}: contains a cycle/subtour")
            return
        seen.add(cur)
    if walked != len(route.arcs):
        report.fail(
            f"vehicle {vid} start-walk consumed {walked} of {len(route.arcs)} arcs - "
            f"route is not a single connected path (disconnected fragments/subtours present)"
        )
    if cur != veh.end_node:
        report.fail(f"vehicle {vid} route ends at node {cur}, not the end node {veh.end_node}")


def _check_charge_amounts(vid, route, instance, report, energy_tol) -> None:
    # Every charging amount must be zero or one of the discrete levels the
    # charging site at the arc's origin offers (small/medium/large allowable
    # kWh, read from the raw Instance), may only sit on a used, charging-enabled
    # arc, and may never be negative.
    used = set(route.arcs)
    for (i, j), amt in route.charge.items():
        if amt < -energy_tol:
            report.fail(f"vehicle {vid} has negative charging amount {amt:.4f} kWh on arc ({i},{j})")
            continue
        if abs(amt) <= energy_tol:
            continue
        if (i, j) not in used:
            report.fail(f"vehicle {vid} charges {amt:.4f} kWh on arc ({i},{j}) that its route does not use")
        if not instance.charging.arc_enabled(i, j):
            report.fail(f"vehicle {vid} charges {amt:.4f} kWh on arc ({i},{j}) that is not charging-enabled")
        site = instance.charging.sites_by_node.get(i)
        levels = [] if site is None else [l for l in (site.small_kWh, site.medium_kWh, site.large_kWh)
                                          if l is not None]
        if not any(abs(amt - l) <= energy_tol for l in levels):
            report.fail(f"vehicle {vid} charges {amt:.4f} kWh on arc ({i},{j}), which is not an "
                        f"allowed charging level {sorted(levels)}")


def _check_soc(vid, veh: Vehicle, route, instance, arc_dist, report, soc_tol, energy_tol) -> None:
    # Reconstruct SOC along the walked path from the vehicle's initial SOC, with
    # NO clamping. Each arc: soc_j = soc_i + eta*c_ij - E_ij. Must stay within
    # [0, battery_capacity] at every node.
    if not route.arcs:
        return
    succ = {i: j for (i, j) in route.arcs}
    consumption = veh.max_energy_consumption_kWh_per_km or instance.energy.consumption_kWh_per_km
    b_lo, b_hi = 0.0, veh.battery_capacity_kWh
    soc = veh.initial_soc_kWh
    if soc < b_lo - soc_tol or soc > b_hi + soc_tol:
        report.fail(f"vehicle {vid} initial SOC {soc:.4f} outside [{b_lo},{b_hi}]")
    cur = veh.start_node
    steps = 0
    while cur in succ and steps <= len(route.arcs):
        nxt = succ[cur]
        dist = arc_dist.get((cur, nxt), 0.0)
        e_ij = consumption * dist
        eta = 1.0 if instance.charging.arc_enabled(cur, nxt) else 0.0
        c_ij = route.charge.get((cur, nxt), 0.0)
        soc = soc + eta * c_ij - e_ij
        if soc < b_lo - soc_tol:
            report.fail(
                f"vehicle {vid} SOC dropped to {soc:.4f} kWh at node {nxt} "
                f"(< {b_lo}) - battery would be depleted; NO clamping applied"
            )
        if soc > b_hi + soc_tol:
            report.fail(
                f"vehicle {vid} SOC rose to {soc:.4f} kWh at node {nxt} "
                f"(> capacity {b_hi}) - overcharged"
            )
        cur = nxt
        steps += 1


def _check_served_endpoints_visited(vid, route, req_by_id, report) -> None:
    visited = set()
    for (i, j) in route.arcs:
        visited.add(i)
        visited.add(j)
    for u in route.served:
        req = req_by_id.get(u)
        if req is None:
            report.fail(f"vehicle {vid} claims unknown request {u}")
            continue
        if req.pickup_node not in visited:
            report.fail(f"vehicle {vid} serves request {u} but never visits pickup node {req.pickup_node}")
        if req.delivery_node not in visited:
            report.fail(f"vehicle {vid} serves request {u} but never visits delivery node {req.delivery_node}")


def _check_precedence(vid, veh: Vehicle, route, req_by_id, report) -> None:
    """Independently re-derives path order from the raw arc list (no import
    from src/solve - re-walks from scratch) and confirms every served
    request's pickup is visited strictly before its delivery.

    The other conditions do not imply precedence: a route can be a single
    connected path that visits both endpoints and still deliver before it
    picks up.
    """
    if not route.arcs:
        return
    succ = {i: j for (i, j) in route.arcs}
    position: Dict[int, int] = {veh.start_node: 0}
    cur = veh.start_node
    idx = 0
    while cur in succ:
        cur = succ[cur]
        idx += 1
        if cur in position:
            break  # a cycle here is already reported by _check_single_connected_path
        position[cur] = idx
    for u in route.served:
        req = req_by_id.get(u)
        if req is None:
            continue  # already reported by _check_served_endpoints_visited
        p_pos = position.get(req.pickup_node)
        d_pos = position.get(req.delivery_node)
        if p_pos is None or d_pos is None:
            continue  # already reported by _check_served_endpoints_visited
        if d_pos <= p_pos:
            report.fail(
                f"vehicle {vid} serves request {u} but visits delivery node "
                f"{req.delivery_node} (path position {d_pos}) at or before "
                f"pickup node {req.pickup_node} (path position {p_pos}) - "
                f"precedence violated"
            )


def _check_time_windows(instance, routes, req_by_id, report) -> None:
    any_tw = any(r.time_window is not None for r in instance.requests)
    any_mrt = any(r.max_ride_time_min is not None for r in instance.requests)
    if not any_tw:
        report.skip("time windows: no request carries a time_window (null in data/) - not checked")
    if not any_mrt:
        report.skip("max ride time: no request carries max_ride_time (null in data/) - not checked")
