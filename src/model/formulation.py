from __future__ import annotations

from typing import Dict, Optional, Tuple

from src.model.milp_ir import MILPProblem
from src.model.types import Instance, Request, Vehicle

# Builds one vehicle's MILP subproblem under Lagrangian relaxation of the
# assignment coupling constraint (10), sum_k q_u^k = 1, the only inter-vehicle
# coupling in the model. Everything else is per-vehicle: constraints (2)-(9)
# and (12) of the reference model, plus flow conservation, start/end pinning,
# SOC path-consistency, sub-tour elimination and precedence, each described
# where it is built.

INF = 1e30


class SubproblemVarIndex:
    def __init__(
        self,
        x: Dict[Tuple[int, int], int],
        c: Dict[Tuple[int, int], int],
        B: Dict[int, int],
        q: Dict[int, int],
        t: Dict[int, int],
        f: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> None:
        self.x = x
        self.c = c
        self.B = B
        self.q = q
        self.t = t
        self.f = f or {}


def build_vehicle_subproblem(
    instance: Instance,
    vehicle: Vehicle,
    requests: Tuple[Request, ...],
    multipliers: Optional[Dict[int, float]] = None,
    include_mtz: bool = True,
    include_soc: bool = True,
    include_redundant_67: bool = True,
    include_flow_conservation: bool = True,
    include_precedence: bool = False,
    include_gg_flow: bool = False,
    gg_steiner_demand: bool = False,
    mandatory_requests: Optional[set] = None,
    discrete_charge_levels: bool = False,
) -> Tuple[MILPProblem, SubproblemVarIndex]:
    """Build vehicle `vehicle`'s subproblem MILP.

    multipliers: dict request_id -> mu_u, the Lagrange multiplier on (10) for
    that request. Missing entries default to mu_u = 0. The objective is the
    per-vehicle distance and energy-cost terms minus sum_u mu_u * q_u^k (the
    relaxed coupling term); src/solve/lagrangian.py assembles the dual value
    from these subproblem optima.

    discrete_charge_levels: restrict each c_ij to {0} or one of the charging
    site's allowed levels (small/medium/large kWh) on a used arc, via one
    binary per level: c_ij = sum_l l*z_ijl, sum_l z_ijl <= x_ij. Off by
    default (continuous 0 <= c_ij <= large_kWh); the monolithic MILP turns it
    on so it uses the same discrete levels as the router, the charging DP and
    the checker.
    """
    mu = multipliers or {}
    nodes = instance.nodes
    s_node = vehicle.start_node
    e_node = vehicle.end_node
    n_nodes = len(nodes)

    consumption_rate = vehicle.max_energy_consumption_kWh_per_km
    if consumption_rate is None:
        consumption_rate = instance.energy.consumption_kWh_per_km

    # Three constraint families use big-M, each with its own tight valid value.
    # The instance's BigM (1000) is sized for (8)/(9); reusing it elsewhere
    # makes the LP relaxation much looser. Each value below is valid: it never
    # cuts off a feasible point.
    M_assign = instance.big_M  # (8)/(9): the instance's value

    # (6)/(7): outflow/inflow sums are bounded by the node's degree; the max
    # degree over the graph is a valid, tight M for the whole family.
    max_degree = 1
    for node_id in nodes:
        max_degree = max(
            max_degree,
            len(instance.out_arcs.get(node_id, ())),
            len(instance.in_arcs.get(node_id, ())),
        )
    M_flow = float(max_degree)

    # SOC linking: when x_ij=0, B_j - B_i - eta*c_ij ranges within
    # [-(battery_cap + max_charge), battery_cap]; battery_cap + max_charge is
    # the tight valid M.
    max_charge_amt = 0.0
    for site in instance.charging.sites_by_node.values():
        if site.large_kWh is not None:
            max_charge_amt = max(max_charge_amt, site.large_kWh)
    M_soc = vehicle.battery_capacity_kWh + max_charge_amt

    problem = MILPProblem(name=f"subproblem_{vehicle.id}", minimize=True)

    x_idx: Dict[Tuple[int, int], int] = {}
    c_idx: Dict[Tuple[int, int], int] = {}
    B_idx: Dict[int, int] = {}
    q_idx: Dict[int, int] = {}
    t_idx: Dict[int, int] = {}

    # --- variables: x_ij (routing), c_ij (charging amount) ---
    # The price is the static base price at the arc's origin node; the hourly
    # multipliers in price_schedule.csv are not used.
    for arc in instance.arcs:
        key = (arc.from_node, arc.to_node)
        # Start/end pinning: a vehicle may only leave the start node and only
        # enter the end node. edges_directed.csv contains arcs into the start
        # and out of the end; left usable, they allow start<->x and end<->y
        # 2-cycles that satisfy (2), (3) and flow conservation without an
        # s-e path. Those arcs are fixed to x = 0.
        forbidden = (arc.to_node == s_node) or (arc.from_node == e_node)
        x_idx[key] = problem.add_variable(
            name=f"x_{arc.from_node}_{arc.to_node}",
            lower=0.0,
            upper=0.0 if forbidden else 1.0,
            integer=True,
            obj=instance.objective.epsilon1 * arc.distance_km,
        )
        eta = 1.0 if instance.charging.arc_enabled(arc.from_node, arc.to_node) else 0.0
        c_upper = 0.0
        if eta > 0.0:
            site = instance.charging.sites_by_node.get(arc.from_node)
            if site is not None and site.large_kWh is not None:
                c_upper = site.large_kWh
        price = instance.prices.base_price_per_node.get(arc.from_node, 0.0)
        c_idx[key] = problem.add_variable(
            name=f"c_{arc.from_node}_{arc.to_node}",
            lower=0.0,
            upper=c_upper,
            obj=instance.objective.epsilon2 * price * eta,
        )
        if discrete_charge_levels and c_upper > 0.0:
            levels = [l for l in (site.small_kWh, site.medium_kWh, site.large_kWh) if l is not None]
            level_coeffs: Dict[int, float] = {c_idx[key]: 1.0}
            choose_coeffs: Dict[int, float] = {x_idx[key]: -1.0}
            for l in levels:
                z = problem.add_variable(
                    name=f"z_{arc.from_node}_{arc.to_node}_{l:g}",
                    lower=0.0, upper=1.0, integer=True, obj=0.0,
                )
                level_coeffs[z] = -l
                choose_coeffs[z] = 1.0
            # c_ij - sum_l l*z_ijl = 0 ;  sum_l z_ijl - x_ij <= 0
            problem.add_constraint(f"charge_level_{arc.from_node}_{arc.to_node}", level_coeffs, 0.0, 0.0)
            problem.add_constraint(f"charge_used_{arc.from_node}_{arc.to_node}", choose_coeffs, -INF, 0.0)
        # The battery-degradation objective term is not built: its
        # parameters are not in data/.

    # --- variables: B_i (battery level at node i), bounded to [0, capacity] ---
    for node_id in nodes:
        B_idx[node_id] = problem.add_variable(
            name=f"B_{node_id}", lower=0.0, upper=vehicle.battery_capacity_kWh, obj=0.0
        )

    # --- variables: q_u (assignment flag, relaxed coupling) ---
    # mandatory_requests fixes q_u = 1, so the vehicle must serve those requests.
    mandatory = mandatory_requests or set()
    for req in requests:
        q_idx[req.id] = problem.add_variable(
            name=f"q_{req.id}",
            lower=1.0 if req.id in mandatory else 0.0,
            upper=1.0,
            integer=True,
            obj=-mu.get(req.id, 0.0),
        )

    # --- variables: t_i (MTZ position), one per node != start ---
    for node_id in nodes:
        if node_id == s_node:
            continue
        t_idx[node_id] = problem.add_variable(
            name=f"t_{node_id}", lower=1.0, upper=float(n_nodes - 1), obj=0.0
        )

    # --- (2): start node has exactly one outflow ---
    out_s = instance.out_arcs.get(s_node, ())
    problem.add_constraint(
        "c2_start_outflow",
        {x_idx[(s_node, arc.to_node)]: 1.0 for arc in out_s},
        1.0,
        1.0,
    )

    # --- (3): end node has exactly one inflow ---
    in_e = instance.in_arcs.get(e_node, ())
    problem.add_constraint(
        "c3_end_inflow",
        {x_idx[(arc.from_node, e_node)]: 1.0 for arc in in_e},
        1.0,
        1.0,
    )

    # --- (4)/(5): interior nodes have out-degree/in-degree <= 1 ---
    for node_id in nodes:
        if node_id in (s_node, e_node):
            continue
        out_i = instance.out_arcs.get(node_id, ())
        if out_i:
            problem.add_constraint(
                f"c4_outdeg_{node_id}",
                {x_idx[(node_id, arc.to_node)]: 1.0 for arc in out_i},
                0.0,
                1.0,
            )
        in_i = instance.in_arcs.get(node_id, ())
        if in_i:
            problem.add_constraint(
                f"c5_indeg_{node_id}",
                {x_idx[(arc.from_node, node_id)]: 1.0 for arc in in_i},
                0.0,
                1.0,
            )

    # --- Flow conservation: inflow(i) = outflow(i) at every interior node.
    # Constraints (6)/(7) alone do not balance inflow and outflow; without
    # this row the solver can pick disconnected arc fragments while marking
    # requests served.
    for node_id in (nodes if include_flow_conservation else ()):
        if node_id in (s_node, e_node):
            continue
        out_i = instance.out_arcs.get(node_id, ())
        in_i = instance.in_arcs.get(node_id, ())
        if not out_i and not in_i:
            continue
        coeffs: Dict[int, float] = {}
        for oarc in out_i:
            key = x_idx[(node_id, oarc.to_node)]
            coeffs[key] = coeffs.get(key, 0.0) + 1.0
        for iarc in in_i:
            key = x_idx[(iarc.from_node, node_id)]
            coeffs[key] = coeffs.get(key, 0.0) - 1.0
        problem.add_constraint(f"flow_conservation_{node_id}", coeffs, 0.0, 0.0)

    # --- (6): big-M flow balance, one pair of rows per arc. The bound depends
    # on x_i_xi through the big-M term, so the two directions cannot share
    # one range row. With binary x these rows are implied by flow
    # conservation and the degree bounds; include_redundant_67 drops them.
    for arc in instance.arcs if include_redundant_67 else ():
        i, xi = arc.from_node, arc.to_node
        if i in (s_node, e_node) or xi == e_node:
            continue
        out_i = instance.out_arcs.get(i, ())
        base = {x_idx[(i, oarc.to_node)]: 1.0 for oarc in out_i}
        var_ixi = x_idx[(i, xi)]

        coeffs_a = dict(base)
        coeffs_a[var_ixi] = coeffs_a.get(var_ixi, 0.0) - (1.0 + M_flow)
        problem.add_constraint(f"c6a_{i}_{xi}", coeffs_a, -M_flow, INF)

        coeffs_b = dict(base)
        coeffs_b[var_ixi] = coeffs_b.get(var_ixi, 0.0) + (M_flow - 1.0)
        problem.add_constraint(f"c6b_{i}_{xi}", coeffs_b, -INF, M_flow)

    # --- (7): the same construction as (6), on inflow ---
    for arc in instance.arcs if include_redundant_67 else ():
        xi, j = arc.from_node, arc.to_node
        if xi == s_node or j in (s_node, e_node):
            continue
        in_j = instance.in_arcs.get(j, ())
        base = {x_idx[(iarc.from_node, j)]: 1.0 for iarc in in_j}
        var_xij = x_idx[(xi, j)]

        coeffs_a = dict(base)
        coeffs_a[var_xij] = coeffs_a.get(var_xij, 0.0) - (1.0 + M_flow)
        problem.add_constraint(f"c7a_{xi}_{j}", coeffs_a, -M_flow, INF)

        coeffs_b = dict(base)
        coeffs_b[var_xij] = coeffs_b.get(var_xij, 0.0) + (M_flow - 1.0)
        problem.add_constraint(f"c7b_{xi}_{j}", coeffs_b, -INF, M_flow)

    # --- (8): one row per request: if vehicle k picks up u, k also delivers u.
    #
    # The row is conditioned on q_u through big-M, as (9) is, so it binds only
    # when q_u = 1. Applied unconditionally it would stop a vehicle from
    # passing through an unserved request's pickup node, which is an ordinary
    # transit node, and would exclude feasible cheaper routes.
    for req in requests:
        out_p = instance.out_arcs.get(req.pickup_node, ())
        in_d = instance.in_arcs.get(req.delivery_node, ())
        coeffs: Dict[int, float] = {}
        for oarc in out_p:
            key = x_idx[(req.pickup_node, oarc.to_node)]
            coeffs[key] = coeffs.get(key, 0.0) + 1.0
        for iarc in in_d:
            key = x_idx[(iarc.from_node, req.delivery_node)]
            coeffs[key] = coeffs.get(key, 0.0) - 1.0
        if coeffs:
            # outflow(pickup) - inflow(delivery) <= M*(1-q_u)
            #   => outflow - inflow + M*q_u <= M
            q_var8 = q_idx[req.id]
            coeffs[q_var8] = coeffs.get(q_var8, 0.0) + M_assign
            problem.add_constraint(f"c8_{req.id}", coeffs, -INF, M_assign)

        # --- (9): service flag q_u forces both pickup and delivery visits ---
        coeffs9: Dict[int, float] = {}
        for oarc in out_p:
            key = x_idx[(req.pickup_node, oarc.to_node)]
            coeffs9[key] = coeffs9.get(key, 0.0) + 1.0
        for iarc in in_d:
            key = x_idx[(iarc.from_node, req.delivery_node)]
            coeffs9[key] = coeffs9.get(key, 0.0) + 1.0
        q_var = q_idx[req.id]
        # (9): A - 2 >= -M(1-q_u) rearranges to A - M*q_u >= 2 - M. The
        # coefficient on q_u must be -M, not +M: with +M the row is vacuous
        # and a vehicle can claim a request while visiting neither endpoint.
        # At q_u = 1 the row forces A >= 2, i.e. pickup outflow and delivery
        # inflow both equal 1 (each is <= 1 by (4)/(5)); at q_u = 0 it is slack.
        coeffs9[q_var] = coeffs9.get(q_var, 0.0) - M_assign
        problem.add_constraint(f"c9_{req.id}", coeffs9, 2.0 - M_assign, INF)

    # --- SOC path-consistency: B_j = B_i + eta*c_ij - e_ij when arc (i,j) is
    # used, linearized with big-M. ---
    for arc in instance.arcs if include_soc else ():
        i, j = arc.from_node, arc.to_node
        eta = 1.0 if instance.charging.arc_enabled(i, j) else 0.0
        e_ij = consumption_rate * arc.distance_km
        var_x = x_idx[(i, j)]
        var_c = c_idx[(i, j)]
        var_bi = B_idx[i]
        var_bj = B_idx[j]

        coeffs_a = {var_bj: 1.0, var_bi: -1.0, var_c: -eta, var_x: (e_ij - M_soc)}
        problem.add_constraint(f"soc_a_{i}_{j}", coeffs_a, -M_soc, INF)

        coeffs_b = {var_bj: 1.0, var_bi: -1.0, var_c: -eta, var_x: (e_ij + M_soc)}
        problem.add_constraint(f"soc_b_{i}_{j}", coeffs_b, -INF, M_soc)

    problem.add_constraint(
        "soc_initial", {B_idx[s_node]: 1.0}, vehicle.initial_soc_kWh, vehicle.initial_soc_kWh
    )

    # --- MTZ sub-tour elimination ---
    for arc in instance.arcs if include_mtz else ():
        i, j = arc.from_node, arc.to_node
        if i == s_node or j == s_node:
            continue
        coeffs = {
            t_idx[i]: 1.0,
            t_idx[j]: -1.0,
            x_idx[(i, j)]: float(n_nodes - 1),
        }
        problem.add_constraint(f"mtz_{i}_{j}", coeffs, -INF, float(n_nodes - 2))

    # --- Precedence through the MTZ position variables. ---
    # Requires include_mtz=True: t_i follows the path order only through the
    # MTZ rows above; without them the precedence rows would constrain
    # unlinked variables.
    if include_precedence:
        if not include_mtz:
            raise ValueError(
                "include_precedence=True requires include_mtz=True: precedence "
                "rows on t_i are meaningless without the per-arc linkage that "
                "makes t_i track actual path position."
            )
        m_prec = float(n_nodes - 1)
        for req in requests:
            if req.pickup_node == s_node or req.delivery_node == s_node:
                continue
            t_pick = t_idx[req.pickup_node]
            t_deliv = t_idx[req.delivery_node]
            q_var = q_idx[req.id]
            # t_deliv >= t_pick + 1 - M(1-q_u)  =>  t_deliv - t_pick - M*q_u >= 1 - M
            coeffs = {t_deliv: 1.0, t_pick: -1.0, q_var: -m_prec}
            problem.add_constraint(f"precedence_{req.id}", coeffs, 1.0 - m_prec, INF)

    # --- Gavish-Graves single-commodity flow: connectivity without a
    # separation loop. One continuous f_ij per arc, with f_ij <= flow_cap * x_ij.
    # A component cut off from the start has no entering flow, so any node in
    # it with positive demand cannot satisfy inflow - outflow = demand. This
    # forbids every disconnected set at once.
    #
    # The demand definition and flow_cap must be chosen together:
    #   gg_steiner_demand=False: every visited node (transit included)
    #     consumes one unit, so flow_cap must bound the longest possible route,
    #     n_nodes - 2. 2*len(requests) is not a valid cap here, because a
    #     route can visit more nodes than that.
    #   gg_steiner_demand=True: only a candidate request's pickup and delivery
    #     nodes consume q_u each; transit nodes conserve flow. Total demand is
    #     then at most 2*len(requests), which is a valid and tight cap. A
    #     transit-only cycle is not forbidden by the flow, but it costs
    #     distance and serves nothing, so the objective never builds one.
    f_idx: Dict[Tuple[int, int], int] = {}
    if include_gg_flow:
        node_demand_qvar: Dict[int, int] = {}
        if gg_steiner_demand:
            for req in requests:
                node_demand_qvar[req.pickup_node] = q_idx[req.id]
                node_demand_qvar[req.delivery_node] = q_idx[req.id]
            flow_cap = float(2 * len(requests)) if requests else 1.0
        else:
            flow_cap = float(n_nodes - 2)

        for arc in instance.arcs:
            key = (arc.from_node, arc.to_node)
            f_idx[key] = problem.add_variable(
                name=f"f_{arc.from_node}_{arc.to_node}", lower=0.0, upper=flow_cap, obj=0.0
            )

        # per-node demand equation: inflow_f - outflow_f - demand(i) = 0
        source_coeffs: Dict[int, float] = {}
        for node_id in nodes:
            if node_id in (s_node, e_node):
                continue
            out_i = instance.out_arcs.get(node_id, ())
            in_i = instance.in_arcs.get(node_id, ())
            coeffs: Dict[int, float] = {}
            for iarc in in_i:
                key = f_idx[(iarc.from_node, node_id)]
                coeffs[key] = coeffs.get(key, 0.0) + 1.0
            for oarc in out_i:
                key = f_idx[(node_id, oarc.to_node)]
                coeffs[key] = coeffs.get(key, 0.0) - 1.0
            if gg_steiner_demand:
                q_var = node_demand_qvar.get(node_id)
                if q_var is not None:
                    coeffs[q_var] = coeffs.get(q_var, 0.0) - 1.0
                    source_coeffs[q_var] = source_coeffs.get(q_var, 0.0) - 1.0
                # else: pure transit, demand=0, plain conservation - no term added
            else:
                for oarc in out_i:
                    key = x_idx[(node_id, oarc.to_node)]
                    coeffs[key] = coeffs.get(key, 0.0) - 1.0
                    source_coeffs[key] = source_coeffs.get(key, 0.0) - 1.0
            if coeffs:
                problem.add_constraint(f"gg_demand_{node_id}", coeffs, 0.0, 0.0)

        # source: outflow_flow(s) = total demand over all i != s, e, built from
        # the same terms as the per-node rows so both use the active demand
        # definition. Inflow into s is zero (arcs into s are pinned to x = 0)
        # and is included for completeness.
        out_s = instance.out_arcs.get(s_node, ())
        in_s = instance.in_arcs.get(s_node, ())
        coeffs = dict(source_coeffs)
        for oarc in out_s:
            key = f_idx[(s_node, oarc.to_node)]
            coeffs[key] = coeffs.get(key, 0.0) + 1.0
        for iarc in in_s:
            key = f_idx[(iarc.from_node, s_node)]
            coeffs[key] = coeffs.get(key, 0.0) - 1.0
        if coeffs:
            problem.add_constraint("gg_source", coeffs, 0.0, 0.0)

        # coupling: f_ij <= flow_cap * x_ij, i.e. f_ij - flow_cap*x_ij <= 0
        for arc in instance.arcs:
            key = (arc.from_node, arc.to_node)
            coeffs = {f_idx[key]: 1.0, x_idx[key]: -flow_cap}
            problem.add_constraint(f"gg_couple_{arc.from_node}_{arc.to_node}", coeffs, -INF, 0.0)

    return problem, SubproblemVarIndex(x=x_idx, c=c_idx, B=B_idx, q=q_idx, t=t_idx, f=f_idx)
