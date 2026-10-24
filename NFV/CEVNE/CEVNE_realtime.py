#!/usr/bin/env python3
"""Congestion-aware and energy-aware VNE (CEVNE).

This program adapts the supplied D-ViNE implementation to the CEVNE model
in Pham et al., IEEE/ACM ToN, 2020.  It keeps the augmented graph, LP
relaxation and D-ViNE rounding, while adding:
  * an active-node (energy) objective;
  * a maximum projected link-utilisation constraint (R = 0.9 by default);
  * congestion-aware shortest/alternative-path link mapping; and
  * an active-node registry whose zero-reference nodes may sleep.

The substrate network is directed in this project.  Consequently, (u, v) and
(v, u) have independent capacities and independent congestion ratios.
"""

from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]

import math
import os
import pickle

import numpy as np
import pyomo.environ as pyo
import pymysql
from NFV.db_config import connect_database
import yaml

from NFV.network_flow.Mutiply_VNR import MultiVNRFlowSimulator
from NFV.network_flow.TM_generator import TrafficMatrixCalculator
from NFV.util.normal_method import (
    load_all_paths_numpy,
    load_substrate_nodes_from_db,
    load_vnet_from_file,
)
from NFV.util.yingshe import (
    build_changed_edges,
    integrate_slink_result,
    rollback_link_resources,
    rollback_node_resources,
    update_link_resources,
    update_node_resources,
)
from NFV.util.yaml import record_resource_usage_yaml
from NFV.virtual_net_atttribute import Network_node_atrribute


CONGESTION_LIMIT_R = 1
EPSILON = 1e-6
VNET_DATASET_DIR = str(_PROJECT_ROOT / 'NFV/vnet_dataset')
EVENT_FILE = "vnr_events.yaml"
PATH_DATABASE_FILE = str(_PROJECT_ROOT / 'NFV/p_net_directed.pkl')


def edge_bandwidth(info):
    """Return the bandwidth value from either a database dict or a number."""
    if isinstance(info, dict):
        return float(info.get("bw_resource", info.get("bw", 0.0)))
    return float(info)


def build_directed_edges(edges):
    return {
        (int(u), int(v)): edge_bandwidth(info)
        for (u, v), info in edges.items()
        if int(u) != int(v)
    }


def snapshot_total_link_bandwidth(initial_edges):
    """Take this once, before the first VNR arrives; values never change."""
    total = build_directed_edges(initial_edges)

    if not total or any(bw <= 0 for bw in total.values()):
        raise ValueError("Unable to record positive initial total link bandwidths.")
    return total


def build_candidate_nodes_cevne(substrate_nodes, virtual_nodes, distance_limit=10):
    """CPU-feasible candidates, retaining the project's optional location rule."""
    candidates = {}
    for virtual_node in virtual_nodes:
        v = int(virtual_node["id"])
        cpu_need = float(virtual_node["cpu"])
        vx, vy = float(virtual_node["pos_x"]), float(virtual_node["pos_y"])
        candidates[v] = []

        for s, substrate_node in substrate_nodes.items():
            s = int(s)
            if float(substrate_node["cpu"]) < cpu_need:
                continue
            if distance_limit is not None:
                distance = math.hypot(vx - float(substrate_node["pos"]),
                                      vy - float(substrate_node["pos2"]))
                if distance > distance_limit:
                    continue
            candidates[v].append(s)

        if not candidates[v]:
            print(f"❌ Virtual node {v} has no CEVNE candidate substrate node")
            return None
    return candidates


def build_cevne_lp_model(
    substrate_nodes,
    residual_edges,
    total_link_bw,
    virtual_nodes,
    virtual_edges,
    distance_limit=10,
    congestion_limit=CONGESTION_LIMIT_R,
    cost_weight=1.0,
    energy_weight=1.0,
    one_vn_per_sn=True,
):
    """Build CEVNE's relaxed augmented-graph LP for a directed substrate."""
    if not 0 < congestion_limit <= 1:
        raise ValueError("congestion_limit R must be in (0, 1].")

    candidates = build_candidate_nodes_cevne(substrate_nodes, virtual_nodes, distance_limit)
    if candidates is None:
        return None, None

    V = [int(v["id"]) for v in virtual_nodes]
    S = [int(s) for s in substrate_nodes]
    cpu_demand = {int(v["id"]): float(v["cpu"]) for v in virtual_nodes}
    residual_bw = build_directed_edges(residual_edges)

    # The initial snapshot must cover every currently available directed edge.
    missing_total = set(residual_bw) - set(total_link_bw)
    if missing_total:
        raise KeyError(f"Initial total capacity missing for links: {sorted(missing_total)}")

    Ev = [(int(e["source"]), int(e["target"])) for e in virtual_edges]
    bw_demand = {(int(e["source"]), int(e["target"])): float(e["bw"])
                 for e in virtual_edges}

    meta_name = {v: f"meta_{v}" for v in V}
    meta_vs = [(v, s) for v in V for s in candidates[v]]
    arcs = list(residual_bw)
    for v, s in meta_vs:
        arcs.extend([(meta_name[v], s), (s, meta_name[v])])
    arcs = list(dict.fromkeys(arcs))
    augmented_nodes = S + list(meta_name.values())
    big_m = max(sum(bw_demand.values()), 1.0)

    model = pyo.ConcreteModel()
    model.V = pyo.Set(initialize=V)
    model.S = pyo.Set(initialize=S)
    model.Ev = pyo.Set(initialize=Ev, dimen=2)
    model.A = pyo.Set(initialize=arcs, dimen=2)
    model.MetaVS = pyo.Set(initialize=meta_vs, dimen=2)
    model.RealE = pyo.Set(initialize=list(residual_bw), dimen=2)

    # x and active are relaxed exactly as in CEVNE. r represents the maximum
    # projected utilisation after placing this VNR.
    model.x = pyo.Var(model.MetaVS, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.flow = pyo.Var(model.Ev, model.A, domain=pyo.NonNegativeReals)
    # ``active`` is a reserved Pyomo Block attribute, so use a non-reserved
    # component name for the LP active-node variable.
    model.node_active = pyo.Var(model.S, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.r = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, congestion_limit))

    def objective_rule(m):
        link_cost = sum(
            m.flow[k, (i, j)] / (residual_bw[(i, j)] + EPSILON)
            for k in m.Ev for (i, j) in m.RealE
        )
        node_cost = sum(
            cpu_demand[v] * m.x[v, s] / (float(substrate_nodes[s]["cpu"]) + EPSILON)
            for (v, s) in m.MetaVS
        )
        energy_cost = sum(m.node_active[s] for s in m.S) / max(len(S), 1)
        return cost_weight * (link_cost + node_cost) + energy_weight * energy_cost

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    model.one_candidate = pyo.Constraint(
        model.V, rule=lambda m, v: sum(m.x[v, s] for s in candidates[v]) == 1
    )

    def cpu_rule(m, s):
        eligible_virtual_nodes = [v for v in V if (v, s) in meta_vs]
        # A substrate node outside every candidate cluster has no CPU decision
        # variable.  Returning Feasible avoids Python evaluating ``0 <= cpu``
        # to a plain True before Pyomo can construct the constraint.
        if not eligible_virtual_nodes:
            return pyo.Constraint.Feasible
        return sum(cpu_demand[v] * m.x[v, s] for v in eligible_virtual_nodes) <= float(substrate_nodes[s]["cpu"])
    model.cpu_capacity = pyo.Constraint(model.S, rule=cpu_rule)

    if one_vn_per_sn:
        def unique_node_rule(m, s):
            eligible_virtual_nodes = [v for v in V if (v, s) in meta_vs]
            if not eligible_virtual_nodes:
                return pyo.Constraint.Feasible
            return sum(m.x[v, s] for v in eligible_virtual_nodes) <= 1
        model.unique_node = pyo.Constraint(
            model.S,
            rule=unique_node_rule,
        )

    # A node carrying a virtual node must be active.  Its minimisation is the
    # energy-aware component.  A continuous variable is intentional: CEVNE
    # solves the LP relaxation and subsequently applies D-ViNE rounding.
    model.activation = pyo.Constraint(
        model.MetaVS, rule=lambda m, v, s: m.x[v, s] <= m.node_active[s]
    )

    def congestion_rule(m, i, j):
        total = total_link_bw[(i, j)]
        current_load = max(0.0, total - residual_bw[(i, j)])
        # current load + this request's LP flow <= r * total capacity.
        return current_load + sum(m.flow[k, (i, j)] for k in m.Ev) <= m.r * total
    model.congestion = pyo.Constraint(model.RealE, rule=congestion_rule)

    # Keep a separate hard residual-resource check for numerical clarity.
    model.residual_bandwidth = pyo.Constraint(
        model.RealE,
        rule=lambda m, i, j: sum(m.flow[k, (i, j)] for k in m.Ev) <= residual_bw[(i, j)],
    )

    def meta_coupling_rule(m, v, s):
        meta = meta_name[v]
        return sum(
            m.flow[k, (meta, s)] + m.flow[k, (s, meta)]
            for k in m.Ev
        ) <= big_m * m.x[v, s]
    model.meta_coupling = pyo.Constraint(model.MetaVS, rule=meta_coupling_rule)

    def flow_conservation_rule(m, u, v, node):
        source, target = meta_name[u], meta_name[v]
        outgoing = sum(m.flow[(u, v), (a, b)] for (a, b) in m.A if a == node)
        incoming = sum(m.flow[(u, v), (a, b)] for (a, b) in m.A if b == node)
        if node == source:
            return outgoing - incoming == bw_demand[(u, v)]
        if node == target:
            return outgoing - incoming == -bw_demand[(u, v)]
        return outgoing - incoming == 0
    model.flow_conservation = pyo.Constraint(model.Ev, augmented_nodes, rule=flow_conservation_rule)

    return model, {
        "candidates": candidates,
        "meta_name": meta_name,
        "residual_bw": residual_bw,
        "congestion_limit": congestion_limit,
    }


def solve_cevne_node_mapping(
    substrate_nodes, residual_edges, total_link_bw, virtual_nodes, virtual_edges,
    distance_limit=10, congestion_limit=CONGESTION_LIMIT_R, solver_name="glpk",
    one_vn_per_sn=True,
):
    """Solve relaxed CEVNE and apply the D-ViNE meta-link-flow rounding rule."""
    model, extra = build_cevne_lp_model(
        substrate_nodes, residual_edges, total_link_bw, virtual_nodes, virtual_edges,
        distance_limit=distance_limit, congestion_limit=congestion_limit,
        one_vn_per_sn=one_vn_per_sn,
    )
    if model is None:
        return False

    result = pyo.SolverFactory(solver_name).solve(model, tee=False)
    if result.solver.status != pyo.SolverStatus.ok or result.solver.termination_condition not in (
        pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible,
    ):
        print("❌ CEVNE LP has no feasible solution:", result.solver.termination_condition)
        return False

    used = set()
    mapping = {}
    for v in model.V:
        scores = {}
        meta = extra["meta_name"][v]
        for s in extra["candidates"][v]:
            if one_vn_per_sn and s in used:
                continue
            meta_flow = sum(
                pyo.value(model.flow[k, (meta, s)]) + pyo.value(model.flow[k, (s, meta)])
                for k in model.Ev
            )
            scores[s] = pyo.value(model.x[v, s]) * meta_flow
        if not scores:
            print(f"❌ CEVNE rounding has no candidate for virtual node {v}")
            return False
        selected = max(scores, key=scores.get)
        mapping[int(v)] = int(selected)
        used.add(int(selected))

    projected_r = pyo.value(model.r)
    print(f"✔ CEVNE LP + D-ViNE rounding complete: {mapping}; LP r={projected_r:.4f}")
    return mapping


def convert_virtual_links_to_substrate_links(node_mapping, virtual_edges):
    links = []
    for edge in virtual_edges:
        v_src, v_dst = int(edge["source"]), int(edge["target"])
        if v_src not in node_mapping or v_dst not in node_mapping:
            return False
        links.append({"source": node_mapping[v_src], "target": node_mapping[v_dst], "bw": float(edge["bw"])})
    return links


def path_nodes(path):
    if not path:
        return set()
    return {int(path[0][0])} | {int(v) for _, v in path}


def path_is_non_congested(path, demand, temporary_residual, total_link_bw, congestion_limit):
    """Check residual capacity and projected total utilisation for all path links."""
    for u, v in path:
        edge = (int(u), int(v))
        total = total_link_bw.get(edge)
        residual = temporary_residual.get(edge, 0.0)
        if total is None or residual + EPSILON < demand:
            return False
        projected_utilisation = (total - residual + demand) / total
        if projected_utilisation > congestion_limit + EPSILON:
            return False
    return True


def map_slinks_cevne(slink_list, residual_edges, path_database, total_link_bw, congestion_limit=CONGESTION_LIMIT_R):
    """CEVNE LiM: shortest non-congested path first, then non-congested alternatives."""
    temporary_residual = build_directed_edges(residual_edges)
    mapped = []
    for link in sorted(slink_list, key=lambda item: item["bw"], reverse=True):
        source, target, demand = int(link["source"]), int(link["target"]), float(link["bw"])
        record = path_database.get((source, target))
        if record is None:
            print(f"❌ No path database entry for {source} → {target}")
            return False

        hops = list(record["hop"])
        paths = list(record["path"])
        candidates = list(zip(hops, paths))
        if not candidates:
            return False
        shortest_hop = min(int(hop) for hop, _ in candidates)
        shortest = [(int(hop), path) for hop, path in candidates if int(hop) == shortest_hop]
        alternatives = sorted(
            [(int(hop), path) for hop, path in candidates if int(hop) != shortest_hop],
            key=lambda item: item[0],
        )

        selected = None
        for hop, path in shortest + alternatives:
            if path_is_non_congested(path, demand, temporary_residual, total_link_bw, congestion_limit):
                selected = (hop, path)
                break
        if selected is None:
            print(f"❌ All candidate paths are congested for {source} → {target}, bw={demand}")
            return False

        hop, path = selected
        bottleneck = min(temporary_residual[(int(u), int(v))] for u, v in path)
        for u, v in path:
            temporary_residual[(int(u), int(v))] -= demand
        mapped.append({
            "source": source, "target": target, "bw": demand, "hop": hop,
            "path_bottleneck": float(bottleneck), "linkpath": path,
        })
    return mapped


def active_nodes_for_embedding(node_mapping, link_mapping):
    """Registry includes virtual-node hosts and every transit node on selected paths."""
    nodes = {int(s) for s in node_mapping.values()}
    for link in link_mapping:
        nodes.update(path_nodes(link["linkpath"]))
    return nodes


def register_active_nodes(active_refcounts, nodes):
    for node in nodes:
        active_refcounts[node] = active_refcounts.get(node, 0) + 1


def unregister_active_nodes(active_refcounts, nodes):
    for node in nodes:
        remaining = active_refcounts.get(node, 0) - 1
        if remaining > 0:
            active_refcounts[node] = remaining
        else:
            active_refcounts.pop(node, None)


def append_one_record_to_yaml(link_mapping, vnr_id, event_time, event_id, file_path="cevne_mapping.yaml"):
    global mapping_record_count
    mapping_record_count += 1
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or []
    data.append({
        "id": mapping_record_count, "vnr_ind": vnr_id, "event_id": event_id,
        "event_time": float(event_time),
        "links": [{"source": item["source"], "target": item["target"], "bw": float(item["bw"]), "linkpath": item["linkpath"]} for item in link_mapping],
    })
    with open(file_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def handle_arrival(event):
    vnr_id, event_time, event_id = event["v_net_id"], event["time"], event["id"]
    print(f"\n🟢 CEVNE arrival: VNR {vnr_id} at t={event_time:.2f}")
    v_nodes, v_edges = load_vnet_from_file(os.path.join(VNET_DATASET_DIR, f"vinre0.5_node_{vnr_id}.yaml"))
    s_nodes = load_substrate_nodes_from_db(conn)
    residual_edges = node_function.load_edges_from_db()

    node_mapping = solve_cevne_node_mapping(
        s_nodes, residual_edges, total_link_bandwidth, v_nodes, v_edges,
        distance_limit=10, congestion_limit=CONGESTION_LIMIT_R, solver_name="glpk",
        one_vn_per_sn=True,
    )
    if node_mapping is False:
        return False

    substrate_links = convert_virtual_links_to_substrate_links(node_mapping, v_edges)
    link_mapping = map_slinks_cevne(
        substrate_links, residual_edges, s_edges2, total_link_bandwidth, CONGESTION_LIMIT_R
    )
    if link_mapping is False:
        return False

    changed_edges = build_changed_edges(link_mapping)
    update_link_resources(conn, changed_edges, s_edges2)
    update_node_resources(conn, node_mapping, v_nodes)
    active_nodes = active_nodes_for_embedding(node_mapping, link_mapping)
    register_active_nodes(active_node_refcounts, active_nodes)

    embedded_id = integrate_slink_result(total_s_link_result, link_mapping)
    active_vnrs[embedded_id] = {
        "vnr_id": vnr_id, "node_mapping": node_mapping, "link_mapping": link_mapping,
        "active_nodes": active_nodes,
    }
    embedded_vnr_history.append({
        "event_id": embedded_id, "vnr_id": vnr_id, "arrival_time": event_time,
        "node_mapping": node_mapping, "link_mapping": link_mapping,
        "changed_edges": changed_edges, "status": "active",
    })
    multi_simulator.add_vnr(vnr_id=embedded_id, s_link_result=link_mapping, vnr_id2=vnr_id)
    append_one_record_to_yaml(link_mapping, vnr_id, event_time, event_id)
    print(f"✔ CEVNE VNR {vnr_id} embedded; active substrate nodes: {sorted(active_node_refcounts)}")
    return True


def handle_leave(event):
    vnr_id, event_time = event["v_net_id"], event["time"]
    print(f"\n🔴 CEVNE leave: VNR {vnr_id} at t={event_time:.2f}")
    embedded_id = next((eid for eid, record in active_vnrs.items() if record["vnr_id"] == vnr_id), None)
    if embedded_id is None:
        print("⚠ VNR was not active; leave event is ignored")
        return

    v_nodes, _ = load_vnet_from_file(os.path.join(VNET_DATASET_DIR, f"vinre0.5_node_{vnr_id}.yaml"))
    record = active_vnrs.pop(embedded_id)
    multi_simulator.remove_vnr(vnr_id=embedded_id)
    rollback_link_resources(conn, build_changed_edges(record["link_mapping"]), s_edges2)
    rollback_node_resources(conn, record["node_mapping"], v_nodes)
    unregister_active_nodes(active_node_refcounts, record["active_nodes"])
    for item in embedded_vnr_history:
        if item["event_id"] == embedded_id:
            item.update({"status": "finished", "leave_time": event_time})
            break
    print(f"✔ CEVNE VNR {vnr_id} released; sleeping nodes: {sorted(all_substrate_nodes - set(active_node_refcounts))}")


if __name__ == "__main__":
    conn = connect_database()
    with open(PATH_DATABASE_FILE, "rb") as handle:
        G = pickle.load(handle)
    with open(EVENT_FILE, "r", encoding="utf-8") as handle:
        events = sorted((yaml.safe_load(handle) or {}).get("events", []), key=lambda item: item["time"])
    if not events:
        raise RuntimeError("No events found in vnr_events.yaml")

    node_function = Network_node_atrribute(conn, G)
    s_edges2 = load_all_paths_numpy(conn)
    # This snapshot is deliberately taken before any VNR can consume resources.
    total_link_bandwidth = snapshot_total_link_bandwidth(node_function.load_edges_from_db())
    all_substrate_nodes = set(load_substrate_nodes_from_db(conn))
    print(f"✔ Recorded total bandwidth for {len(total_link_bandwidth)} directed links; R={CONGESTION_LIMIT_R}")

    mapping_record_count = 0
    active_vnrs, embedded_vnr_history, total_s_link_result = {}, [], {}
    active_node_refcounts = {}
    multi_simulator = MultiVNRFlowSimulator(total_s_link_result=total_s_link_result, alpha_range=(0.6, 0.9), link_noise=0)
    tm_calculator = TrafficMatrixCalculator(conn)
    domains = [[7, 2, 4, 5, 6], [18, 1, 3, 8, 10], [20, 9, 15, 17, 19], [12, 0, 21, 22, 23], [11, 14, 24, 13, 16]]
    previous_tm = [tm_calculator.compute_domain_temperature(domain)["TM"] for domain in domains]

    time_window, event_index = 100, 0
    arrivals, accepted = 0, 0
    for current_time in range(0, int(events[-1]["time"]) + time_window, time_window):
        window_events = []
        while event_index < len(events) and events[event_index]["time"] < current_time + time_window:
            window_events.append(events[event_index])
            event_index += 1
        for event in window_events:
            if event["type"] == 0:
                handle_leave(event)
        domain_temperatures = []
        for event in window_events:
            if event["type"] != 1:
                continue
            arrivals += 1
            multi_simulator.step_and_write(conn, current_time=current_time + time_window, yaml_path="cevne_vnr_flow_history.yaml")
            domain_temperatures = []
            for index, domain in enumerate(domains):
                value = tm_calculator.compute_domain_temperature(domain, prev_tm=previous_tm[index])
                previous_tm[index] = value["TM"]
                domain_temperatures.append({"domain_id": index, "T": value["T"], "F": value["F"], "H": value["H"]})
            accepted += int(handle_arrival(event))
        multi_simulator.step_and_write(conn, current_time=current_time + time_window, yaml_path="cevne_vnr_flow_history.yaml")
        record_resource_usage_yaml(
            conn, current_time=current_time + time_window, yaml_path="cevne_resource_usage.yaml",
            domain_temperatures=domain_temperatures, acceptance_rate=(accepted / arrivals if arrivals else 0.0),
        )
    print("✔ CEVNE dynamic embedding finished")