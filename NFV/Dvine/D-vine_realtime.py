from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]

import json
import math
import numpy as np
import pyomo.environ as pyo
import os
from NFV.network_flow.Mutiply_VNR import MultiVNRFlowSimulator
from NFV.network_flow.TM_generator import TrafficMatrixCalculator
from NFV.virtual_net_atttribute import Network_node_atrribute
from NFV.util.normal_method import load_vnet_from_file,load_substrate_nodes_from_db,load_all_paths_numpy
from NFV.util.yingshe import build_changed_edges,update_link_resources,rollback_link_resources,update_node_resources,rollback_node_resources,integrate_slink_result
from NFV.util.yaml import record_resource_usage_yaml

# Resolve output relative to this script, independent of the working directory.
DVINE_RESULT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "result_storage", "Dvine"
))


def create_run_output_dir(root=DVINE_RESULT_ROOT):
    os.makedirs(root, exist_ok=True)
    run_numbers = [
        int(name) for name in os.listdir(root)
        if name.isascii() and name.isdigit()
        and os.path.isdir(os.path.join(root, name))
    ]
    next_number = max(run_numbers, default=0) + 1
    while True:
        output_dir = os.path.join(root, str(next_number))
        try:
            os.mkdir(output_dir)
            return output_dir
        except FileExistsError:
            next_number += 1


def build_candidate_nodes_vine(substrate_nodes, virtual_nodes, D):
    """Construct the candidate set Omega(n) defined in the paper."""
    candidate_dict = {}

    for v in virtual_nodes:
        v_id = int(v["id"])
        v_cpu = float(v["cpu"])
        vx, vy = float(v["pos_x"]), float(v["pos_y"])

        candidate_dict[v_id] = []

        for s_id, s in substrate_nodes.items():
            s_cpu = float(s["cpu"])
            sx, sy = float(s["pos"]), float(s["pos2"])

            if s_cpu < v_cpu:
                continue

            dist = math.sqrt((vx - sx) ** 2 + (vy - sy) ** 2)
            if dist > D:
                continue

            candidate_dict[v_id].append(int(s_id))

        if len(candidate_dict[v_id]) == 0:
            print(f"❌ 虚拟节点 {v_id} 没有候选物理节点")
            return False

    return candidate_dict


def build_directed_substrate_edges_for_vine(substrate_edges):
    """Build directed substrate-link capacities for D-ViNE.
        Unlike the original undirected ViNE substrate, (u,v) and (v,u) are independent links with separate capacities.
        Return directed_edges: {(u, v): bw}."""

    directed_edges = {}

    for (u, v), info in substrate_edges.items():
        u = int(u)
        v = int(v)

        if u == v:
            continue

        if isinstance(info, dict):
            bw = info.get("bw_resource", info.get("bw", 0))
        else:
            bw = info

        directed_edges[(u, v)] = float(bw)

    return directed_edges

def build_vine_lp_model(
        substrate_nodes,
        substrate_edges,
        virtual_nodes,
        virtual_edges,
        D=10,
        alpha=1.0,
        beta=1.0,
        epsilon=1e-6,
        one_vn_per_sn=True
):
    """Strict D-ViNE LP relaxation for node mapping with directed substrate links.
        x[v,s] selects meta edges and is relaxed to [0,1].
        f[k,a,b] is virtual link k's flow on augmented edge (a,b).
        Round x times meta-edge flow to obtain an integer node mapping."""

    candidate_dict = build_candidate_nodes_vine(
        substrate_nodes=substrate_nodes,
        virtual_nodes=virtual_nodes,
        D=D
    )

    if candidate_dict is False:
        return None, None

    # Base sets.
    V = [int(v["id"]) for v in virtual_nodes]
    S = [int(s) for s in substrate_nodes.keys()]

    Cv = {int(v["id"]): float(v["cpu"]) for v in virtual_nodes}

    Ev = []
    Bv = {}

    for idx, e in enumerate(virtual_edges):
        u = int(e["source"])
        v = int(e["target"])
        bw = float(e["bw"])
        k = (u, v)
        Ev.append(k)
        Bv[k] = bw

    # Directed substrate graph: retain independent capacities for (i,j) and (j,i).
    Es_directed = build_directed_substrate_edges_for_vine(substrate_edges)

    # Augmented graph nodes.
    meta_nodes = {v: f"meta_{v}" for v in V}
    aug_nodes = list(S) + list(meta_nodes.values())

    # Use directed substrate links directly as the arc set.
    arcs = []

    # substrate arcs
    for (i, j) in Es_directed.keys():
        arcs.append((i, j))

    # meta arcs
    meta_arcs = []

    for v in V:
        m = meta_nodes[v]
        for s in candidate_dict[v]:
            arcs.append((m, s))
            arcs.append((s, m))
            meta_arcs.append((v, s))

    arcs = list(dict.fromkeys(arcs))

    total_virtual_bw = sum(Bv.values())
    big_m = max(total_virtual_bw, 1.0)

    model = pyo.ConcreteModel()

    model.V = pyo.Set(initialize=V)
    model.S = pyo.Set(initialize=S)
    model.Ev = pyo.Set(initialize=Ev, dimen=2)
    model.A = pyo.Set(initialize=arcs, dimen=2)
    model.MetaVS = pyo.Set(initialize=meta_arcs, dimen=2)

    model.x = pyo.Var(model.MetaVS, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.f = pyo.Var(model.Ev, model.A, domain=pyo.NonNegativeReals)

    # Objective and constraints.
    def obj_rule(model):
        substrate_cost = 0.0
        node_cost = 0.0

        for (u, v) in model.Ev:
            bw_req = Bv[(u, v)]

            for (i, j) in model.A:
                # Charge flow costs only on real substrate edges, not meta edges.
                if isinstance(i, int) and isinstance(j, int):
                    if (i, j) in Es_directed:
                        residual_bw = Es_directed[(i, j)]
                        substrate_cost += (
                            alpha
                            * model.f[(u, v), (i, j)]
                            / (residual_bw + epsilon)
                        )

        for (v, s) in model.MetaVS:
            node_cost += (
                beta
                * Cv[v]
                * model.x[v, s]
                / (float(substrate_nodes[s]["cpu"]) + epsilon)
            )

        return substrate_cost + node_cost

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Constraint 1: each virtual node selects exactly one candidate physical node.
    def one_meta_edge_rule(model, v):
        return sum(model.x[v, s] for s in candidate_dict[v]) == 1

    model.one_meta_edge = pyo.Constraint(model.V, rule=one_meta_edge_rule)

    # Constraint 2: physical-node CPU capacity.
    def cpu_capacity_rule(model, s):
        related_vs = [v for v in V if (v, s) in model.MetaVS]

        if len(related_vs) == 0:
            return pyo.Constraint.Feasible

        return (
                sum(Cv[v] * model.x[v, s] for v in related_vs)
                <= float(substrate_nodes[s]["cpu"])
        )

    model.cpu_capacity = pyo.Constraint(model.S, rule=cpu_capacity_rule)

    # Constraint 3: at most one virtual node per physical node within the same VN.
    if one_vn_per_sn:
        def one_vn_per_sn_rule(model, s):
            related_vs = [v for v in V if (v, s) in model.MetaVS]

            if len(related_vs) == 0:
                return pyo.Constraint.Feasible

            return sum(model.x[v, s] for v in related_vs) <= 1

        model.one_vn_per_sn = pyo.Constraint(model.S, rule=one_vn_per_sn_rule)

    # Constraint 4: substrate link capacity.
    # Constrain each direction independently, rather than using f(i,j) + f(j,i).
    def substrate_bw_capacity_rule(model, i, j):
        if (i, j) not in Es_directed:
            return pyo.Constraint.Skip

        return (
            sum(
                model.f[(u, v), (i, j)]
                for (u, v) in model.Ev
                if (i, j) in model.A
            )
            <= Es_directed[(i, j)]
        )

    model.substrate_bw_capacity = pyo.Constraint(
        [(i, j) for (i, j) in Es_directed.keys()],
        rule=substrate_bw_capacity_rule
    )

    # Constraint 5: couple meta edges with x.
    # A meta edge cannot carry flow when its x[v,s] approaches zero.
    def meta_flow_coupling_rule(model, vv, ss):
        m = meta_nodes[vv]

        related_flow = 0.0

        for (u, v) in model.Ev:
            if (m, ss) in model.A:
                related_flow += model.f[(u, v), (m, ss)]
            if (ss, m) in model.A:
                related_flow += model.f[(u, v), (ss, m)]

        return related_flow <= big_m * model.x[vv, ss]

    model.meta_flow_coupling = pyo.Constraint(
        model.MetaVS,
        rule=meta_flow_coupling_rule
    )

    # Constraint 6: flow conservation.
    def flow_conservation_rule(model, u, v, n):
        src = meta_nodes[u]
        dst = meta_nodes[v]
        demand = Bv[(u, v)]

        out_flow = sum(
            model.f[(u, v), (n, j)]
            for (a, j) in model.A
            if a == n
        )

        in_flow = sum(
            model.f[(u, v), (i, n)]
            for (i, b) in model.A
            if b == n
        )

        if n == src:
            return out_flow - in_flow == demand
        elif n == dst:
            return out_flow - in_flow == -demand
        else:
            return out_flow - in_flow == 0

    model.flow_conservation = pyo.Constraint(
        model.Ev,
        aug_nodes,
        rule=flow_conservation_rule
    )

    extra = {
        "candidate_dict": candidate_dict,
        "meta_nodes": meta_nodes,
        "Bv": Bv,
        "Es_directed": Es_directed,
        "aug_nodes": aug_nodes
    }

    return model, extra


def solve_vine_lp_node_mapping(
        substrate_nodes,
        substrate_edges,
        virtual_nodes,
        virtual_edges,
        D=10,
        solver_name="glpk",
        deterministic=True,
        one_vn_per_sn=True,
        seed=None
):
    """Strict D-ViNE/R-ViNE node mapping.
        deterministic=True selects D-ViNE; False selects R-ViNE."""

    if seed is not None:
        np.random.seed(seed)

    model, extra = build_vine_lp_model(
        substrate_nodes=substrate_nodes,
        substrate_edges=substrate_edges,
        virtual_nodes=virtual_nodes,
        virtual_edges=virtual_edges,
        D=D,
        one_vn_per_sn=one_vn_per_sn
    )

    if model is None:
        return False

    solver = pyo.SolverFactory(solver_name)
    result = solver.solve(model, tee=False)

    status = result.solver.status
    termination = result.solver.termination_condition

    if status != pyo.SolverStatus.ok:
        print("❌ ViNE LP 求解失败:", status)
        return False

    if termination not in [
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible
    ]:
        print("❌ ViNE LP 无可行解:", termination)
        return False

    candidate_dict = extra["candidate_dict"]
    meta_nodes = extra["meta_nodes"]

    node_mapping = {}
    used_substrate_nodes = set()

    # ---------- Rounding ----------
    for v in model.V:

        scores = {}

        for s in candidate_dict[v]:

            if one_vn_per_sn and s in used_substrate_nodes:
                continue

            x_val = pyo.value(model.x[v, s])
            m = meta_nodes[v]

            total_meta_flow = 0.0

            for (u, w) in model.Ev:
                if (m, s) in model.A:
                    total_meta_flow += pyo.value(model.f[(u, w), (m, s)])
                if (s, m) in model.A:
                    total_meta_flow += pyo.value(model.f[(u, w), (s, m)])

            # Core idea of the original D-ViNE formulation.
            # score = x_value × total flow through meta edge
            scores[s] = x_val * total_meta_flow

        if not scores:
            print(f"❌ 虚拟节点 {v} rounding 阶段没有可用物理节点")
            return False

        # D-ViNE
        if deterministic:
            selected_s = max(scores, key=scores.get)

        # R-ViNE
        else:
            keys = list(scores.keys())
            vals = np.array([max(scores[k], 0.0) for k in keys], dtype=float)

            if vals.sum() <= 0:
                probs = np.ones(len(keys)) / len(keys)
            else:
                probs = vals / vals.sum()

            selected_s = int(np.random.choice(keys, p=probs))

        node_mapping[int(v)] = int(selected_s)
        used_substrate_nodes.add(int(selected_s))

    print("✔ ViNE LP + rounding 节点映射完成:", node_mapping)

    return node_mapping

def append_one_record_to_yaml(
        s_link_result_temp,
        vnr_id,
        event_time,
        event_id,

        file_path="vne_mapping.yaml"

    ):
    global count

    # 1. Auto-increment the ID.
    count += 1
    print(f"当前ID: {count}")

    # 2. Expand automatically.
    if isinstance(s_link_result_temp, list) and len(s_link_result_temp) > 0:
        if isinstance(s_link_result_temp[0], list):
            s_link_result_temp = s_link_result_temp[0]

    # 3. Read existing data.
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = yaml.safe_load(f) or []
            except:
                data = []
    else:
        data = []

    # 4. Build the current VNR record.
    links = []

    for item in s_link_result_temp:
        links.append({
            "source": item["source"],
            "target": item["target"],
            "bw": float(item["bw"]),
            "linkpath": item["linkpath"]
        })

    new_record = {
        "id": count,
        "vnr_ind": vnr_id,
        "event_id":event_id,
        "event_time": float(event_time),
        "links": links,

    }

    data.append(new_record)

    # 5. Write back.
    with open(file_path, "w") as f:
        yaml.dump(
            data,
            f,
            sort_keys=False,
            default_flow_style=False
        )

    print(f"✅ 已追加 VNR {vnr_id}，event_time={event_time}，包含 {len(links)} 条链路")

def convert_virtual_links_to_substrate_links(node_mapping, virtual_edges):
    """Convert virtual links to physical-endpoint requests using the node mapping.
        Return [{'source': s_src, 'target': s_dst, 'bw': bw}, ...]."""
    slink_list = []

    for link in virtual_edges:
        v_src = int(link["source"])
        v_dst = int(link["target"])
        bw = float(link["bw"])

        if v_src not in node_mapping or v_dst not in node_mapping:
            print(f"❌ 虚拟节点 {v_src} 或 {v_dst} 不在 node_mapping 中")
            return False

        s_src = node_mapping[v_src]
        s_dst = node_mapping[v_dst]

        slink_list.append({
            "source": s_src,
            "target": s_dst,
            "bw": bw
        })

    return slink_list


def map_slinks_to_shortest_paths_numpy(slink_list, substrate_edges, substrate_edges2):
    """Map links using shortest feasible paths; follow map_slinks_to_paths_numpy_test in EO_test_two_stage_link.py.
        substrate_edges = s_edges stores residual single-hop bandwidth {(u,v): bw}.
        substrate_edges2 = s_edges2 stores path candidates {(src,dst): {'hop': ..., 'bw': ..., 'path': ...}}."""

    results = []

    # Use a temporary bandwidth copy to prevent multiple links in one VNR from overallocating capacity.
    temp_edge_bw = substrate_edges.copy()

    # Prefer mapping high-bandwidth virtual links first.
    slink_list = sorted(slink_list, key=lambda x: x["bw"], reverse=True)

    for sl in slink_list:
        src = int(sl["source"])
        dst = int(sl["target"])
        bw_required = float(sl["bw"])
        key = (src, dst)

        if key not in substrate_edges2:
            print(f"❌ 未找到物理端点路径 {src} → {dst}")
            return False

        hop_arr = substrate_edges2[key]["hop"]
        path_list = substrate_edges2[key]["path"]

        chosen_idx = None
        chosen_bottleneck = None

        # path_db is sorted by hop count, so the first feasible path is the shortest feasible one.
        for i, path in enumerate(path_list):

            path_bottleneck = float("inf")
            feasible = True

            for u, v in path:
                edge_key = (int(u), int(v))
                remain_bw = float(temp_edge_bw.get(edge_key, 0))

                if remain_bw < bw_required:
                    feasible = False
                    break

                path_bottleneck = min(path_bottleneck, remain_bw)

            if feasible:
                chosen_idx = i
                chosen_bottleneck = path_bottleneck
                break

        if chosen_idx is None:
            print(f"❌ 带宽不足，无法映射 {src} → {dst}，需要 bw={bw_required}")
            return False

        chosen_path = path_list[chosen_idx]

        # Deduct temporary link bandwidth.
        for u, v in chosen_path:
            temp_edge_bw[(int(u), int(v))] -= bw_required

        results.append({
            "source": src,
            "target": dst,
            "bw": bw_required,
            "hop": int(hop_arr[chosen_idx]),
            "path_bottleneck": float(chosen_bottleneck),
            "linkpath": chosen_path
        })

    return results

def handle_arrival(event):
    vnr_id = event["v_net_id"]
    event_time = event["time"]
    event_id = event["id"]

    print(f"\n🟢 Arrival VNR {vnr_id} at t={event_time:.2f}")

    vnet_path = (
        str(_PROJECT_ROOT / f'NFV/vnet_dataset/vinre0.5_node_{vnr_id}.yaml')
    )

    v_nodes, v_edges = load_vnet_from_file(vnet_path)

    s_nodes = load_substrate_nodes_from_db(conn)
    s_edges = node_function.load_edges_from_db()

    node_mapping = solve_vine_lp_node_mapping(
        substrate_nodes=s_nodes,
        substrate_edges=s_edges,
        virtual_nodes=v_nodes,
        virtual_edges=v_edges,
        D=10,
        solver_name="glpk",
        deterministic=True,
        one_vn_per_sn=True,
        seed=42
    )

    if node_mapping is False:
        print("❌ D-ViNE 节点映射失败，该 VNR 被拒绝")
        return

    slink_list = convert_virtual_links_to_substrate_links(
        node_mapping=node_mapping,
        virtual_edges=v_edges
    )

    if slink_list is False:
        print("❌ 物理端点链路请求生成失败，该 VNR 被拒绝")
        return

    s_link_result = map_slinks_to_shortest_paths_numpy(
        slink_list=slink_list,
        substrate_edges=s_edges,
        substrate_edges2=s_edges2
    )

    if s_link_result is False:
        print("❌ D-ViNE 链路映射失败，该 VNR 被拒绝")
        return

    changed_edges = build_changed_edges(s_link_result)

    update_link_resources(conn, changed_edges, s_edges2)
    update_node_resources(conn, node_mapping, v_nodes)

    vnr_event_id = integrate_slink_result(
        total_s_link_result,
        s_link_result
    )

    active_vnrs[vnr_event_id] = {
        "vnr_id": vnr_id,
        "best_list": node_mapping,
        "s_link_result": s_link_result
    }

    embedded_vnr_history.append({
        "event_id": vnr_event_id,
        "vnr_id": vnr_id,
        "arrival_time": event_time,
        "node_mapping": node_mapping,
        "link_mapping": s_link_result,
        "changed_edges": changed_edges,
        "status": "active"
    })

    # Register with the traffic simulator.
    multi_simulator.add_vnr(
        vnr_id=vnr_event_id,
        s_link_result=s_link_result,
        vnr_id2=vnr_id
    )


    append_one_record_to_yaml(
        s_link_result_temp=s_link_result,
        vnr_id=vnr_id,
        event_time=event_time,
        event_id=event_id,
        file_path=mapping_output_path
    )

    print(f"✔ D-ViNE VNR {vnr_id} 成功嵌入")
    print("arrival changed_edges =", changed_edges)
    print("arrival node_mapping =", node_mapping)


def handle_leave(event):
    vnr_id = event["v_net_id"]
    event_time = event["time"]

    print(f"\n🔴 Leave VNR {vnr_id} at t={event_time:.2f}")

    vnet_path = (
        str(_PROJECT_ROOT / f'NFV/vnet_dataset/vinre0.5_node_{vnr_id}.yaml')
    )

    v_nodes, v_edges = load_vnet_from_file(vnet_path)

    vnr_event_id = None

    for eid, record in active_vnrs.items():
        if record["vnr_id"] == vnr_id:
            vnr_event_id = eid
            break

    if vnr_event_id is None:
        print("⚠ VNR 未在服务中，忽略该 leave 事件")
        return

    record = active_vnrs.pop(vnr_event_id)
    multi_simulator.remove_vnr(
        vnr_id=vnr_event_id
    )
    node_mapping = record["best_list"]
    s_link_result = record["s_link_result"]



    changed_edges = build_changed_edges(s_link_result)

    rollback_link_resources(conn, changed_edges, s_edges2)
    rollback_node_resources(conn, node_mapping, v_nodes)

    for item in embedded_vnr_history:
        if item["event_id"] == vnr_event_id:
            item["status"] = "finished"
            item["leave_time"] = event_time
            break

    print(f"✔ D-ViNE VNR {vnr_id}（event_id={vnr_event_id}）资源已释放")
    print("leave changed_edges =", changed_edges)
    print("leave node_mapping =", node_mapping)


if __name__ == "__main__":

    import pickle
    import yaml
    import pymysql
    from NFV.db_config import connect_database

    run_output_dir = create_run_output_dir()
    mapping_output_path = os.path.join(run_output_dir, "vne_mapping.yaml")
    flow_output_path = os.path.join(run_output_dir, "vnr_flow_history.yaml")
    resource_output_path = os.path.join(run_output_dir, "resource_usage.yaml")
    # Keep all three outputs present even when no mapping succeeds.
    for output_path, initial_data in (
        (mapping_output_path, []),
        (flow_output_path, {"records": []}),
        (resource_output_path, {"records": []}),
    ):
        with open(output_path, "x", encoding="utf-8") as output_file:
            yaml.safe_dump(initial_data, output_file, sort_keys=False)
    print(f"Dvine results: {run_output_dir}")

    conn = connect_database()

    print("✔ 数据库连接成功")

    count = 0

    with open(str(_PROJECT_ROOT / 'NFV/p_net_directed.pkl'), "rb") as f:
        G = pickle.load(f)

    print("✔ 底层网络读取完成")

    with open("vnr_events.yaml", "r", encoding="utf-8") as f:
        event_data = yaml.safe_load(f)

    events = sorted(event_data["events"], key=lambda e: e["time"])

    node_function = Network_node_atrribute(conn, G)
    s_edges2 = load_all_paths_numpy(conn)

    print(f"✔ 物理路径库读取完成，共 {len(s_edges2)} 对端点路径")

    TIME_WINDOW = 100
    current_event_id = 0

    active_vnrs = {}
    embedded_vnr_history = []
    total_s_link_result = {}
    # Simulate VNR traffic generation.
    multi_simulator = MultiVNRFlowSimulator(
        total_s_link_result=total_s_link_result,
        alpha_range=(0.6, 0.9),
        link_noise=0
    )
    # Initialize the network temperature calculation matrix.
    tm_calculator = TrafficMatrixCalculator(conn)
    # Temporarily define temperature domains manually for testing.
    domains = [
        [7, 2, 4, 5, 6],
        [18, 1, 3, 8, 10],
        [20, 9, 15, 17, 19],
        [12, 0, 21, 22, 23],
        [11, 14, 24, 13, 16]
    ]
    prev_tms = [None] * len(domains)
    # Initialize temperature calculation.
    for i, domain_nodes in enumerate(domains):
        result_0 = tm_calculator.compute_domain_temperature(domain_nodes)
        prev_tms[i] = result_0["TM"]
        print(f"[Init] Domain {i}: F={result_0['F']:.2f}, "
              f"H={result_0['H']:.4f}, T={result_0['T']}")


    # Initialize related parameters.
    temperatures = []
    total_arrival_count = 0
    total_accept_count = 0
    for current_time in range(
            0,
            int(events[-1]["time"]) + TIME_WINDOW,
            TIME_WINDOW
    ):
        print(f"\n=== Time Window [{current_time}, {current_time + TIME_WINDOW}) ===")

        window_events = []

        while (
                current_event_id < len(events)
                and events[current_event_id]["time"] < current_time + TIME_WINDOW
        ):
            window_events.append(events[current_event_id])
            current_event_id += 1

        # Process departure events first.
        for event in window_events:
            if event["type"] == 0:
                handle_leave(event)

        # Then process arrival events.
        for event in window_events:
            if event["type"] == 1:
                # Record the acceptance rate.
                total_arrival_count += 1
                before_accept_num = len(embedded_vnr_history)
                # Also simulate before arrivals; otherwise temperature changes are not captured beforehand.
                multi_simulator.step_and_write(
                    conn,
                    current_time=current_time + TIME_WINDOW,
                    yaml_path=flow_output_path
                )
                domain_temperatures = []

                for i, domain_nodes in enumerate(domains):
                    result = tm_calculator.compute_domain_temperature(
                        domain_nodes,
                        prev_tm=prev_tms[i]
                    )

                    prev_tms[i] = result["TM"]

                    domain_temperatures.append({
                        "domain_id": i,
                        "T": result["T"],
                        "F": result["F"],
                        "H": result["H"]
                    })
                handle_arrival(event)
                # Record the acceptance rate.
                after_accept_num = len(embedded_vnr_history)
                if after_accept_num > before_accept_num:
                    total_accept_count += 1

        multi_simulator.step_and_write(
            conn,
            current_time=current_time + TIME_WINDOW,
            yaml_path=flow_output_path
        )
        # Record the acceptance rate.
        if total_arrival_count > 0:
            total_acceptance_rate = total_accept_count / total_arrival_count
        else:
            total_acceptance_rate = 0.0
        record_resource_usage_yaml(
            conn,
            current_time=current_time + TIME_WINDOW,
            yaml_path=resource_output_path,
            domain_temperatures=domain_temperatures,
            acceptance_rate=total_acceptance_rate,
        )

        print("当前在线 VNR 数量：", len(active_vnrs))
        print("成功嵌入数量：", len(embedded_vnr_history))

    print("\n✔ D-ViNE 静态映射事件处理完成")