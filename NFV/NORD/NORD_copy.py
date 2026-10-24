

from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]

import os
import math
import pickle
import yaml
import pymysql
from NFV.db_config import connect_database
import numpy as np

from NFV.NORD.NORD_function import NORDRanker
from NFV.virtual_net_atttribute import Network_node_atrribute
from NFV.network_flow.Mutiply_VNR import MultiVNRFlowSimulator
from NFV.network_flow.TM_generator import TrafficMatrixCalculator
from NFV.util.normal_method import (
    load_vnet_from_file,
    load_substrate_nodes_from_db,
    load_all_paths_numpy
)
from NFV.util.yingshe import (
    build_changed_edges,
    update_link_resources,
    rollback_link_resources,
    update_node_resources,
    rollback_node_resources,
    integrate_slink_result
)
from NFV.util.yaml import record_resource_usage_yaml


# =========================================================
# 0. Basic configuration.
# =========================================================

P_NET_PATH = str(_PROJECT_ROOT / 'NFV/p_net_directed.pkl')
VNET_DATASET_DIR = str(_PROJECT_ROOT / 'NFV/vnet_dataset')
VNR_EVENTS_PATH = "vnr_events.yaml"

NORD_MAPPING_YAML = "nord_vne_mapping.yaml"
NORD_FLOW_YAML = "NORD_flow_history.yaml"
NORD_RESOURCE_YAML = "NORD_resource_usage.yaml"

CRITERIA = [
    "crb_strength",
    "bw_strength",
    "degree",
    "betweenness",
    "eigen"
]

EPS = 1e-12


# =========================================================
# 1. Entropy-weighted TOPSIS ranking.
# =========================================================

def _matrix_from_attr_dict(attr_dict, criteria=CRITERIA):
    """Convert NORDRanker's attribute dictionary into alternatives and a decision matrix.
        Expected structure: {node_id: {'crb_strength': ..., 'bw_strength': ..., 'degree': ..., 'betweenness': ..., 'eigen': ...}}."""
    alternatives = [int(k) for k in attr_dict.keys()]
    alternatives = sorted(alternatives)

    matrix = []
    for node_id in alternatives:
        attrs = attr_dict[node_id]
        row = [float(attrs.get(c, 0.0)) for c in criteria]
        matrix.append(row)

    return alternatives, np.asarray(matrix, dtype=float)


def entropy_weight(decision_matrix):
    """Algorithm 1 in the NORD paper: criteria-weight computation.
        Treat every attribute as a benefit criterion."""
    D = np.asarray(decision_matrix, dtype=float).copy()

    if D.ndim != 2 or D.shape[0] == 0 or D.shape[1] == 0:
        raise ValueError("decision_matrix must be a non-empty 2D matrix")

    m, n = D.shape
    if m == 1:
        return np.ones(n, dtype=float) / n

    # Guard against negative inputs to entropy weighting; centrality and resource attributes are normally nonnegative.
    min_per_col = D.min(axis=0)
    for j in range(n):
        if min_per_col[j] < 0:
            D[:, j] = D[:, j] - min_per_col[j]

    col_sum = D.sum(axis=0)
    P = np.divide(D, col_sum + EPS)

    # Treat 0 * log(0) as zero.
    logP = np.zeros_like(P)
    positive_mask = P > 0
    logP[positive_mask] = np.log(P[positive_mask])

    entropy = -(1.0 / math.log(m)) * np.sum(P * logP, axis=0)
    diversity = 1.0 - entropy

    if np.sum(diversity) <= EPS:
        return np.ones(n, dtype=float) / n

    weights = diversity / (np.sum(diversity) + EPS)
    return weights


def topsis_rank(decision_matrix, alternatives, weights=None):
    """Algorithm 2 in the NORD paper: TOPSIS rank determination.
        Return rank_score; higher scores indicate higher ranks."""
    D = np.asarray(decision_matrix, dtype=float).copy()

    if D.ndim != 2 or D.shape[0] == 0:
        return {}

    if weights is None:
        weights = entropy_weight(D)
    weights = np.asarray(weights, dtype=float)

    # Vector normalization.
    norm = np.sqrt(np.sum(D ** 2, axis=0))
    R = np.divide(D, norm + EPS)

    # Weighted normalized matrix.
    V = R * weights

    # All attributes are benefit criteria: ideal best=max, ideal worst=min.
    ideal_best = np.max(V, axis=0)
    ideal_worst = np.min(V, axis=0)

    dist_best = np.sqrt(np.sum((V - ideal_best) ** 2, axis=1))
    dist_worst = np.sqrt(np.sum((V - ideal_worst) ** 2, axis=1))

    closeness = dist_worst / (dist_best + dist_worst + EPS)

    return {
        int(node_id): float(score)
        for node_id, score in zip(alternatives, closeness)
    }


def sort_rank_desc(rank_score):
    return sorted(rank_score.keys(), key=lambda node_id: rank_score[node_id], reverse=True)


# =========================================================
# 2. Candidate nodes and greedy node mapping.
# =========================================================

def build_nord_candidate_nodes(substrate_nodes, virtual_nodes, D=10):
    """Candidate-node constraints: sufficient CPU and geographical distance at most D.
        Set D=None to disable the geographical constraint."""
    candidate_dict = {}

    for v in virtual_nodes:
        v_id = int(v["id"])
        v_cpu = float(v["cpu"])
        vx = float(v.get("pos_x", 0.0))
        vy = float(v.get("pos_y", 0.0))

        candidate_dict[v_id] = []

        for s_id, s in substrate_nodes.items():
            s_id = int(s_id)
            s_cpu = float(s["cpu"])

            if s_cpu < v_cpu:
                continue

            if D is not None:
                sx = float(s.get("pos", 0.0))
                sy = float(s.get("pos2", 0.0))
                dist = math.sqrt((vx - sx) ** 2 + (vy - sy) ** 2)
                if dist > D:
                    continue

            candidate_dict[v_id].append(s_id)

        if len(candidate_dict[v_id]) == 0:
            print(f"❌ NORD: 虚拟节点 {v_id} 没有满足 CPU/位置约束的候选物理节点")
            return False

    return candidate_dict


def nord_greedy_node_mapping(
        virtual_rank,
        substrate_rank,
        candidate_dict,
        virtual_nodes,
        substrate_nodes,
        one_vn_per_sn=True
):
    """NORD Allocator VM embedding:
        Process higher-TOPSIS-score VMs first.
        Choose the highest-ranked feasible candidate server.
        By default, two VMs in the same VNR cannot share a substrate node."""
    v_cpu_dict = {int(v["id"]): float(v["cpu"]) for v in virtual_nodes}
    used_substrate_nodes = set()
    node_mapping = {}

    for v_id in virtual_rank:
        if v_id not in candidate_dict:
            continue

        selected_s = None
        for s_id in substrate_rank:
            if s_id not in candidate_dict[v_id]:
                continue

            if one_vn_per_sn and s_id in used_substrate_nodes:
                continue

            s_cpu = float(substrate_nodes[s_id]["cpu"])
            if s_cpu < v_cpu_dict[v_id]:
                continue

            selected_s = s_id
            break

        if selected_s is None:
            print(f"❌ NORD: 虚拟节点 {v_id} 贪心映射失败")
            return False

        node_mapping[v_id] = selected_s
        used_substrate_nodes.add(selected_s)

    if len(node_mapping) != len(virtual_nodes):
        print("❌ NORD: 节点映射不完整")
        return False

    return node_mapping


# =========================================================
# 3. Link mapping and evaluation.
# =========================================================

def convert_virtual_links_to_substrate_links(node_mapping, virtual_edges):
    slink_list = []

    for link in virtual_edges:
        v_src = int(link["source"])
        v_dst = int(link["target"])
        bw = float(link["bw"])

        if v_src not in node_mapping or v_dst not in node_mapping:
            print(f"❌ NORD: 虚拟节点 {v_src} 或 {v_dst} 不在 node_mapping 中")
            return False

        s_src = int(node_mapping[v_src])
        s_dst = int(node_mapping[v_dst])

        slink_list.append({
            "source": s_src,
            "target": s_dst,
            "bw": bw,
            "virtual_source": v_src,
            "virtual_target": v_dst
        })

    return slink_list


def map_slinks_to_shortest_paths_numpy(slink_list, substrate_edges, substrate_edges2):
    """NORD VL embedding uses shortest feasible paths.
        path_db is sorted by hop count, so the first bandwidth-feasible candidate is the shortest feasible path."""
    results = []
    temp_edge_bw = substrate_edges.copy()

    # Map high-bandwidth links first to reduce later bottleneck failures.
    slink_list = sorted(slink_list, key=lambda x: x["bw"], reverse=True)

    for sl in slink_list:
        src = int(sl["source"])
        dst = int(sl["target"])
        bw_required = float(sl["bw"])
        key = (src, dst)

        if key not in substrate_edges2:
            print(f"❌ NORD: 未找到物理端点路径 {src} → {dst}")
            return False

        hop_arr = substrate_edges2[key]["hop"]
        path_list = substrate_edges2[key]["path"]

        chosen_idx = None
        chosen_bottleneck = None

        for i, path in enumerate(path_list):
            path_bottleneck = float("inf")
            feasible = True

            for u, v in path:
                edge_key = (int(u), int(v))
                remain_bw = float(temp_edge_bw.get(edge_key, 0.0))

                if remain_bw < bw_required:
                    feasible = False
                    break

                path_bottleneck = min(path_bottleneck, remain_bw)

            if feasible:
                chosen_idx = i
                chosen_bottleneck = path_bottleneck
                break

        if chosen_idx is None:
            print(f"❌ NORD: 带宽不足，无法映射 {src} → {dst}，需要 bw={bw_required}")
            return False

        chosen_path = path_list[chosen_idx]

        for u, v in chosen_path:
            temp_edge_bw[(int(u), int(v))] -= bw_required

        results.append({
            "source": src,
            "target": dst,
            "bw": bw_required,
            "hop": int(hop_arr[chosen_idx]),
            "path_bottleneck": float(chosen_bottleneck),
            "linkpath": chosen_path,
            "virtual_source": int(sl["virtual_source"]),
            "virtual_target": int(sl["virtual_target"])
        })

    return results


def calculate_nord_revenue_cost_ratio(virtual_nodes, virtual_edges, s_link_result, epsilon=1e-6):
    """NORD revenue and cost:
        Revenue = sum(VM CPU demand) + sum(VL BW demand).
        Cost = sum(VM CPU demand) + sum(VL BW demand * mapped hop count)."""
    node_revenue = sum(float(v["cpu"]) for v in virtual_nodes)
    link_revenue = sum(float(e["bw"]) for e in virtual_edges)

    revenue = node_revenue + link_revenue

    link_cost = 0.0
    total_hop = 0.0
    for item in s_link_result:
        bw = float(item["bw"])
        hop = float(len(item["linkpath"]))
        link_cost += bw * hop
        total_hop += hop

    cost = node_revenue + link_cost
    revenue_cost_ratio = revenue / (cost + epsilon)

    return revenue_cost_ratio, revenue, cost, total_hop


# =========================================================
# 4. Main NORD mapping procedure.
# =========================================================

def nord_node_link_mapping(
        conn,
        G,
        substrate_nodes,
        substrate_edges,
        substrate_edges2,
        virtual_nodes,
        virtual_edges,
        D=10,
        one_vn_per_sn=True
):
    """Complete NORD mapping for one VNR:
        Rank Generator -> Greedy VM embedding -> Shortest-path VL embedding."""
    nord_ranker = NORDRanker(conn, G)

    # Construct the decision matrix for five attributes.
    virtual_attr_matrix = nord_ranker.build_virtual_decision_matrix(
        virtual_nodes=virtual_nodes,
        virtual_edges=virtual_edges
    )
    substrate_attr_matrix = nord_ranker.build_substrate_decision_matrix()

    v_alternatives, v_matrix = _matrix_from_attr_dict(virtual_attr_matrix)
    s_alternatives, s_matrix = _matrix_from_attr_dict(substrate_attr_matrix)

    # Entropy weights and TOPSIS scores.
    virtual_weights = entropy_weight(v_matrix)
    substrate_weights = entropy_weight(s_matrix)

    virtual_score = topsis_rank(v_matrix, v_alternatives, virtual_weights)
    substrate_score = topsis_rank(s_matrix, s_alternatives, substrate_weights)

    virtual_rank = sort_rank_desc(virtual_score)
    substrate_rank = sort_rank_desc(substrate_score)

    print("[NORD] virtual_weights =", dict(zip(CRITERIA, np.round(virtual_weights, 4))))
    print("[NORD] substrate_weights =", dict(zip(CRITERIA, np.round(substrate_weights, 4))))
    print("[NORD] virtual_rank =", virtual_rank)
    print("[NORD] substrate_rank[:10] =", substrate_rank[:10])

    # Candidate nodes and greedy node mapping.
    candidate_dict = build_nord_candidate_nodes(
        substrate_nodes=substrate_nodes,
        virtual_nodes=virtual_nodes,
        D=D
    )
    if candidate_dict is False:
        return False

    node_mapping = nord_greedy_node_mapping(
        virtual_rank=virtual_rank,
        substrate_rank=substrate_rank,
        candidate_dict=candidate_dict,
        virtual_nodes=virtual_nodes,
        substrate_nodes=substrate_nodes,
        one_vn_per_sn=one_vn_per_sn
    )
    if node_mapping is False:
        return False

    # Link mapping.
    slink_list = convert_virtual_links_to_substrate_links(node_mapping, virtual_edges)
    if slink_list is False:
        return False

    s_link_result = map_slinks_to_shortest_paths_numpy(
        slink_list=slink_list,
        substrate_edges=substrate_edges,
        substrate_edges2=substrate_edges2
    )
    if s_link_result is False:
        return False

    ratio, revenue, cost, hop = calculate_nord_revenue_cost_ratio(
        virtual_nodes=virtual_nodes,
        virtual_edges=virtual_edges,
        s_link_result=s_link_result
    )

    return {
        "node_mapping": node_mapping,
        "s_link_result": s_link_result,
        "revenue_cost_ratio": ratio,
        "revenue": revenue,
        "cost": cost,
        "hop": hop,
        "virtual_score": virtual_score,
        "substrate_score": substrate_score,
        "virtual_weights": virtual_weights.tolist(),
        "substrate_weights": substrate_weights.tolist()
    }


# =========================================================
# 5. Record YAML results.
# =========================================================

def append_one_record_to_yaml(
        s_link_result_temp,
        vnr_id,
        event_time,
        event_id,
        record_id,
        file_path=NORD_MAPPING_YAML
):
    if isinstance(s_link_result_temp, list) and len(s_link_result_temp) > 0:
        if isinstance(s_link_result_temp[0], list):
            s_link_result_temp = s_link_result_temp[0]

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f) or []
            except Exception:
                data = []
    else:
        data = []

    links = []
    for item in s_link_result_temp:
        links.append({
            "source": int(item["source"]),
            "target": int(item["target"]),
            "bw": float(item["bw"]),
            "linkpath": item["linkpath"],
            "virtual_source": int(item.get("virtual_source", -1)),
            "virtual_target": int(item.get("virtual_target", -1))
        })

    new_record = {
        "id": int(record_id),
        "vnr_ind": int(vnr_id),
        "event_id": int(event_id),
        "event_time": float(event_time),
        "links": links
    }

    data.append(new_record)

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True
        )

    print(f"✅ 已追加 NORD VNR {vnr_id}，event_time={event_time}，包含 {len(links)} 条链路")


# =========================================================
# 6. Process arrival and departure events.
# =========================================================

def handle_arrival(event):
    global count
    global conn
    global G
    global node_function
    global s_edges2
    global active_vnrs
    global embedded_vnr_history
    global total_s_link_result
    global multi_simulator

    vnr_id = event["v_net_id"]
    event_time = event["time"]
    event_id = event["id"]

    print(f"\n🟢 [NORD] Arrival VNR {vnr_id} at t={event_time:.2f}")

    vnet_path = os.path.join(VNET_DATASET_DIR, f"vinre0.5_node_{vnr_id}.yaml")
    v_nodes, v_edges = load_vnet_from_file(vnet_path)

    s_nodes = load_substrate_nodes_from_db(conn)
    s_edges = node_function.load_edges_from_db()

    solution = nord_node_link_mapping(
        conn=conn,
        G=G,
        substrate_nodes=s_nodes,
        substrate_edges=s_edges,
        substrate_edges2=s_edges2,
        virtual_nodes=v_nodes,
        virtual_edges=v_edges,
        D=10,
        one_vn_per_sn=True
    )

    if solution is False:
        print("❌ NORD 节点-链路映射失败，该 VNR 被拒绝")
        return

    node_mapping = solution["node_mapping"]
    s_link_result = solution["s_link_result"]

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
        "s_link_result": s_link_result,
        "changed_edges": changed_edges,
        "revenue_cost_ratio": solution["revenue_cost_ratio"],
        "revenue": solution["revenue"],
        "cost": solution["cost"],
        "hop": solution["hop"],
        "status": "active"
    })

    multi_simulator.add_vnr(
        vnr_id=vnr_event_id,
        s_link_result=s_link_result,
        vnr_id2=vnr_id
    )

    count += 1

    append_one_record_to_yaml(
        s_link_result_temp=s_link_result,
        vnr_id=vnr_id,
        event_time=event_time,
        event_id=event_id,
        record_id=count,
        file_path=NORD_MAPPING_YAML
    )

    print(f"✔ NORD VNR {vnr_id} 成功嵌入")
    print("revenue_cost_ratio =", solution["revenue_cost_ratio"])
    print("revenue =", solution["revenue"])
    print("cost =", solution["cost"])
    print("hop =", solution["hop"])
    print("arrival changed_edges =", changed_edges)
    print("arrival node_mapping =", node_mapping)


def handle_leave(event):
    global conn
    global s_edges2
    global active_vnrs
    global embedded_vnr_history
    global multi_simulator

    vnr_id = event["v_net_id"]
    event_time = event["time"]

    print(f"\n🔴 [NORD] Leave VNR {vnr_id} at t={event_time:.2f}")

    vnet_path = os.path.join(VNET_DATASET_DIR, f"vinre0.5_node_{vnr_id}.yaml")
    v_nodes, _ = load_vnet_from_file(vnet_path)

    vnr_event_id = None
    for eid, record in active_vnrs.items():
        if record["vnr_id"] == vnr_id:
            vnr_event_id = eid
            break

    if vnr_event_id is None:
        print("⚠ NORD: VNR 未在服务中，忽略该 leave 事件")
        return

    record = active_vnrs.pop(vnr_event_id)

    multi_simulator.remove_vnr(vnr_id=vnr_event_id)

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

    print(f"✔ NORD VNR {vnr_id}（event_id={vnr_event_id}）资源已释放")
    print("leave changed_edges =", changed_edges)
    print("leave node_mapping =", node_mapping)


# =========================================================
# 7. Main function: automated event deployment.
# =========================================================

if __name__ == "__main__":
    conn = connect_database()
    print("✔ 数据库连接成功")

    count = 0

    with open(P_NET_PATH, "rb") as f:
        G = pickle.load(f)
    print("✔ 底层网络读取完成")

    with open(VNR_EVENTS_PATH, "r", encoding="utf-8") as f:
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

    multi_simulator = MultiVNRFlowSimulator(
        total_s_link_result=total_s_link_result,
        alpha_range=(0.6, 0.9),
        link_noise=0
    )

    tm_calculator = TrafficMatrixCalculator(conn)

    # Reuse the five fixed temperature domains from GA_copy.py; replace domains for automatic partitioning.
    domains = [
        [7, 2, 4, 5, 6],
        [18, 1, 3, 8, 10],
        [20, 9, 15, 17, 19],
        [12, 0, 21, 22, 23],
        [11, 14, 24, 13, 16]
    ]

    prev_tms = [None] * len(domains)
    for i, domain_nodes in enumerate(domains):
        result_0 = tm_calculator.compute_domain_temperature(domain_nodes)
        prev_tms[i] = result_0["TM"]
        print(
            f"[Init] Domain {i}: "
            f"F={result_0['F']:.2f}, "
            f"H={result_0['H']:.4f}, "
            f"T={result_0['T']}"
        )

    total_arrival_count = 0
    total_accept_count = 0
    domain_temperatures = []

    for current_time in range(0, int(events[-1]["time"]) + TIME_WINDOW, TIME_WINDOW):
        print(f"\n=== [NORD] Time Window [{current_time}, {current_time + TIME_WINDOW}) ===")

        window_events = []
        while (
                current_event_id < len(events)
                and events[current_event_id]["time"] < current_time + TIME_WINDOW
        ):
            window_events.append(events[current_event_id])
            current_event_id += 1

        # Handle departures first to release resources.
        for event in window_events:
            if event["type"] == 0:
                handle_leave(event)

        # Then handle arrivals and deploy requests.
        for event in window_events:
            if event["type"] == 1:
                total_arrival_count += 1
                before_accept_num = len(embedded_vnr_history)

                # Advance traffic simulation before arrivals so temperature uses the latest link state.
                multi_simulator.step_and_write(
                    conn,
                    current_time=current_time + TIME_WINDOW,
                    yaml_path=NORD_FLOW_YAML
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

                after_accept_num = len(embedded_vnr_history)
                if after_accept_num > before_accept_num:
                    total_accept_count += 1

        # At window end, advance traffic simulation and record utilization, temperature, and cumulative acceptance.
        multi_simulator.step_and_write(
            conn,
            current_time=current_time + TIME_WINDOW,
            yaml_path=NORD_FLOW_YAML
        )

        total_acceptance_rate = (
            total_accept_count / total_arrival_count
            if total_arrival_count > 0 else 0.0
        )

        record_resource_usage_yaml(
            conn,
            current_time=current_time + TIME_WINDOW,
            yaml_path=NORD_RESOURCE_YAML,
            domain_temperatures=domain_temperatures,
            acceptance_rate=total_acceptance_rate
        )

        print("当前在线 VNR 数量：", len(active_vnrs))
        print("累计到达 VNR 数量：", total_arrival_count)
        print("累计成功嵌入数量：", total_accept_count)
        print("累计接受率：", total_acceptance_rate)

    conn.close()
    print("\n✔ NORD 节点-链路映射自动化事件处理完成")