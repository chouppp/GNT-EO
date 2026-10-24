from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]

import json
import os
import pickle
import networkx as nx
import pymysql
from NFV.db_config import connect_database
import math
import yaml
from EO2_test import EO2
from EO_test_two_stage_link import EO2_twostage
from PSO_two_stage import PSO_twostage
from GA_two_stage import GA_twostage
#from EO2_test_two_stage import EO2_twostage
from NFV.virtual_net_atttribute import Network_node_atrribute
from NFV.network_flow.network_flow import VNRFlowSimulator
from NFV.network_flow.Mutiply_VNR import MultiVNRFlowSimulator
from NFV.network_flow.TM_generator import TrafficMatrixCalculator
from NFV.util.normal_method import load_vnet_from_file,load_substrate_nodes_from_db,load_all_paths_numpy
from NFV.util.yingshe import build_changed_edges,update_path_db_bandwidth,update_link_resources,rollback_link_resources,update_node_resources,rollback_node_resources,integrate_slink_result
from NFV.util.yaml import record_resource_usage_yaml  # Remember to move this helper back.
import sys
import numpy as np

# Select the optimizer for one complete, independent simulation run.
# Use "EO2" for the proposed method and "PSO"/"GA" for the baselines.
OPTIMIZER_NAME = "EO2"  # valid values: "EO2", "PSO", "GA"
# PSO an+
# ..d GA append every VNR's iteration history as JSONL records.
RECORD_OPTIMIZER_ITERATION_HISTORY = True
ITERATION_HISTORY_PATHS = {
    "EO2": "iteration_history.jsonl",
    "PSO": "iteration_history.jsonl",
    "GA": "iteration_history.jsonl",
}

# The traffic simulator still advances its state for GNT calculation, but its
# vnr_flow_history.yaml output is discarded when this flag is False.
RECORD_VNR_FLOW_HISTORY = False
VNR_FLOW_HISTORY_PATH = (
    "vnr_flow_history.yaml" if RECORD_VNR_FLOW_HISTORY else os.devnull
)

# -----------------------------
# Helper for reading nodes and edges from this JSON-like/GML-like format.
# -----------------------------

# Following the meta-node definition, filter feasible physical candidates for each virtual node.
def filter_candidate_nodes(substrate_nodes, virtual_nodes, D):
    """substrate_nodes: {sid: {'cpu': ..., 'pos': ..., 'pos2': ...}}.
        virtual_nodes: [{id, cpu, pos_x, pos_y}, ...].
        D: Euclidean-distance threshold."""

    result = {}

    for v in virtual_nodes:
        v_id = v["id"]
        v_cpu = v["cpu"]
        vx, vy = v["pos_x"], v["pos_y"]

        result[v_id] = []

        for s_id, s in substrate_nodes.items():
            sx, sy = s["pos"], s["pos2"]

            # CPU constraint.
            if s["cpu"] < v_cpu:
                continue

            # Euclidean-distance constraint.
            dist = math.sqrt((vx - sx)**2 + (vy - sy)**2)
            if dist > D:
                continue

            result[v_id].append(s_id)

    return result

# Convert one virtual node's physical candidates to a discrete probability distribution.
def candidate_to_probability_intervals(candidate_list):
    n = len(candidate_list)
    if n == 0:
        return {}

    p = 1.0 / n
    intervals = {}
    left = 0.0
    for node in candidate_list:
        right = left + p
        intervals[node] = (round(left, 6), round(right, 6))
        left = right

    return intervals


# Convert a single virtual network.
def build_full_mapping_table(candidate_dict):
    """Example input: {0: [4, 8, 9, 16, 18, 20], 1: [5, 14, 15, 19], ...}.
        Output: {0: {'candidates': [...], 'probabilities': [...], 'intervals': {...}}, ...}."""

    mapping_table = {}

    for v_id, cand_list in candidate_dict.items():
        n = len(cand_list)

        # Uniform probability distribution.
        if n > 0:
            probabilities = [1.0 / n] * n
        else:
            probabilities = []

        # Interval distribution.
        intervals = candidate_to_probability_intervals(cand_list)

        mapping_table[v_id] = {
            "candidates": cand_list,
            "probabilities": probabilities,
            "intervals": intervals
        }

    return mapping_table

# Define the fitness function.
def fitness_func(X_row, mapping_table, substrate_nodes, substrate_edges, node_function):
    """X_row, e.g. [0.1, 0.8, 0.3], contains one probability value per virtual node for interval lookup.
        mapping_table example: {0: {'intervals': {4: (0,0.16), 8: (0.16,0.33), ...}}, 1: {...}, 2: {...}}."""
    fitness_sum = 0.0
    # The index in X_row is the virtual-node ID.
    for virtual_node_id, r in enumerate(X_row):

        # A virtual node missing from the table makes the solution invalid.
        if virtual_node_id not in mapping_table:
            return -99999

        intervals = mapping_table[virtual_node_id]["intervals"]

        # Find the interval containing r.
        mapped_substrate_node = None

        for s_node, (low, high) in intervals.items():
            if low <= r <= high:
                mapped_substrate_node = s_node
                break

        if mapped_substrate_node is None:
            return -99999

        # Look up substrate-node information.
        if mapped_substrate_node not in substrate_nodes:
            return -99999

        cpu_val_m = substrate_nodes[mapped_substrate_node]["cpu"]

        # Call the node-evaluation function.
        RB_m = node_function.RB_node(
            m=mapped_substrate_node,
            Cm=cpu_val_m,
            edges=substrate_edges
        )
        score=RB_m
        fitness_sum += score

    return fitness_sum


def fitness_func_link(
    X_row,
    mapping_table,
    substrate_nodes,
    substrate_edges,
    node_function,
    s_link_result=None,
    w_node=0.5,
    w_link=0.5,
    eps=1e-9
):
    fitness_sum = 0.0

    # Node evaluation.
    for virtual_node_id, r in enumerate(X_row):

        if virtual_node_id not in mapping_table:
            return -99999

        intervals = mapping_table[virtual_node_id]["intervals"]

        mapped_substrate_node = None
        for s_node, (low, high) in intervals.items():
            if low <= r <= high:
                mapped_substrate_node = s_node
                break

        if mapped_substrate_node is None:
            return -99999

        if mapped_substrate_node not in substrate_nodes:
            return -99999

        cpu_val_m = substrate_nodes[mapped_substrate_node]["cpu"]

        RB_m = node_function.RB_node(
            m=mapped_substrate_node,
            Cm=cpu_val_m,
            edges=substrate_edges
        )

        fitness_sum += RB_m

    node_fitness = fitness_sum

    # Link evaluation.
    if s_link_result is None or s_link_result is False:
        return -99999

    link_cost = 0.0
    for item in s_link_result:
        bw = item["bw"]
        hop = item["hop"]
        link_cost += bw * hop

    link_score = 1.0 / (link_cost + eps)

    # Weighted sum.
    #total_fitness = w_node * node_fitness + w_link * link_score
    total_fitness =  w_link * link_score
    return total_fitness

def fitness_func_link_afbd(
    s_link,
    substrate_nodes,
    substrate_edges,
    virtual_degrees,
    s_link_result=None,
    w_node=1.0,
    w_link=1.0,
    eps=1e-9,
    allow_same_substrate=False
):
    # Step 0: process s_link.
    if isinstance(s_link, list):
        if len(s_link) == 0:
            print("奇怪的情况")
            return -99999
        if isinstance(s_link[0], dict):
            s_link = s_link[0]
        else:
            print("奇怪的情况")
            return -99999

    # Step 1: validate s_link.
    if s_link is None or s_link is False or len(s_link) == 0:
        print("输入的slink为空")
        return -99999

    #for v_node, s_node in s_link.items():
        #if s_node not in substrate_nodes:
            # print("Physical node not found?")
            #return -99999

    # Step 4: count physical-node degrees.
    substrate_degree = {}
    for (u, v) in substrate_edges.keys():
        substrate_degree[u] = substrate_degree.get(u, 0) + 1
        substrate_degree[v] = substrate_degree.get(v, 0) + 1

    # Step 5: calculate the AFBD node term.
    node_cost = 0.0
    for v_node, s_node in s_link.items():
        deg_v = virtual_degrees.get(v_node, 0)
        deg_s = substrate_degree.get(s_node, 0)

        # Recommended for greater robustness.
        node_cost += abs(deg_s - deg_v)

        # To follow the paper's original formula exactly, use:
        # node_cost += (deg_s - deg_v)

    # Step 6: validate link mapping.
    if s_link_result is None or s_link_result is False:
        print("输入的slink为空")
        return -99999

    # Step 7: calculate the link term.
    link_cost = 0.0
    for item in s_link_result:
        bw = item["bw"]
        hop = item["hop"]
        link_cost += bw * hop

    # Step 8: combine total cost.
    total_cost = w_node * node_cost + w_link * link_cost

    if total_cost < 0:
        total_cost = 0.0

    # Step 9: convert to fitness where larger is better.
    fitness =total_cost
    return fitness

def fitness_func_link_tempreature(
    s_link,
    substrate_nodes,
    substrate_edges,
    virtual_degrees,
    s_link_result=None,
    domain_temperatures=None,
    domains=None,
    w_node=0.45,
    w_link=0.45,
    w_temp=0.1
):
    # Step 0: process s_link.
    if isinstance(s_link, list):
        if len(s_link) == 0:
            return -99999
        if isinstance(s_link[0], dict):
            s_link = s_link[0]
        else:
            return -99999

    # Step 1: validate s_link.
    if s_link is None or s_link is False or len(s_link) == 0:
        return -99999

    # Step 2: count physical-node outdegrees.
    substrate_degree = {node: 0 for node in substrate_nodes}

    for (u, v) in substrate_edges.keys():
        if u in substrate_degree:
            substrate_degree[u] += 1
        else:
            substrate_degree[u] = 1

        if v not in substrate_degree:
            substrate_degree[v] = 0

    # Step 3: node term.
    node_cost = 0.0
    for v_node, s_node in s_link.items():
        deg_v = virtual_degrees.get(v_node, 0)   # Prefer supplying virtual-node outdegrees here.
        deg_s = substrate_degree.get(s_node, 0)
        node_cost += abs(deg_s - deg_v)

    # Step 4: link term.
    if s_link_result is None or s_link_result is False:
        return -99999

    link_cost = 0.0
    for item in s_link_result:
        bw = item["bw"]
        hop = item["hop"]
        link_cost += bw * hop

    # Step 5: optional temperature term.
    temp_cost = 0.0

    if domain_temperatures is not None and domains is not None:

        # Build the node-to-domain mapping.
        node_to_domain = {}
        for d_id, nodes in enumerate(domains):
            for node in nodes:
                node_to_domain[node] = d_id

        # Build the domain-to-temperature mapping.
        domain_temp_dict = {}
        for item in domain_temperatures:
            d_id = item["domain_id"]
            T = item["T"]

            if T is None:
                domain_temp_dict[d_id] = 1.0
            else:
                domain_temp_dict[d_id] = max(float(T), 0.0)

        # Count hosted virtual nodes in each domain.
        domain_node_count = {}
        total_v_nodes = len(s_link)

        for v_node, s_node in s_link.items():
            d_id = node_to_domain.get(s_node, None)
            if d_id is None:
                continue
            domain_node_count[d_id] = domain_node_count.get(d_id, 0) + 1

        # Weighted mean temperature.
        if total_v_nodes > 0:
            for d_id, cnt in domain_node_count.items():
                ratio = cnt / total_v_nodes
                temp_cost += ratio * domain_temp_dict.get(d_id, 0.0)

    # Step 6: combine total cost.
    total_cost = (
        w_node * node_cost +
        w_link * link_cost +
        w_temp * temp_cost
    )

    return total_cost



def extract_physical_nodes(intervals):
    """Normalize mapping_table[v]['intervals'] to a list, accepting list, dict, or set inputs."""
    # Case 1: already a list.
    if isinstance(intervals, list):
        return intervals

    # Case 2: use dictionary keys as candidate physical nodes.
    if isinstance(intervals, dict):
        return list(intervals.keys())

    # Case 3: convert a set to a list.
    if isinstance(intervals, set):
        return list(intervals)

    raise ValueError(f"无法解析 intervals 的格式: {intervals}")
# Check feasibility of the initial meta-node mapping.
def validate_mapping_table(mapping_table):

    forced_map = {}  # {physical_node: [virtual_nodes]}.

    for v_id, info in mapping_table.items():

        if "intervals" not in info:
            print(f"❌ 虚拟节点 {v_id} 缺少 intervals 字段")
            return False

        # Normalize to a list.
        intervals = extract_physical_nodes(info["intervals"])

        if len(intervals) == 0:
            print(f"❌ 虚拟节点 {v_id} 没有可用物理节点")
            return False

        # Check conflicts among nodes with a single candidate.
        if len(intervals) == 1:
            p = intervals[0]   # This now avoids KeyError.

            if p not in forced_map:
                forced_map[p] = []

            forced_map[p].append(v_id)

    # Check all conflicts consistently.
    for p, v_list in forced_map.items():
        if len(v_list) > 1:
            print(f"❌ 冲突：物理节点 {p} 被多个虚拟节点强制占用: {v_list}")
            return False

    print("✔ 映射表检查通过")
    return True
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

# Process an arriving VNR.
def handle_arrival(event,domain_temperatures,domains):

    vnr_id_temp = event["v_net_id"]
    event_time=event["time"]
    event_id=event["id"]
    print(f"\n🟢 Arrival VNR {vnr_id_temp} at t={event['time']:.2f}")

    vnet_path = str(_PROJECT_ROOT / f'NFV/vnet_dataset/vinre0.5_node_{vnr_id_temp}.yaml')
    v_nodes, v_edges = load_vnet_from_file(vnet_path)

    s_nodes = load_substrate_nodes_from_db(conn)
    s_edges = node_function.load_edges_from_db()

    result = filter_candidate_nodes(s_nodes, v_nodes, 10)
    map_table = build_full_mapping_table(result)
    if not validate_mapping_table(map_table):
        print("❌ 无法构建映射表")
        return

    dim = len(v_nodes)
    lb = np.zeros(dim)
    ub = np.ones(dim)

    optimizer_classes = {
        "EO2": EO2_twostage,
        "PSO": PSO_twostage,
        "GA": GA_twostage,
    }
    if OPTIMIZER_NAME not in optimizer_classes:
        raise ValueError(
            f"Unsupported OPTIMIZER_NAME={OPTIMIZER_NAME!r}; "
            f"choose from {tuple(optimizer_classes)}."
        )

    optimizer_cls = optimizer_classes[OPTIMIZER_NAME]
    print(f"🚀 Optimizer: {OPTIMIZER_NAME}")

    optimizer_kwargs = dict(
        func=fitness_func_link_tempreature,
        lb=lb,
        ub=ub,
        dim=dim,
        pop_size=5,
        iters=220,
        mapping_table=map_table,
        substrate_nodes=s_nodes,
        substrate_edges=s_edges,
        node_function=node_function,
        conn=conn,
        virtual_edges=v_edges,
        substrate_edges2=s_edges2,
        domain_temperatures=domain_temperatures,
        domains=domains,
    )

    # These arguments belong only to the comparison optimizers.
    if OPTIMIZER_NAME in ("PSO", "GA"):
        optimizer_kwargs.update(
            record_iteration_history=RECORD_OPTIMIZER_ITERATION_HISTORY,
            iteration_history_path=ITERATION_HISTORY_PATHS[OPTIMIZER_NAME],
            vnr_id=vnr_id_temp,
        )

    EO = optimizer_cls(**optimizer_kwargs)

    result = EO.run()

    if result is False or result[0] is False:
        print("❌ 映射失败（该 VNR 被拒绝）")
        return

    best_list_temp, best_fit, _, s_link_result_temp = result

    # Update resources.
    changed_edges = build_changed_edges(s_link_result_temp)
    update_link_resources(conn, changed_edges, s_edges2)
    update_node_resources(conn, best_list_temp, v_nodes)
    print("arrival changed_edges =", changed_edges)
    print("arrival best_list =", best_list_temp)
    # Register in active_vnrs.
    vnr_evet_id_temp = integrate_slink_result(total_s_link_result, s_link_result_temp)
    active_vnrs[vnr_evet_id_temp] = {
        "best_list": best_list_temp,
        "s_link_result": s_link_result_temp,
        "vnr_id":vnr_id_temp,

    }

    embedded_vnr_history.append({
        "event_id": vnr_evet_id_temp,
        "vnr_id": vnr_id_temp,
        "arrival_time": event["time"],
        "best_fit": best_fit,
        "node_mapping": best_list_temp,
        "link_mapping": s_link_result_temp,
        "changed_edges": changed_edges,
        "optimizer": OPTIMIZER_NAME,
        "status": "active"
    })

    # Register the new VNR and simulate traffic and temperature changes.
    # Leave this working code as is for now; previous edits became confusing.
    multi_simulator.add_vnr(
        vnr_id=vnr_evet_id_temp,
        s_link_result=s_link_result_temp,
        vnr_id2=vnr_id_temp
    )
    #multi_simulator.step_and_write(conn)
    print(f"✔ VNR {vnr_id_temp} 成功嵌入")
    # Save records.

    append_one_record_to_yaml(s_link_result_temp=s_link_result_temp,vnr_id=vnr_id_temp,event_time=event_time,event_id=event_id)


def handle_leave(event):
    vnr_id = event["v_net_id"]
    print(f"\n🔴 Leave VNR {vnr_id} at t={event['time']:.2f}")
    vnet_path = str(_PROJECT_ROOT / f'NFV/vnet_dataset/vinre0.5_node_{vnr_id}.yaml')
    v_nodes, v_edges= load_vnet_from_file(vnet_path)

    # 1. Find vnr_event_id from vnr_id.
    vnr_event_id = None
    for eid, record in active_vnrs.items():
        if record["vnr_id"] == vnr_id:
            vnr_event_id = eid
            break

    if vnr_event_id is None:
        print("⚠ VNR 未在服务中，忽略")
        return

    # 2. Retrieve the record.
    record = active_vnrs.pop(vnr_event_id)
    s_link_result = record["s_link_result"]
    best_list = record["best_list"]

    # 3. Release resources using the event ID.
    multi_simulator.remove_vnr(
        vnr_id=vnr_event_id,
        #s_link_result=s_link_result
    )
    # Update traffic when releasing a request as well.
    #multi_simulator.step_and_write(conn)
    # Update resources.
    changed_edges = build_changed_edges(s_link_result)
    rollback_link_resources(conn, changed_edges, s_edges2)
    rollback_node_resources(conn, best_list, v_nodes)
    print("leave changed_edges =", changed_edges)
    print("leave best_list =", best_list)
    # Roll back database resources if needed.
    # rollback_link_resources(conn, s_link_result)
    # rollback_node_resources(conn, best_list)

    print(f"✔ VNR {vnr_id}（event_id={vnr_event_id}）资源已释放")



if __name__ == "__main__":
    # Connect using the shared database.ini in the project root.
    conn = connect_database()

    # Global variables.
    count = 0

    # Open the saved network model.
    with open(str(_PROJECT_ROOT / 'NFV/p_net_directed.pkl'), "rb") as f:
        G = pickle.load(f)
    # Get VNR mapping events.
    with open("vnr_events.yaml", "r", encoding="utf-8") as f:
        event_data = yaml.safe_load(f)

    events = event_data["events"]
    events = sorted(events, key=lambda e: e["time"])

    TIME_WINDOW = 100
    current_event_id = 0
    active_vnrs = {}  # vnr_id -> mapping information.
    embedded_vnr_history = []   # History of successful mappings.
    # Get virtual-network nodes and links.

    node_function = Network_node_atrribute(conn, G)

    s_edges2 = load_all_paths_numpy(conn)
    print(f"数据库链路数据读取完成")
    # Collect physical-link mapping results.
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

        # Process departures first.
        for event in window_events:
            if event["type"] == 0:
                handle_leave(event)

        # Then process arrivals.
        for event in window_events:
            if event["type"] == 1:
                # Record the acceptance rate.
                total_arrival_count += 1
                before_accept_num = len(embedded_vnr_history)
                # Also simulate before arrivals; otherwise temperature changes are not captured beforehand.
                multi_simulator.step_and_write(
                    conn,
                    current_time=current_time + TIME_WINDOW,
                    yaml_path=VNR_FLOW_HISTORY_PATH
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

                handle_arrival(event, domain_temperatures, domains)
                # Record the acceptance rate.
                after_accept_num = len(embedded_vnr_history)

                if after_accept_num > before_accept_num:
                    total_accept_count += 1

        # Finally record actual traffic of online VNRs at the end of this window.
        multi_simulator.step_and_write(
            conn,
            current_time=current_time + TIME_WINDOW,
            yaml_path=VNR_FLOW_HISTORY_PATH
        )
        # Record the acceptance rate.
        if total_arrival_count > 0:
            total_acceptance_rate = total_accept_count / total_arrival_count
        else:
            total_acceptance_rate = 0.0
        record_resource_usage_yaml(
            conn,
            current_time=current_time + TIME_WINDOW,
            yaml_path="resource_usage.yaml",
            domain_temperatures=domain_temperatures,
            acceptance_rate=total_acceptance_rate,
        )
        print("当前在线 VNR 数量：", len(multi_simulator.vnr_simulators))

        print(embedded_vnr_history[-1])
        print("成功嵌入数量：", len(embedded_vnr_history))