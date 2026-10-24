from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]

import os
import math
import copy
import random
import pickle
import yaml
import pymysql
from NFV.db_config import connect_database

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


# Save all GA results in the GA folder beside this script.
GA_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GA")
GA_MAPPING_PATH = os.path.join(GA_OUTPUT_DIR, "ga_vne_mapping.yaml")
GA_FLOW_HISTORY_PATH = os.path.join(GA_OUTPUT_DIR, "vnr_flow_history.yaml")
GA_RESOURCE_USAGE_PATH = os.path.join(GA_OUTPUT_DIR, "GA_resource_usage.yaml")


# =========================================================
# 1. Generate GA candidate nodes.
# =========================================================

def build_ga_candidate_nodes(substrate_nodes, virtual_nodes, D=10):
    candidate_dict = {}

    for v in virtual_nodes:
        v_id = int(v["id"])
        v_cpu = float(v["cpu"])
        vx, vy = float(v["pos_x"]), float(v["pos_y"])

        candidate_dict[v_id] = []

        for s_id, s in substrate_nodes.items():
            s_id = int(s_id)
            s_cpu = float(s["cpu"])
            sx, sy = float(s["pos"]), float(s["pos2"])

            if s_cpu < v_cpu:
                continue

            dist = math.sqrt((vx - sx) ** 2 + (vy - sy) ** 2)

            if dist <= D:
                candidate_dict[v_id].append(s_id)

        if len(candidate_dict[v_id]) == 0:
            print(f"❌ 虚拟节点 {v_id} 没有候选物理节点")
            return False

    return candidate_dict


# =========================================================
# 2. Generate a random node mapping.
# =========================================================

def generate_random_node_mapping(candidate_dict, virtual_nodes, one_vn_per_sn=True):
    node_mapping = {}
    used_p_nodes = set()

    # Shuffle virtual-node order.
    v_ids = list(candidate_dict.keys())
    random.shuffle(v_ids)

    for v_id in v_ids:

        available = [
            s for s in candidate_dict[v_id]
            if s not in used_p_nodes
        ]

        if len(available) == 0:
            return False

        selected = random.choice(available)

        node_mapping[v_id] = selected
        used_p_nodes.add(selected)

    return node_mapping

# =========================================================
# 3. Convert virtual links to physical-endpoint requests.
# =========================================================

def convert_virtual_links_to_substrate_links(node_mapping, virtual_edges):
    slink_list = []

    for link in virtual_edges:
        v_src = int(link["source"])
        v_dst = int(link["target"])
        bw = float(link["bw"])

        if v_src not in node_mapping or v_dst not in node_mapping:
            print(f"❌ 虚拟节点 {v_src} 或 {v_dst} 不在 node_mapping 中")
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


# =========================================================
# 4. GA-STP link mapping: shortest feasible paths; duplicated implementation.
# =========================================================
def map_slinks_to_shortest_paths_numpy(slink_list, substrate_edges, substrate_edges2):
    """Map links using shortest feasible paths; follow map_slinks_to_paths_numpy_test in EO_test_two_stage_link.py.
        substrate_edges = s_edges stores residual single-hop bandwidth {(u,v): bw}.
        substrate_edges2 = s_edges2 stores path candidates {(src,dst): {'hop': ..., 'bw': ..., 'path': ...}}."""

    results = []
    failed_links = []
    temp_edge_bw = substrate_edges.copy()

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
            failed_links.append({
                "source": src,
                "target": dst,
                "bw": bw_required,
                "virtual_source": sl["virtual_source"],
                "virtual_target": sl["virtual_target"]
            })

            continue

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

    if len(failed_links) > 0:
        return {
            "success": False,
            "success_links": results,
            "failed_links": failed_links
        }

    return {
        "success": True,
        "success_links": results,
        "failed_links": []
    }

# =========================================================
# 5. Fitness function from the paper.
# =========================================================

def calculate_ga_fitness(
        node_mapping,
        s_link_result,
        virtual_nodes,
        substrate_nodes=None,
        wc=1.0,
        wh=1.0,
        wp=1.0,
        epsilon=1e-6
):
    cpu_cost = sum(float(v["cpu"]) for v in virtual_nodes)

    link_cost = 0.0
    total_hop = 0.0
    total_delay = 0.0

    for item in s_link_result:
        bw = float(item["bw"])
        hop = float(len(item["linkpath"]))

        delay = float(item.get("delay", hop))

        link_cost += bw * hop
        total_hop += hop
        total_delay += delay

    total_cost = cpu_cost + link_cost

    fitness = (
        wc * (1.0 / (total_cost + epsilon))
        + wh * (1.0 / (total_hop + epsilon))
        + wp * (1.0 / (total_delay + epsilon))
    )

    return fitness, total_cost, total_hop, total_delay

# =========================================================
# 6. Evaluate one chromosome.
# =========================================================
def select_repair_virtual_nodes(failed_links, virtual_edges):
    """Select virtual nodes for remapping using the Conciliation Strategy."""

    # Mp: number of failed links associated with each virtual node.
    failed_count = {}

    # Mn: each virtual node's degree in the virtual network.
    degree_count = {}

    for edge in virtual_edges:
        u = int(edge["source"])
        v = int(edge["target"])

        degree_count[u] = degree_count.get(u, 0) + 1
        degree_count[v] = degree_count.get(v, 0) + 1

    for link in failed_links:
        u = int(link["virtual_source"])
        v = int(link["virtual_target"])

        failed_count[u] = failed_count.get(u, 0) + 1
        failed_count[v] = failed_count.get(v, 0) + 1

    repair_nodes = []

    for link in failed_links:
        u = int(link["virtual_source"])
        v = int(link["virtual_target"])

        fu = failed_count.get(u, 0)
        fv = failed_count.get(v, 0)

        if fu > fv:
            repair_nodes.append(u)
        elif fv > fu:
            repair_nodes.append(v)
        else:
            du = degree_count.get(u, 0)
            dv = degree_count.get(v, 0)

            # For tied failure counts, repair lower-degree nodes first to limit the affected area.
            if du <= dv:
                repair_nodes.append(u)
            else:
                repair_nodes.append(v)

    return list(set(repair_nodes))

def conciliation_repair(
        node_mapping,
        failed_links,
        virtual_edges,
        candidate_dict,
        substrate_edges,
        substrate_edges2,
        repair_rounds=5
):
    """Locally repair node mappings: remap a few nodes involved in failed links, then rerun link mapping."""

    repair_nodes = select_repair_virtual_nodes(
        failed_links=failed_links,
        virtual_edges=virtual_edges
    )

    if len(repair_nodes) == 0:
        return False

    for _ in range(repair_rounds):

        new_mapping = copy.deepcopy(node_mapping)

        used_nodes = set(new_mapping.values())
        repair_success = True
        for v_id in repair_nodes:
            old_s_id = new_mapping[v_id]
            used_nodes.discard(old_s_id)

            candidates = candidate_dict[v_id].copy()
            random.shuffle(candidates)

            replaced = False

            for s_id in candidates:
                if s_id == old_s_id:
                    continue

                if s_id in used_nodes:
                    continue

                new_mapping[v_id] = s_id
                used_nodes.add(s_id)
                replaced = True
                break

            if not replaced:
                repair_success = False
                break
        if not repair_success:
            continue

        slink_list = convert_virtual_links_to_substrate_links(
            node_mapping=new_mapping,
            virtual_edges=virtual_edges
        )

        if slink_list is False:
            continue

        result = map_slinks_to_shortest_paths_numpy(
            slink_list=slink_list,
            substrate_edges=substrate_edges,
            substrate_edges2=substrate_edges2
        )

        if result["success"]:
            return {
                "node_mapping": new_mapping,
                "s_link_result": result["success_links"]
            }

    return False
def evaluate_chromosome(
        node_mapping,
        virtual_edges,
        virtual_nodes,
        substrate_nodes,
        substrate_edges,
        substrate_edges2,
        candidate_dict=None
):
    slink_list = convert_virtual_links_to_substrate_links(
        node_mapping=node_mapping,
        virtual_edges=virtual_edges
    )

    if slink_list is False:
        return False

    link_result = map_slinks_to_shortest_paths_numpy(
        slink_list=slink_list,
        substrate_edges=substrate_edges,
        substrate_edges2=substrate_edges2
    )

    if link_result["success"]:
        s_link_result = link_result["success_links"]
    else:
        if candidate_dict is None:
            return False

        repair_result = conciliation_repair(
            node_mapping=node_mapping,
            failed_links=link_result["failed_links"],
            virtual_edges=virtual_edges,
            candidate_dict=candidate_dict,
            substrate_edges=substrate_edges,
            substrate_edges2=substrate_edges2,
            repair_rounds=5
        )

        if repair_result is False:
            return False

        node_mapping = repair_result["node_mapping"]
        s_link_result = repair_result["s_link_result"]

    fitness, cost, hop, delay = calculate_ga_fitness(
        node_mapping=node_mapping,
        s_link_result=s_link_result,
        virtual_nodes=virtual_nodes,
        substrate_nodes=substrate_nodes
    )

    return {
        "node_mapping": node_mapping,
        "s_link_result": s_link_result,
        "fitness": fitness,
        "cost": cost,
        "hop": hop,
        "delay": delay
    }


# =========================================================
# 7. Main GA algorithm.
# =========================================================

def ga_joint_node_link_mapping(
        substrate_nodes,
        substrate_edges,
        substrate_edges2,
        virtual_nodes,
        virtual_edges,
        D=10,
        population_size=10,
        max_iteration=20,
        crossover_prob=0.8,
        mutation_prob=0.2
):
    candidate_dict = build_ga_candidate_nodes(
        substrate_nodes=substrate_nodes,
        virtual_nodes=virtual_nodes,
        D=D
    )

    if candidate_dict is False:
        return False

    population = []

    # Initialize the population.
    for _ in range(population_size * 3):
        if len(population) >= population_size:
            break

        node_mapping = generate_random_node_mapping(
            candidate_dict=candidate_dict,
            virtual_nodes=virtual_nodes,
            one_vn_per_sn=True
        )

        if node_mapping is False:
            continue

        chromosome = evaluate_chromosome(
            node_mapping=node_mapping,
            virtual_edges=virtual_edges,
            virtual_nodes=virtual_nodes,
            substrate_nodes=substrate_nodes,
            substrate_edges=substrate_edges,
            substrate_edges2=substrate_edges2,
            candidate_dict=candidate_dict
        )

        if chromosome is not False:
            population.append(chromosome)

    if len(population) == 0:
        print("❌ GA 初始化失败，没有可行染色体")
        return False

    # Iterative evolution.
    for iteration in range(max_iteration):
        population = sorted(
            population,
            key=lambda x: x["fitness"],
            reverse=True
        )

        new_population = population[:2]
        attempt = 0
        max_attempt = population_size * 20
        while len(new_population) < population_size and attempt < max_attempt:
            attempt += 1
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            child_mapping = copy.deepcopy(parent1["node_mapping"])

            # ---------- Crossover ----------
            if random.random() < crossover_prob:
                v_ids = list(child_mapping.keys())

                if len(v_ids) >= 2:
                    cut = random.randint(1, len(v_ids) - 1)

                    for v_id in v_ids[cut:]:
                        child_mapping[v_id] = parent2["node_mapping"][v_id]

            # Repair duplicate physical-node assignments.
            used = set()
            valid = True

            for v_id in list(child_mapping.keys()):
                if child_mapping[v_id] in used:
                    candidates = candidate_dict[v_id].copy()
                    random.shuffle(candidates)

                    replaced = False

                    for s_id in candidates:
                        if s_id not in used:
                            child_mapping[v_id] = s_id
                            replaced = True
                            break

                    if not replaced:
                        valid = False
                        break

                used.add(child_mapping[v_id])

            if not valid:
                continue

            # ---------- Mutation ----------
            if random.random() < mutation_prob:
                v_id = random.choice(list(child_mapping.keys()))
                candidates = candidate_dict[v_id].copy()
                random.shuffle(candidates)

                used_nodes = set(child_mapping.values())
                used_nodes.discard(child_mapping[v_id])

                for s_id in candidates:
                    if s_id not in used_nodes:
                        child_mapping[v_id] = s_id
                        break

            child = evaluate_chromosome(
                node_mapping=child_mapping,
                virtual_edges=virtual_edges,
                virtual_nodes=virtual_nodes,
                substrate_nodes=substrate_nodes,
                substrate_edges=substrate_edges,
                substrate_edges2=substrate_edges2,
                candidate_dict=candidate_dict
            )

            if child is not False:
                new_population.append(child)

        population = new_population

        best = max(population, key=lambda x: x["fitness"])

        print(
            f"[GA] iter={iteration}, "
            f"fitness={best['fitness']:.6f}, "
            f"cost={best['cost']:.4f}, "
            f"hop={best['hop']}"
        )

    best = max(population, key=lambda x: x["fitness"])

    return best


# =========================================================
# 8. Record YAML results.
# =========================================================

def append_one_record_to_yaml(
        s_link_result_temp,
        vnr_id,
        event_time,
        event_id,
        record_id,
        file_path=GA_MAPPING_PATH
):
    if isinstance(s_link_result_temp, list) and len(s_link_result_temp) > 0:
        if isinstance(s_link_result_temp[0], list):
            s_link_result_temp = s_link_result_temp[0]

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

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
            "linkpath": item["linkpath"]
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

    print(f"✅ 已追加 GA VNR {vnr_id}，event_time={event_time}，包含 {len(links)} 条链路")


# =========================================================
# 9. Arrival events.
# =========================================================

def handle_arrival(event):
    global count
    global conn
    global node_function
    global s_edges2
    global active_vnrs
    global embedded_vnr_history
    global total_s_link_result

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

    best_solution = ga_joint_node_link_mapping(
        substrate_nodes=s_nodes,
        substrate_edges=s_edges,
        substrate_edges2=s_edges2,
        virtual_nodes=v_nodes,
        virtual_edges=v_edges,
        D=10,
        population_size=10,
        max_iteration=20,
        crossover_prob=0.8,
        mutation_prob=0.2
    )

    if best_solution is False:
        print("❌ GA 节点-链路联合映射失败，该 VNR 被拒绝")
        return

    node_mapping = best_solution["node_mapping"]
    s_link_result = best_solution["s_link_result"]

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
        "fitness": best_solution["fitness"],
        "cost": best_solution["cost"],
        "hop": best_solution["hop"],
        "delay": best_solution["delay"],
        "status": "active"
    })
    # Register with the traffic simulator.
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
        file_path=GA_MAPPING_PATH
    )

    print(f"✔ GA VNR {vnr_id} 成功嵌入")
    print("fitness =", best_solution["fitness"])
    print("cost =", best_solution["cost"])
    print("hop =", best_solution["hop"])
    print("arrival changed_edges =", changed_edges)
    print("arrival node_mapping =", node_mapping)


# =========================================================
# 10. Departure events.
# =========================================================

def handle_leave(event):
    global conn
    global s_edges2
    global active_vnrs
    global embedded_vnr_history

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
    # Remove from the traffic simulator.
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

    print(f"✔ GA VNR {vnr_id}（event_id={vnr_event_id}）资源已释放")
    print("leave changed_edges =", changed_edges)
    print("leave node_mapping =", node_mapping)


# =========================================================
# 11. Main function.
# =========================================================

if __name__ == "__main__":

    os.makedirs(GA_OUTPUT_DIR, exist_ok=True)

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
    # Initialize the VNR traffic simulator.
    multi_simulator = MultiVNRFlowSimulator(
        total_s_link_result=total_s_link_result,
        alpha_range=(0.6, 0.9),
        link_noise=0
    )

    # Initialize the network temperature calculator.
    tm_calculator = TrafficMatrixCalculator(conn)

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
                # Also simulate before arrivals; otherwise temperature changes are not captured beforehand.
                multi_simulator.step_and_write(
                    conn,
                    current_time=current_time + TIME_WINDOW,
                    yaml_path=GA_FLOW_HISTORY_PATH
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

        multi_simulator.step_and_write(
            conn,
            current_time=current_time + TIME_WINDOW,
            yaml_path=GA_FLOW_HISTORY_PATH
        )
        record_resource_usage_yaml(
            conn,
            current_time=current_time + TIME_WINDOW,
            yaml_path=GA_RESOURCE_USAGE_PATH
        )

        print("当前在线 VNR 数量：", len(active_vnrs))
        print("成功嵌入数量：", len(embedded_vnr_history))

    print("\n✔ GA 节点-链路联合映射事件处理完成")
