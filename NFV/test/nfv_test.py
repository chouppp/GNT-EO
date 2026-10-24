from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]

import json
import pickle
import networkx as nx
import pymysql
from NFV.db_config import connect_database
import math
from EO2_test import EO2
from EO2_test_two_stage import EO2_twostage
from NFV.test.nfv_realtime import integrate_slink_result
from NFV.virtual_net_atttribute import Network_node_atrribute
from NFV.network_flow.network_flow import VNRFlowSimulator
from NFV.network_flow.Mutiply_VNR import MultiVNRFlowSimulator
from NFV.network_flow.TM_generator import TrafficMatrixCalculator
import sys
import numpy as np
# -----------------------------
# Helper for reading nodes and edges from this JSON-like/GML-like format.
# -----------------------------
def load_vnet_from_file(filepath):
    """Parse a virtual network from a GML-like file.
        Return nodes and edges, each as a list of dictionaries."""

    nodes = []
    edges = []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Temporary variables.
    current = None
    reading_node = False
    reading_edge = False
    pos_count = 0

    for line in lines:
        line = line.strip()

        # -----------------------
        # Start reading a node.
        # -----------------------
        if line.startswith("node ["):
            current = {}
            reading_node = True
            reading_edge = False
            pos_count = 0
            continue

        # -----------------------
        # Start reading an edge.
        # -----------------------
        if line.startswith("edge ["):
            current = {}
            reading_edge = True
            reading_node = False
            continue

        # -----------------------
        # Read node contents.
        # -----------------------
        if reading_node:
            if line.startswith("]"):
                nodes.append(current)
                reading_node = False
                continue

            parts = line.split()
            key = parts[0]
            val = parts[1]

            if key == "id":
                current["id"] = int(val)

            elif key == "label":
                current["label"] = val.replace('"', "")

            elif key == "pos":
                # Two pos lines.
                if pos_count == 0:
                    current["pos_x"] = float(val)
                else:
                    current["pos_y"] = float(val)
                pos_count += 1

            elif key == "cpu":
                current["cpu"] = int(val)

            elif key == "max_cpu":
                current["max_cpu"] = int(val)

            continue

        # -----------------------
        # Read edge contents.
        # -----------------------
        if reading_edge:
            if line.startswith("]"):
                edges.append(current)
                reading_edge = False
                continue

            parts = line.split()
            key = parts[0]
            val = parts[1]

            if key == "source":
                current["source"] = int(val)

            elif key == "target":
                current["target"] = int(val)

            elif key == "bw":
                current["bw"] = float(val)

            elif key == "max_bw":
                current["max_bw"] = float(val)

            continue

    return nodes, edges

# Load node data from the database.
def load_substrate_nodes_from_db(conn):
    """Read substrate-node information from the database.
        Return {node_id: {'cpu': ..., 'pos': ..., 'pos2': ...}}."""
    sql = "SELECT node_id, cpu, pos_x, pos_y FROM node_single_table"

    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    cursor.close()

    substrate_nodes = {
        row[0]: {
            "cpu": row[1],
            "pos": row[2],
            "pos2": row[3]
        }
        for row in rows
    }

    return substrate_nodes
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
        """
        NOVm_val = 0.0
        for n in substrate_nodes.keys():
            if n == mapped_substrate_node:
                continue

            Cn = substrate_nodes[n]["cpu"]  # 直接从 substrate_nodes 提取 CPU 信息

            NOVm_val += node_function.NOV_mn(
                m=mapped_substrate_node,
                n=n,
                Cm=cpu_val_m,  # 当前节点 CPU
                Cn=Cn,  # 其他节点 CPU
                edges=substrate_edges
            )
        """
        score=RB_m
        fitness_sum += score

    return fitness_sum

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

# Read database data into local storage.
import numpy as np

def load_all_paths_numpy(conn):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT source_node, destination_node, bw_resource, hop_count, link_path
        FROM link_table
        ORDER BY source_node, destination_node, hop_count ASC
    """)
    rows = cursor.fetchall()
    cursor.close()

    # Step 1: convert to NumPy once for performance.
    rows_np = np.array(rows, dtype=object)

    # Split into columns to avoid Python loops.
    src_arr = rows_np[:, 0].astype(np.int32)
    dst_arr = rows_np[:, 1].astype(np.int32)
    bw_arr  = rows_np[:, 2].astype(np.float32)
    hop_arr = rows_np[:, 3].astype(np.int32)
    raw_path_arr = rows_np[:, 4]   # JSON string.

    # Step 1.5: parse JSON link pairs into Python lists.
    # This essential step must not be skipped.
    path_edges_arr = np.array(
        [json.loads(p) for p in raw_path_arr],
        dtype=object
    )

    # Step 2: group by (src, dst).
    pairs = np.stack((src_arr, dst_arr), axis=1)
    unique_pairs = np.unique(pairs, axis=0)

    path_db = {}

    # Step 3: store the structure for each (src, dst).
    for src, dst in unique_pairs:
        mask = (src_arr == src) & (dst_arr == dst)

        path_db[(src, dst)] = {
            "hop": hop_arr[mask],
            "bw": bw_arr[mask],
            "path": path_edges_arr[mask].tolist()  # list of list-of-pairs
        }

    return path_db


def build_changed_edges(mapped_links):
    """Return changed_edges: {(u,v): consumed_bw}."""
    changed_edges = {}

    for item in mapped_links:
        bw = item["bw"]
        link_edges = item["linkpath"]  # Already represented as [[u,v], [v,w], ...].

        for (u, v) in link_edges:
            key = (u, v)

            if key not in changed_edges:
                changed_edges[key] = 0.0

            changed_edges[key] += bw

    return changed_edges


def update_path_db_bandwidth(path_db, bw_dict):
    """Update path bandwidths from residual link bandwidths.
        path_db[(src,dst)] = {'hop': ndarray, 'bw': ndarray, 'path': lists of edge pairs, e.g. [[16,14],[14,17]]}.
        bw_dict maps (u,v) to remaining_bw."""
    for key, record in path_db.items():
        path_list = record["path"]  # list of edge-pair lists
        bw_arr = record["bw"]  # numpy array (float32)

        new_bw_list = []

        # Iterate over all path candidates.
        for edges in path_list:

            # Each edge is [u,v].
            # Look up residual bandwidth; an edge absent from the database is treated as non-bottlenecking.
            edge_bw_list = []
            for (u, v) in edges:
                if (u, v) in bw_dict:
                    edge_bw_list.append(bw_dict[(u, v)])
                else:
                    # If the edge is absent from the update table, its stored bandwidth remains valid.
                    # Treat it as having infinite bandwidth so it does not limit the path.
                    edge_bw_list.append(float("inf"))

            # Path bandwidth is the minimum of its link bandwidths.
            new_bw = min(edge_bw_list)
            new_bw_list.append(new_bw)

        # Update the bandwidth array in path_db.
        record["bw"] = np.array(new_bw_list, dtype=np.float32)


def update_link_resources(conn, changed_edges,path_db):
    """changed_edges is a dictionary, e.g. {(20,17): 27.0, (20,22): 30.0, ...}."""
    cursor = conn.cursor()
    # Step 1: read all link resources from the database.
    cursor.execute("""
        SELECT source_node, destination_node, bw_resource
        FROM link_single_table
    """)
    rows = cursor.fetchall()

    # Convert to a NumPy array.
    rows_np = np.array(rows, dtype=object)
    src_arr = rows_np[:, 0].astype(np.int32)
    dst_arr = rows_np[:, 1].astype(np.int32)
    bw_arr  = rows_np[:, 2].astype(np.float32)

    # Step 2: deduct resources using NumPy.
    for (u, v), used_bw in changed_edges.items():

        mask = (src_arr == u) & (dst_arr == v)

        if mask.any():
            bw_arr[mask] = np.maximum(bw_arr[mask] - used_bw, 0)
        else:
            # Optionally report links missing from the database.
            # print(f"Warning: link ({u},{v}) not found in DB")
            pass

    # Step 3: batch-write to the database.
    sql = """
        UPDATE link_single_table
        SET bw_resource = %s
        WHERE source_node = %s AND destination_node = %s
    """

    batch_data = [
        (float(bw_arr[i]), int(src_arr[i]), int(dst_arr[i]))
        for i in range(len(bw_arr))
    ]

    cursor.executemany(sql, batch_data)
    conn.commit()
    cursor.close()

    # Step 4: build the residual-bandwidth dictionary for path_db updates.
    bw_dict = {(int(src_arr[i]), int(dst_arr[i])): float(bw_arr[i])
               for i in range(len(src_arr))}

    # Step 5: update path bandwidths using bottleneck bandwidth.
    update_path_db_bandwidth(path_db, bw_dict)
    print(f"✔ 链路带宽更新完成，共 {len(batch_data)} 条记录")

# Update node resources.
def update_node_resources(conn, node_mapping, v_nodes):
    """Deduct physical-node CPU according to node mappings and write to the database.
        node_mapping: {virtual_node_id: substrate_node_id}.
        v_nodes: list of dictionaries containing at least id and cpu.
        Database table: node_single_table(node_id, cpu)."""

    # Step 1: build a virtual-node CPU lookup table.
    # v_cpu[v_id] = cpu_demand
    v_cpu = {int(v["id"]): float(v["cpu"]) for v in v_nodes}

    # Step 2: sum CPU deductions for each physical node.
    # cpu_used[s_id] = sum of cpu demands
    cpu_used = {}
    for v_id, s_id in node_mapping.items():
        cpu_used[s_id] = cpu_used.get(s_id, 0.0) + v_cpu[v_id]

    # Step 3: read physical-node CPU from the database.
    cursor = conn.cursor()
    cursor.execute("""
        SELECT node_id, cpu
        FROM node_single_table
    """)
    rows = cursor.fetchall()

    rows_np = np.array(rows, dtype=object)
    node_id_arr = rows_np[:, 0].astype(np.int32)
    cpu_arr     = rows_np[:, 1].astype(np.float32)

    # Step 4: deduct CPU using NumPy.
    for s_id, used_cpu in cpu_used.items():
        mask = (node_id_arr == s_id)
        if mask.any():
            cpu_arr[mask] = np.maximum(cpu_arr[mask] - used_cpu, 0.0)

    # Step 5: batch-write to the database.
    sql = """
        UPDATE node_single_table
        SET cpu = %s
        WHERE node_id = %s
    """
    batch_data = [
        (float(cpu_arr[i]), int(node_id_arr[i]))
        for i in range(len(node_id_arr))
    ]

    cursor.executemany(sql, batch_data)
    conn.commit()
    cursor.close()

    print(f"✔ 节点 CPU 更新完成，涉及物理节点 {len(cpu_used)} 个")
    return cpu_used

# Collect mapping information for all VNRs.
def integrate_slink_result2(
    total_s_link_result: dict,
    s_link_result: list,
    vnr_id: int | None = None
) -> int:
    # Automatically generate VNR IDs.
    if vnr_id is None:
        if not total_s_link_result:
            vnr_id = 0
        else:
            vnr_id = max(total_s_link_result.keys()) + 1

    total_s_link_result[vnr_id] = {
        "slink": s_link_result
    }

    return vnr_id



if __name__ == "__main__":
    # Connect using the shared database.ini in the project root.
    conn = connect_database()


    # Open the saved network model.
    with open(str(_PROJECT_ROOT / 'NFV/p_net_directed.pkl'), "rb") as f:
        G = pickle.load(f)
    # Get virtual-network nodes and links.

    node_function = Network_node_atrribute(conn, G)

    s_edges2 = load_all_paths_numpy(conn)
    print(f"数据库链路数据读取完成")
    # Collect physical-link mapping results.
    total_s_link_result = {}
    # Simulate VNR traffic generation.
    multi_simulator = MultiVNRFlowSimulator(
        total_s_link_result=total_s_link_result,
        alpha_range=(0.7, 1),
        link_noise=0.05
    )
    # Initialize the network temperature calculation matrix.
    tm_calculator = TrafficMatrixCalculator(conn)
    # Temporarily define temperature domains manually for testing.
    domains = [
        [19, 0, 1, 2, 5],
        [9, 4, 10, 11, 12],
        [3, 7, 8, 13, 14],
        [16, 15, 17, 18, 21],
        [6, 20, 22, 23, 24]
    ]
    prev_tms = [None] * len(domains)
    # Initialize temperature calculation.
    for i, domain_nodes in enumerate(domains):
        result_0 = tm_calculator.compute_domain_temperature(domain_nodes)
        prev_tms[i] = result_0["TM"]
        print(f"[Init] Domain {i}: F={result_0['F']:.2f}, "
              f"H={result_0['H']:.4f}, T={result_0['T']}")


    temperatures = []
    for vid in range(1,51):  # vinre_node_1 through vinre_node_10.
      print(f"开始执行第{vid}条链路的映射")
      vnet_path = str(_PROJECT_ROOT / f'NFV/vnet_dataset/vinre0.5_node_{vid}.yaml')
      v_nodes, v_edges = load_vnet_from_file(vnet_path)

      print(v_edges)
      s_edges = node_function.load_edges_from_db()
      # Get physical-network nodes and links.
      s_nodes = load_substrate_nodes_from_db(conn)
      # Perform initial candidate-node filtering.
      result=filter_candidate_nodes(s_nodes,v_nodes,5)
      # Build each virtual node's mapping table from its filtered candidates.
      map_table=build_full_mapping_table(result)
      print(map_table)
      # Check whether the topology can be generated successfully.
      if not validate_mapping_table(map_table):
          continue
      # Create a node-based fitness function.
      # Test node evaluation.
      # deg=node_function.RB_node(m=0,Cm=s_nodes[0]["cpu"],edges=s_edges)
      # Test the fitness function.
      # deg=fitness_func([0.30914602 ,0.30735585 ,0.28300242, 0.77766074, 0.11556166],mapping_table=map_table,substrate_nodes=s_nodes,substrate_edges=s_edges,node_function=node_function)
      # print(deg)
      # Initialize the input function.
      dim = len(v_nodes)
      lb = np.zeros(dim)
      ub = np.ones(dim)


      """"
      EO_test = EO2(
          func=fitness_func,
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
          virtual_edges=v_edges
      )
      """
      EO_twostage_test = EO2_twostage(
          func=fitness_func,
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
      )
      # EO_test.run()
      result = EO_twostage_test.run()

      # Calculate temperature at each time step.
      print(f"\n=== Time step {vid} ===")
      for i, domain_nodes in enumerate(domains):
          temp_result = tm_calculator.compute_domain_temperature(
              domain_nodes,
              prev_tm=prev_tms[i]
          )
          print(f"Domain {i}: F={temp_result['F']:.2f}, "
                f"H={temp_result['H']:.4f}, T={temp_result['T']}")
          # Update prev_tm.
          prev_tms[i] = temp_result["TM"]
      # Skip immediately if mapping fails.
      if result is False or result[0] is False:
          print("❌ 映射失败，跳过该虚拟网络")
          continue  # Use this inside a for loop; otherwise use return.
      # Unpack normally after successful mapping.
      best_list, best_fit, convergence_curve, s_link_result = result
      print(f"最终映射节点为: {best_list}")
      print(f"最终映射结果为: {s_link_result}")
      # Update resources.
      changed_edges = build_changed_edges(s_link_result)
      print(f"需要更改的链路为: {changed_edges}")
      update_link_resources(conn, changed_edges, s_edges2)
      update_node_resources(conn, best_list, v_nodes)
      print("✔ 资源视图更新完成")
      vnr_id = integrate_slink_result(total_s_link_result, s_link_result)
      # Register the new VNR and simulate traffic and temperature changes.
      multi_simulator.add_vnr(
          vnr_id=vnr_id,
          s_link_result=s_link_result
      )
      multi_simulator.step_and_write(conn)




