import numpy as np


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


def update_path_db_bandwidth(path_db, updated_bw_dict, bw_dict):
    """Update only path_db paths affected by updated_bw_dict.
        Recompute bottlenecks using the complete bw_dict.
    
        Parameters
        ----------
        path_db: mapping from (src,dst) to hop and bw ndarrays and lists of edge-pair paths.
            Example: path_db[(1,5)] = {'hop': np.array([2,3]), 'bw': np.array([10.0,8.0], dtype=np.float32), 'path': [[[1,2],[2,5]], [[1,3],[3,4],[4,5]]]}.
        updated_bw_dict: residual bandwidths of modified links only, e.g. {(20,17): 27.0, (20,22): 30.0}.
        bw_dict: complete current single-hop residual bandwidths, e.g. {(1,2): 40.0, (2,5): 20.0, (20,17): 27.0, ...}."""
    if not updated_bw_dict:
        print("update_path_db_bandwidth: updated_bw_dict 为空，无需更新")
        return

    changed_edges = set(updated_bw_dict.keys())

    updated_pair_count = 0   # Number of updated (src, dst) pairs.
    updated_path_count = 0   # Number of individual updated paths.

    for key, record in path_db.items():
        path_list = record.get("path", [])
        old_bw_arr = record.get("bw", None)

        if old_bw_arr is None or len(path_list) == 0:
            continue

        # Copy the old bw array to avoid problems during in-place updates.
        new_bw_arr = np.array(old_bw_arr, copy=True, dtype=np.float32)

        pair_updated = False

        for idx, edges in enumerate(path_list):
            # Convert path edges to tuples so [u,v] and (u,v) use a consistent representation.
            edge_tuples = [tuple(edge) for edge in edges]

            # Skip paths containing no affected links.
            if not any(edge in changed_edges for edge in edge_tuples):
                continue

            # Recompute bottleneck bandwidth for this affected path.
            edge_bw_list = []
            missing_edge = False

            for edge in edge_tuples:
                if edge in bw_dict:
                    edge_bw_list.append(bw_dict[edge])
                else:
                    # Normally this should not be missing.
                    print(f"Warning: path_db{key} 的路径 {edges} 中链路 {edge} 不在 bw_dict 中，跳过该路径更新")
                    missing_edge = True
                    break

            if missing_edge or len(edge_bw_list) == 0:
                continue

            # Path bandwidth is the minimum residual bandwidth over all its links.
            new_bw = min(edge_bw_list)
            new_bw_arr[idx] = np.float32(new_bw)

            updated_path_count += 1
            pair_updated = True

        if pair_updated:
            record["bw"] = new_bw_arr
            updated_pair_count += 1

    print(f"✔ path_db 更新完成：共更新 {updated_pair_count} 个(src,dst)，{updated_path_count} 条路径")


def update_link_resources(conn, changed_edges, path_db):
    """
    changed_edges: dict {(u, v): used_bw}
    """
    if not changed_edges:
        return {}

    cursor = conn.cursor()

    # Step 1: update only links in changed_edges.
    sql = """
        UPDATE link_single_table
        SET bw_resource = GREATEST(bw_resource - %s, 0)
        WHERE source_node = %s AND destination_node = %s
    """

    batch_data = [
        (float(used_bw), int(u), int(v))
        for (u, v), used_bw in changed_edges.items()
    ]

    cursor.executemany(sql, batch_data)
    conn.commit()

    # Step 2: read all link bandwidths to update path_db.
    cursor.execute("""
        SELECT source_node, destination_node, bw_resource
        FROM link_single_table
    """)
    rows = cursor.fetchall()
    cursor.close()

    bw_dict = {
        (int(u), int(v)): float(bw)
        for u, v, bw in rows
    }

    # Step 3: collect the latest bandwidths of affected links.
    updated_bw_dict = {
        (int(u), int(v)): bw_dict[(int(u), int(v))]
        for (u, v) in changed_edges.keys()
        if (int(u), int(v)) in bw_dict
    }

    # Step 4: update path_db.
    update_path_db_bandwidth(path_db, updated_bw_dict, bw_dict)

    print(f"✔ 链路带宽更新完成，实际更新 {len(batch_data)} 条链路")
    return updated_bw_dict
# Release link resources.
def rollback_link_resources(conn, changed_edges, path_db):
    """
    changed_edges: dict {(u, v): freed_bw}
    """
    if not changed_edges:
        return {}

    cursor = conn.cursor()

    # Step 1: release only links in changed_edges.
    sql = """
        UPDATE link_single_table
        SET bw_resource = bw_resource + %s
        WHERE source_node = %s AND destination_node = %s
    """

    batch_data = [
        (float(freed_bw), int(u), int(v))
        for (u, v), freed_bw in changed_edges.items()
    ]

    cursor.executemany(sql, batch_data)
    conn.commit()

    # Step 2: read all link bandwidths to update path_db.
    cursor.execute("""
        SELECT source_node, destination_node, bw_resource
        FROM link_single_table
    """)
    rows = cursor.fetchall()
    cursor.close()

    bw_dict = {
        (int(u), int(v)): float(bw)
        for u, v, bw in rows
    }

    # Step 3: collect the latest bandwidths of affected links.
    updated_bw_dict = {
        (int(u), int(v)): bw_dict[(int(u), int(v))]
        for (u, v) in changed_edges.keys()
        if (int(u), int(v)) in bw_dict
    }

    # Step 4: update path_db.
    update_path_db_bandwidth(path_db, updated_bw_dict, bw_dict)

    print(f"✔ 链路带宽释放完成，实际回滚 {len(batch_data)} 条链路")
    return updated_bw_dict

# Update node resources.
def update_node_resources(conn, node_mapping, v_nodes):
    # v_cpu[v_id] = cpu_demand
    v_cpu = {int(v["id"]): float(v["cpu"]) for v in v_nodes}

    # Sum CPU deductions for each physical node.
    cpu_used = {}
    for v_id, s_id in node_mapping.items():
        s_id = int(s_id)
        cpu_used[s_id] = cpu_used.get(s_id, 0.0) + v_cpu[int(v_id)]

    if not cpu_used:
        return {}

    cursor = conn.cursor()

    sql = """
        UPDATE node_single_table
        SET cpu = GREATEST(cpu - %s, 0)
        WHERE node_id = %s
    """

    batch_data = [
        (float(used_cpu), int(s_id))
        for s_id, used_cpu in cpu_used.items()
    ]

    cursor.executemany(sql, batch_data)
    conn.commit()
    cursor.close()

    print(f"✔ 节点 CPU 更新完成，涉及物理节点 {len(cpu_used)} 个")
    return cpu_used


# Release node CPU resources.
def rollback_node_resources(conn, node_mapping, v_nodes):
    v_cpu = {int(v["id"]): float(v["cpu"]) for v in v_nodes}

    cpu_freed = {}
    for v_id, s_id in node_mapping.items():
        s_id = int(s_id)
        cpu_freed[s_id] = cpu_freed.get(s_id, 0.0) + v_cpu[int(v_id)]

    if not cpu_freed:
        return {}

    cursor = conn.cursor()

    sql = """
        UPDATE node_single_table
        SET cpu = cpu + %s
        WHERE node_id = %s
    """

    batch_data = [
        (float(freed_cpu), int(s_id))
        for s_id, freed_cpu in cpu_freed.items()
    ]

    cursor.executemany(sql, batch_data)
    conn.commit()
    cursor.close()

    print(f"✔ 节点 CPU 资源释放完成，涉及物理节点 {len(cpu_freed)} 个")
    return cpu_freed
# Collect mapping information for all VNRs.
def integrate_slink_result(total_s_link_result, s_link_result) -> int:
    """Append this VNR's link mapping to total_s_link_result and return the new vnr_event_id, incrementing from zero."""

    # vnr_event_id increments from zero.
    if not total_s_link_result:
        vnr_event_id = 0
    else:
        vnr_event_id = max(total_s_link_result.keys()) + 1

    total_s_link_result[vnr_event_id] = {
        "slink": s_link_result
    }

    return vnr_event_id


def map_slinks_to_paths_numpy(self, slink_list):
        results = []
        # Sort by bandwidth first.
        slink_list = sorted(slink_list, key=lambda x: x["bw"], reverse=True)
        # Temporary copy of the current global residual bandwidth.
        temp_edge_bw = self.substrate_edges.copy()

        for sl in slink_list:
            src = sl["source"]
            dst = sl["target"]
            bw_required = sl["bw"]
            key = (src, dst)
            # 1. Handle missing paths for (src,dst).
            if key not in self.substrate_edges2:
                print(f"❌ 未找到链路 {src} → {dst}")
                return False  # Fail immediately.
            hop_arr = self.substrate_edges2[key]["hop"]  # NumPy array
            bw_arr = self.substrate_edges2[key]["bw"]  # NumPy array
            path_list = self.substrate_edges2[key]["path"]
            # 2. Find bandwidth-feasible paths.
            valid_idx = np.where(bw_arr >= bw_required)[0]
            if valid_idx.size == 0:
                print(f"❌ 带宽不足 {src} → {dst} (需要 {bw_required})")
                return False  # Fail immediately.
            # 3. Since hops are sorted, choose the feasible path with the fewest hops.
            i = valid_idx[0]

            result_item = {
                "source": src,
                "target": dst,
                "bw": bw_required,
                "hop": hop_arr[i],
                "linkpath": path_list[i]
            }

            results.append(result_item)

        # 4. All virtual links have been mapped successfully.
        return results