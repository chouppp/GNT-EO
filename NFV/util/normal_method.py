import json

import numpy as np
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