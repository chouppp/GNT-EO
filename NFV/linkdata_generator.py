import pickle
import networkx as nx
import pymysql
if __package__:
    from .db_config import connect_database
else:
    from db_config import connect_database
# Query the database.
def db_query(conn, sql, args=None):
    with conn.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(sql, args)
        return cursor.fetchall()
# Insert into the database.
def db_execute(conn, sql, args=None):
    with conn.cursor() as cursor:
        cursor.execute(sql, args)
    conn.commit()

def build_like_conditions(a, b):
    pat1 = f"'{a},{b},%'"     # Beginning.
    pat2 = f"'%,{a},{b},%'"   # Middle.
    pat3 = f"'%,{a},{b}'"     # End.
    return f"(link_path LIKE {pat1} OR link_path LIKE {pat2} OR link_path LIKE {pat3})"
# Load single-hop link data.
def load_link_resources(conn):
    sql = "SELECT source_node, destination_node, bw_resource FROM link_single_table"
    rows = db_query(conn, sql)

    link_bw = {}
    for r in rows:
        u = r["source_node"]
        v = r["destination_node"]
        bw = r["bw_resource"]
        # Directed graph; no need to store both directions here.
        link_bw[(u,v)] = bw


    return link_bw
# Generate paths, hop counts, and resources between node pairs.
def find_all_directed_paths(G, link_bw, cutoff=5):
    """Find all reachable simple paths in directed graph G; cutoff limits path length.
        Return all_paths[(u, v)] = [{path, hop, bw}, ...]."""
    all_paths = {}

    for u in G.nodes():
        for v in G.nodes():
            if u == v:
                continue

            # Get all simple (cycle-free) paths from u to v.
            try:
                paths = list(nx.all_simple_paths(G, u, v, cutoff=cutoff))
            except nx.NetworkXNoPath:
                paths = []

            path_infos = []

            for path in paths:
                hops = len(path) - 1

                # Path bottleneck bandwidth is the minimum over its single-hop links.
                min_bw = float("inf")
                for a, b in zip(path[:-1], path[1:]):
                    min_bw = min(min_bw, link_bw.get((a, b), 0))

                path_infos.append({
                    "path": path,
                    "hop": hops,
                    "bw": min_bw
                })

            all_paths[(u, v)] = path_infos

    return all_paths

def cut_off_path_single(G,source_node,destinantion_node):
    all_paths = {}
    paths = list(nx.all_simple_paths(G, source_node, destinantion_node, cutoff=6))  # At most six hops.
    path_infos = []
    for path in paths:
        hops = len(path) - 1
        path_infos.append({
            "path": path,
            "hop": hops
        })
    all_paths[(source_node, destinantion_node)] = path_infos

    return all_paths

# -----------------------------------------
# 4. Write results to link_table.
# -----------------------------------------
import json

def insert_paths_into_db(conn, all_paths):
    """Database representation: link_path is a JSON array, e.g. [[16,18],[18,15]]."""
    sql = """
    INSERT INTO link_table 
        (source_node, destination_node, hop_count, bw_resource, max_bw_resource, link_path)
    VALUES 
        (%s, %s, %s, %s, %s, %s)
    """

    batch_data = []

    for (u, v), path_list in all_paths.items():
        for info in path_list:
            path_nodes = info["path"]   # For example, [16,18,15].
            bw = info["bw"]
            hop = info["hop"]

            # Key step: construct the list of link pairs.
            edge_pairs = [
                [path_nodes[i], path_nodes[i+1]]
                for i in range(len(path_nodes) - 1)
            ]

            # Serialize as JSON for insertion into a MySQL JSON column.
            edge_json = json.dumps(edge_pairs, ensure_ascii=False)

            batch_data.append((u, v, hop, bw, bw, edge_json))

    # Batch-insert into the database.
    cursor = conn.cursor()
    cursor.executemany(sql, batch_data)
    conn.commit()
    cursor.close()

    print(f"✨ 批量插入完成，共 {len(batch_data)} 行")

# Find affected strings.
def get_affected_ids(conn, changed_edges):
    where_clauses = []
    for (a, b) in changed_edges:
        where_clauses.append(build_like_conditions(a, b))

    where_sql = " OR ".join(where_clauses)
    sql = f"SELECT link_id, link_path FROM link_table WHERE {where_sql}"
    return db_query(conn, sql)

# Experimental fast update method.
def update_path_bw_fast(conn, changed_edges):

    # Step 1: identify affected paths.
    affected_rows = get_affected_ids(conn, changed_edges)

    update_sql = """
    UPDATE link_table
    SET bw_resource = %s
    WHERE link_id = %s
    """

    batch_updates = []

    for row in affected_rows:
        path = list(map(int, row["link_path"].split(',')))
        edges = list(zip(path[:-1], path[1:]))

        min_bw = float('inf')
        for (a, b) in edges:
            if (a, b) in changed_edges:
                min_bw = min(min_bw, changed_edges[(a, b)])

        batch_updates.append((min_bw, row["link_id"]))

    # Batch updates in a single commit.
    cursor = conn.cursor()
    cursor.executemany(update_sql, batch_updates)
    conn.commit()
    cursor.close()

    print(f"✨ 更新完成，共 {len(batch_updates)} 条路径被更新")

# Update path resources after orchestration.
def update_path_bw(conn, changed_edges):
    """Update bw_resource and max_bw_resource in link_table for paths using changed_edges.
        changed_edges maps (u, v) to new_bw. Include (v, u) when needed for undirected links."""

    # Read the path table.
    sql = "SELECT link_id, link_path FROM link_table"
    rows = db_query(conn, sql)

    update_sql = """
    UPDATE link_table
    SET bw_resource = %s
    WHERE link_id = %s
    """

    for row in rows:
        path_str = row["link_path"]
        hops = list(map(int, path_str.split(",")))
        edges = list(zip(hops[:-1], hops[1:]))

        # Check whether the path contains any edge in changed_edges.
        affected = any(
            (a, b) in changed_edges or (b, a) in changed_edges
            for a, b in edges
        )

        if not affected:
            continue  # Skip unaffected paths.

        # Calculate path bandwidth.
        min_bw = 100
        for a, b in edges:
            if (a, b) in changed_edges:
                bw = changed_edges[(a, b)]
            else:
                continue  # This edge is unchanged and needs no update.

            min_bw = min(min_bw, bw)

        # Update the path.
        db_execute(conn, update_sql, (min_bw, row["link_id"]))

    print("✨ link_table 所有受影响路径 bw_resource 已更新完成！")


# Update single-hop link resources after orchestration.
# Update path resources after orchestration.
def update_single_links(conn, changed_edges):
    """Update bw_resource in link_single_table using changed_edges: {(u, v): new_bw}."""
    sql = """
        UPDATE link_single_table
        SET bw_resource = %s
        WHERE source_node = %s AND destination_node = %s
    """

    for (u, v), bw in changed_edges.items():
        db_execute(conn, sql, (bw, u, v))   # Update (u, v).
        # For undirected links, also update (v, u).


    print("✨ 单跳链路 bw_resource 更新完成！")



if __name__ == "__main__":
    # Connect using the shared database.ini in the project root.
    conn = connect_database()

    # Open the saved network model.
    with open("p_net_directed.pkl", "rb") as f:
        G = pickle.load(f)
    link_bw = load_link_resources(conn)
    # Specify source and destination addresses.
    source_node=15
    destinantion_node=11
    all_p={}
    # Load all network paths, currently limited to six hops.
    all_p=find_all_directed_paths(G, link_bw,cutoff=6)
    print("节点之间的路径为", all_p)
    # Initialize the database with current network parameters.
    insert_paths_into_db(conn, all_p)
    # Test bandwidth updates during orchestration.
    changed_edges = {
        (0, 9): 40,
        (1, 6): 60,
    }
    # Update overall path resources after one orchestration operation.
    #update_path_bw_fast(conn,changed_edges)-                            ,..,
    #update_path_bw(conn, changed_edges)
    # Update single-hop link resources.
    #update_single_links(conn,changed_edges)
    # print("Paths between nodes:", all_p)




