import re
import pymysql
if __package__:
    from .db_config import connect_database
else:
    from db_config import connect_database

# -------------------------
# 1. Read the custom YAML file.
# -------------------------

def load_edges_from_custom_yaml(yaml_file):
    edges = []
    with open(yaml_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Match link blocks; link_attrs may be named link or edge.
    # Assume each link block has the following format.
    # link [
    #   source 0
    #   target 3
    #   bw 96
    #   max_bw 96
    # ]
    edge_blocks = re.findall(r'edge\s*\[(.*?)\]', content, re.DOTALL)

    for block in edge_blocks:
        edge = {}
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            # Separate keys and values by whitespace.
            parts = line.split()
            if len(parts) >= 2:
                key, value = parts[0], " ".join(parts[1:])
                # Try converting to int or float.
                if value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except:
                        pass  # Keep the value as a string.
                edge[key] = value
        edges.append(edge)

    return edges


# -------------------------
# 2. Insert into MySQL.
# -------------------------

def insert_edges_into_db(edges):
    # Configure the MySQL connection in database.ini.
    conn = connect_database()

    cursor = conn.cursor()

    sql = """
        INSERT INTO link_single_table 
        (source_node, destination_node, bw_resource, bw_max_resource)
        VALUES (%s, %s, %s, %s)
    """

    for e in edges:
        cursor.execute(sql, (
            e["source"],
            e["target"],
            e["max_bw"],   # Initialize bw_resource to max_bw.
            e["max_bw"]    # bw_max_resource = max_bw
        ))

    conn.commit()
    cursor.close()
    conn.close()
    print("所有链路已经成功写入数据库！")

# Parse nodes from the file.
def load_nodes_from_file(filepath):
    nodes = []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current = None
    reading_node = False
    pos_count = 0   # There are two pos lines.

    for line in lines:
        line = line.strip()

        if line.startswith("node ["):
            current = {}
            reading_node = True
            pos_count = 0
            continue

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

            elif key == "pos":
                if pos_count == 0:
                    current["pos_x"] = float(val)
                else:
                    current["pos_y"] = float(val)
                pos_count += 1

            elif key == "cpu":
                current["cpu"] = int(val)

            elif key == "max_cpu":
                current["max_cpu"] = int(val)

    return nodes
# Insert into the database.
def insert_nodes_into_db(nodes):
    conn = connect_database()

    cursor = conn.cursor()

    sql = """
        INSERT INTO node_single_table
        (node_id, pos_x, pos_y, cpu, max_cpu)
        VALUES (%s, %s, %s, %s, %s)
    """

    for n in nodes:
        cursor.execute(sql, (
            n["id"],
            n["pos_x"],
            n["pos_y"],
            n["cpu"],
            n["max_cpu"]
        ))

    conn.commit()
    cursor.close()
    conn.close()

    print("所有节点数据已成功写入 node_single_table！")
# -------------------------
# 3. Run the main program.
# -------------------------

if __name__ == "__main__":
    edges = load_edges_from_custom_yaml("p_net.yaml")
    print("读取到边数量:", len(edges))
    insert_edges_into_db(edges)
    # nodes = load_nodes_from_file("p_net.yaml")  # Parse nodes.
    # insert_nodes_into_db(nodes)  # Write to the database.