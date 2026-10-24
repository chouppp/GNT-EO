import networkx as nx
import pickle

def build_directed_graph_from_p_net_yaml(yaml_path, output_pkl):
    G = nx.DiGraph()

    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # --------------------
        # 1. Parse nodes.
        # --------------------
        if line.startswith("node ["):
            node_id = None
            pos = []

            i += 1
            while not lines[i].strip().startswith("]"):
                parts = lines[i].strip().split()
                if parts[0] == "id":
                    node_id = int(parts[1])
                elif parts[0] == "pos":
                    pos.append(float(parts[1]))
                i += 1

            # Include only attributes natively supported by NetworkX.
            if len(pos) == 2:
                G.add_node(node_id, pos=(pos[0], pos[1]))
            else:
                G.add_node(node_id)

        # --------------------
        # 2. Parse directed edges.
        # --------------------
        elif line.startswith("edge ["):
            src = None
            dst = None

            i += 1
            while not lines[i].strip().startswith("]"):
                parts = lines[i].strip().split()
                if parts[0] == "source":
                    src = int(parts[1])
                elif parts[0] == "target":
                    dst = int(parts[1])
                i += 1

            if src is not None and dst is not None:
                G.add_edge(src, dst)

        i += 1

    # --------------------
    # 3. Save the pickle file.
    # --------------------
    with open(output_pkl, "wb") as f:
        pickle.dump(G, f)

    print("✅ p_net_directed.pkl 构建完成")
    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")
    print(f"是否强连通: {nx.is_strongly_connected(G)}")

    return G

if __name__ == "__main__":
    G = build_directed_graph_from_p_net_yaml("p_net.yaml","p_net_directed.pkl")
