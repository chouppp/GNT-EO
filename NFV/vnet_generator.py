from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]

import networkx as nx
import random
import os

# ======================================================
# Generate a connected Waxman graph.
# ======================================================
def generate_connected_waxman(num_nodes, wm_alpha=0.5, wm_beta=0.5,
                              max_attempts=100, domain=(0, 0, 25, 25), seed=None):
    for i in range(max_attempts):
        G = nx.waxman_graph(
            n=num_nodes,
            alpha=wm_alpha,
            beta=wm_beta,
            domain=domain,
            seed=None if seed is None else seed + i
        )
        if nx.is_connected(G):
            print(f"✅ 成功生成连通图: 尝试 {i+1} 次, 节点={G.number_of_nodes()}, 边={G.number_of_edges()}")
            return G
    raise RuntimeError(f"❌ 尝试 {max_attempts} 次后无法生成连通图，请调整 Waxman 模型参数！")

def generate_connected_directed_waxman_vnr(
    num_nodes,
    wm_alpha=0.5,
    wm_beta=0.5,
    max_attempts=100,
    domain=(0, 0, 25, 25),
    seed=None,
    extra_directed_ratio=0.05   # Virtual networks are generally less directional than physical networks.
):

    rng = random.Random(seed)

    for i in range(max_attempts):
        # Step 1: undirected Waxman graph.
        G_undirected = nx.waxman_graph(
            n=num_nodes,
            alpha=wm_alpha,
            beta=wm_beta,
            domain=domain,
            seed=None if seed is None else seed + i
        )

        if not nx.is_connected(G_undirected):
            continue

        # Step 2: convert to bidirectional edges.
        DG = nx.DiGraph()
        DG.add_nodes_from(G_undirected.nodes(data=True))

        for u, v in G_undirected.edges():
            DG.add_edge(u, v)
            DG.add_edge(v, u)

        # Step 3: add extra one-way edges.
        base_edges = DG.number_of_edges()
        extra_edges = int(base_edges * extra_directed_ratio)

        candidate_pairs = [
            (u, v) for u in DG.nodes for v in DG.nodes
            if u != v and not DG.has_edge(u, v)
        ]
        rng.shuffle(candidate_pairs)

        for u, v in candidate_pairs[:extra_edges]:
            DG.add_edge(u, v)

        # Step 4: check strong connectivity.
        if nx.is_strongly_connected(DG):
            print(
                f"✅ 成功生成强连通有向 VNR: "
                f"节点={DG.number_of_nodes()}, 边={DG.number_of_edges()}"
            )
            return DG

    raise RuntimeError(
        f"❌ 尝试 {max_attempts} 次后仍未生成强连通有向虚拟网络，"
        f"请调整 alpha / beta 或 extra_directed_ratio"
    )


# ======================================================
# Write the YAML file.
# ======================================================
def write_vnet_yaml(G, save_path):
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("graph [\n")

            f.write("  topology [\n")
            f.write('    type "waxman"\n')
            f.write("  ]\n")
            f.write(f"  num_nodes {G.number_of_nodes()}\n")
            f.write(f"  num_edges {G.number_of_edges()}\n\n")

            # Nodes.
            for node_id, attr in G.nodes(data=True):
                pos_x, pos_y = attr["pos"]
                cpu = random.randint(0, 20)

                f.write("  node [\n")
                f.write(f"    id {node_id}\n")
                f.write(f"    label \"{node_id}\"\n")
                f.write(f"    pos {pos_x:.6f}\n")
                f.write(f"    pos {pos_y:.6f}\n")
                f.write(f"    cpu {cpu}\n")
                f.write(f"    max_cpu {cpu}\n")
                f.write("  ]\n")

            f.write("\n")

            # Links.
            for u, v in G.edges():
                bw = random.randint(0, 50)

                f.write("  edge [\n")
                f.write(f"    source {u}\n")
                f.write(f"    target {v}\n")
                f.write(f"    bw {bw}\n")
                f.write(f"    max_bw {bw}\n")
                f.write("  ]\n")

            f.write("]\n")

        print(f"📁 文件保存成功：{save_path}")
        return True

    except Exception as e:
        print(f"❌ 写入失败：{e}")
        return False


# ======================================================
# Main function: generate a virtual network.
# ======================================================
if __name__ == "__main__":
    save_dir = str(_PROJECT_ROOT / 'NFV/vnet_dataset')
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(3520, 4000):
        print("\n==============================")
        print(f"🚀 正在生成虚拟网络 {idx}/50")
        print("==============================")

        # Choose a random node count for each run.
        num_nodes = random.randint(2, 10)
        print(f"🌐 本次 VNet 节点数量: {num_nodes}")

        # Generate a connected Waxman virtual network.
        G = generate_connected_directed_waxman_vnr(num_nodes=num_nodes)

        # Filenames: vinre_node_1.yaml through vinre_node_50.yaml.
        file_path = os.path.join(save_dir, f"vinre0.5_node_{idx}.yaml")

        # Write output.
        write_vnet_yaml(G, file_path)