import networkx as nx
import random
import pickle

import networkx as nx
import pickle
import random

def generate_connected_directed_waxman(
        num_nodes=50,
        wm_alpha=0.5,
        wm_beta=1,
        max_attempts=100,
        domain=(0, 0, 25, 25),
        seed=None,
        extra_directed_ratio=0   # Proportion of extra random one-way edges.
    ):
    """Generate a strongly connected directed Waxman graph.
        1. Generate an undirected Waxman graph.
        2. Convert it to a bidirectional DiGraph.
        3. Add random one-way edges to increase directionality.
        4. Ensure strong connectivity."""
    for i in range(max_attempts):
        # Step 1: generate an undirected Waxman graph.
        G_undirected = nx.waxman_graph(
            n=num_nodes,
            alpha=wm_alpha,
            beta=wm_beta,
            domain=domain,
            seed=None if seed is None else seed + i
        )

        # The graph must be connected before it can become strongly connected.
        if not nx.is_connected(G_undirected):
            continue

        # Step 2: convert to a bidirectional directed graph.
        DG = nx.DiGraph()
        DG.add_nodes_from(G_undirected.nodes(data=True))

        # Add both edge directions.
        for u, v in G_undirected.edges():
            DG.add_edge(u, v)
            DG.add_edge(v, u)

        # Step 3: add extra random one-way edges.
        all_pairs = [(u, v) for u in DG.nodes for v in DG.nodes if u != v]
        random.shuffle(all_pairs)

        extra_edges = int(len(all_pairs) * extra_directed_ratio)

        for (u, v) in all_pairs[:extra_edges]:
            if not DG.has_edge(u, v):
                DG.add_edge(u, v)

        # Step 4: check strong connectivity.
        if nx.is_strongly_connected(DG):
            print(f"✅ 第 {i+1} 次生成成功（强连通）: 节点={DG.number_of_nodes()}, 边={DG.number_of_edges()}")
            with open("p_net_directed.pkl", "wb") as f:
                pickle.dump(DG, f)
            return DG

    raise RuntimeError(f"❌ {max_attempts} 次尝试后仍未生成强连通 Waxman 有向图，请增大 alpha/beta 或 extra_directed_ratio")
def write_graph_to_yaml(G, save_path="p_net.yaml"):
    """Write a Virne-style YAML file with node and edge attributes.
        Nodes: id, label, pos(x,y), cpu, max_cpu.
        Edges: source, target, bw, max_bw."""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write('graph [\n')

        # -------------------------------
        # Set node attributes.
        # -------------------------------
        f.write('  node_attrs_setting [\n')
        f.write('    name "cpu"\n')
        f.write('    distribution "uniform"\n')
        f.write('    dtype "int"\n')
        f.write('    generative 1\n')
        f.write('    high 100\n')
        f.write('    low 50\n')
        f.write('    owner "node"\n')
        f.write('    type "resource"\n')
        f.write('  ]\n')
        f.write('  node_attrs_setting [\n')
        f.write('    name "max_cpu"\n')
        f.write('    originator "cpu"\n')
        f.write('    owner "node"\n')
        f.write('    type "extrema"\n')
        f.write('  ]\n\n')

        # -------------------------------
        # Set link attributes.
        # -------------------------------
        f.write('  link_attrs_setting [\n')
        f.write('    name "bw"\n')
        f.write('    distribution "uniform"\n')
        f.write('    dtype "int"\n')
        f.write('    generative 1\n')
        f.write('    high 100\n')
        f.write('    low 50\n')
        f.write('    owner "link"\n')
        f.write('    type "resource"\n')
        f.write('  ]\n')
        f.write('  link_attrs_setting [\n')
        f.write('    name "max_bw"\n')
        f.write('    originator "bw"\n')
        f.write('    owner "link"\n')
        f.write('    type "extrema"\n')
        f.write('  ]\n\n')

        # -------------------------------
        # Basic topology information.
        # -------------------------------
        f.write('  topology [\n')
        f.write('    type "waxman"\n')
        f.write('  ]\n')
        f.write(f'  num_nodes {G.number_of_nodes()}\n')
        f.write(f'  num_edges {G.number_of_edges()}\n\n')

        # -------------------------------
        # Write node information.
        # -------------------------------
        for node_id, attr in G.nodes(data=True):
            pos_x, pos_y = attr["pos"]
            cpu = random.randint(50, 100)
            f.write('  node [\n')
            f.write(f'    id {node_id}\n')
            f.write(f'    label "{node_id}"\n')
            f.write(f'    pos {pos_x:.6f}\n')
            f.write(f'    pos {pos_y:.6f}\n')
            f.write(f'    cpu {cpu}\n')
            f.write(f'    max_cpu {cpu}\n')
            f.write('  ]\n')

        f.write('\n')

        # -------------------------------
        # Write edge information.
        # -------------------------------
        for u, v in G.edges():
            bw = random.randint(50, 100)
            f.write('  edge [\n')
            f.write(f'    source {u}\n')
            f.write(f'    target {v}\n')
            f.write(f'    bw {bw}\n')
            f.write(f'    max_bw {bw}\n')
            f.write('  ]\n')

        f.write(']\n')

    print(f"✅ 已生成并保存拓扑文件到：{save_path}")

# ------------------------------------
# Main function.
# ------------------------------------
if __name__ == "__main__":
    G = generate_connected_directed_waxman(num_nodes=25, wm_alpha=0.5, wm_beta=1)
    write_graph_to_yaml(G, "p_net.yaml")