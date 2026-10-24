import pickle
import networkx as nx
import pymysql
import math
import re
from copy import deepcopy
# First perform preliminary K-shell decomposition.
def k_shell_decomposition(G):
    H = deepcopy(G)
    ks = {node: 0 for node in H.nodes()}
    k = 1

    while H.number_of_nodes() > 0:
        changed = True
        while changed:
            changed = False
            # Find all nodes with degree <= k.
            to_remove = [n for n, deg in H.degree() if deg <= k]
            if to_remove:
                changed = True
                for n in to_remove:
                    ks[n] = k
                    H.remove_node(n)

        # Increment the shell index after removing all nodes with degree <= k.
        k += 1

    return ks


def salton_index(G, i, j):
    """Calculate the Salton index between nodes i and j."""
    # Neighbor set.
    neighbors_i = set(G.neighbors(i))
    neighbors_j = set(G.neighbors(j))

    # Number of common neighbors.
    intersection_count = len(neighbors_i & neighbors_j)

    # Degree.
    ki = len(neighbors_i)
    kj = len(neighbors_j)

    if ki == 0 or kj == 0:
        return 0.0  # Avoid division by zero.

    return intersection_count / math.sqrt(ki * kj)

def salton_sum_for_node(G, v):
    """Sum the Salton indices between v and its neighbors: S(v) = sum(Salton(v, u) for u in neighbors(v))."""
    total = 0.0
    for u in G.neighbors(v):
        total += salton_index(G, v, u)
    return total

# Compute node_gravity for every node.
def node_gravity(G):
    BC = nx.betweenness_centrality(G, normalized=True)
    CC = nx.closeness_centrality(G)

    gravity = {}

    for i in G.nodes():
        total = 0
        for j in G.neighbors(i):
            total += BC[i] * BC[j] / (CC[i] ** 2)
        gravity[i] = total
    return gravity


def node_p_attribute(G, w):
    # Gravity centrality.
    node_gra = node_gravity(G)

    # K-shell value.
    ks = k_shell_decomposition(G)

    node_p = {}

    # Compute the P attribute for every node.
    for i in G.nodes():
        node_p[i] = w * ks[i] + (2 - w) * node_gra[i] * (salton_sum_for_node(G,i))
    return node_p

import networkx as nx
# Calculate mean path length LG.
def average_path_length_formula(G):
    nodes = list(G.nodes())
    N = len(nodes)

    # Calculate all shortest-path distances.
    # shortest_path_length returns {target: distance}.
    dist = dict(nx.all_pairs_shortest_path_length(G))

    total = 0
    count = N * (N - 1) / 2  # Combination count C(N, 2).

    for i in range(N):
        for j in range(i + 1, N):
            total += dist[nodes[i]][nodes[j]]

    # Substitute into the formula.
    L = total / count
    return L

def compute_NG(G):

    num_nodes = G.number_of_nodes()

    # Mean degree d_avg(G) = 2 * |E| / |V|.
    d_avg = sum(dict(G.degree()).values()) / num_nodes

    # Mean path length L(G).
    Lg = average_path_length_formula(G)

    # Formula N(G).
    NG = num_nodes * d_avg * Lg

    return NG

def partition_balanced(G, controller_nodes, max_size=5):
    """Partition the network subject to controller capacity: at most max_size nodes per controller.
        Return region: {controller: [nodes]} and assigned: {node: controller}."""

    # All-pairs shortest-path distances.
    all_dist = dict(nx.all_pairs_shortest_path_length(G))

    # Initialize.
    region = {c: [] for c in controller_nodes}
    assigned = {}

    # Globally sort nodes by distance to their nearest controller.
    nodes_sorted = sorted(
        G.nodes(),
        key=lambda v: min(all_dist[c].get(v, float("inf")) for c in controller_nodes)
    )

    for v in nodes_sorted:
        # Assign controller nodes to themselves.
        if v in controller_nodes:
            region[v].append(v)
            assigned[v] = v
            continue

        # Sort controllers by increasing distance.
        dist_list = sorted(
            [(c, all_dist[c].get(v, float("inf"))) for c in controller_nodes],
            key=lambda x: x[1]
        )

        # Find the first controller with spare capacity.
        placed = False
        for c, d in dist_list:
            if len(region[c]) < max_size:
                region[c].append(v)
                assigned[v] = c
                placed = True
                break

        # If all controllers are full, force assignment to the nearest one.
        if not placed:
            c, _ = dist_list[0]
            region[c].append(v)
            assigned[v] = c

    return region, assigned



if __name__ == "__main__":

    # Open the saved network model.
    with open("../p_net_directed.pkl", "rb") as f:
        G = pickle.load(f)
    # This formula assumes an undirected graph; applying it to directed graphs is problematic.
    G = G.to_undirected()
    # Calculate node K-shell values.
    ks=k_shell_decomposition(G)
    ks_nx = nx.core_number(G)
    print(f"节点19的K-shell值为 = {ks[0]}")
    print(f"节点18的K-shell值为 = {ks_nx[0]}")
    # Calculate node Salton scores.
    node_salton_socre=salton_sum_for_node(G,5)
    print(f"节点1的salton值为 = {node_salton_socre}")

    node_gra=node_gravity(G)
    print(f"节点1的graviaty值为 = {node_gra[0]}")

    p=node_p_attribute(G,0.5)
    print(f"P值为 = {p}")
    sorted_nodes = sorted(p.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_nodes[:5]
    print("前 5 个节点及其 P 值：")
    for node, value in top_5:
        print(f"节点 {node} 的 P 值 = {value}")

    NG=compute_NG(G)
    print(f"NG值为 = {NG}")

    core_nodes = [7, 18, 20, 12, 11]  # Example.

    # Partition the network into regions.
    region, assigned = partition_balanced(G, core_nodes, max_size=5)

    print("各控制域划分结果：")
    for c, nodes in region.items():
        print(f"控制器 {c}: 管理节点 {nodes}")
