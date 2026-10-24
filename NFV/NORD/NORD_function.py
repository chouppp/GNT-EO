import networkx as nx
import math
from NFV.util.normal_method import (
    load_vnet_from_file,
    load_substrate_nodes_from_db,
    load_all_paths_numpy
)
from NFV.virtual_net_atttribute import Network_node_atrribute

class NORDRanker:

    def __init__(self, conn, G):
        self.conn = conn
        self.G = G
        self.G_undirected = G.to_undirected()
        self.node_function = Network_node_atrribute(conn, G)

    # ---------- Topology ----------
    def build_virtual_graph(self, virtual_nodes, virtual_edges):
        """Construct a NetworkX virtual graph from virtual nodes and links."""

        import networkx as nx

        Gv = nx.Graph()

        # Add virtual nodes.
        for v in virtual_nodes:
            Gv.add_node(int(v["id"]))

        # Add virtual links.
        for e in virtual_edges:
            src = int(e["source"])
            dst = int(e["target"])
            bw = float(e["bw"])

            Gv.add_edge(
                src,
                dst,
                bw=bw
            )

        return Gv

    def find_all_paths(self, network, start, end, path=None):
        if path is None:
            path = []

        path = path + [start]

        if start == end:
            return [path]

        if start not in network:
            return []

        paths = []

        for node in network[start]:
            if node not in path:
                newpaths = self.find_all_paths(network, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)

        return paths

    def get_shortest_path_dist(self, paths):
        if not paths:
            return []

        paths.sort(key=len)
        shortest_len = len(paths[0])

        return [
            path for path in paths
            if len(path) == shortest_len
        ]

    def betweenness_centrality_node(self, network, node):
        vertices = list(network.keys())
        node_pair_list = []
        bc_sum_node = 0.0

        for start_node in vertices:
            for end_node in vertices:
                node_pair = [start_node, end_node]
                node_pair.sort()

                if (
                        start_node != end_node
                        and start_node != node
                        and end_node != node
                        and node_pair not in node_pair_list
                ):
                    all_paths = self.find_all_paths(
                        network,
                        start_node,
                        end_node,
                        path=[]
                    )

                    shortest_paths = self.get_shortest_path_dist(all_paths)

                    path_count = 0
                    bc_node_in_path = 0

                    for sp_path in shortest_paths:
                        if node in sp_path:
                            bc_node_in_path += 1
                        path_count += 1

                    if path_count != 0:
                        bc_sum_node += float(bc_node_in_path) / float(path_count)

                    node_pair_list.append(node_pair)

        return bc_sum_node
    def degree_centrality(self, graph):
        return {
            int(node): graph.degree(node)
            for node in graph.nodes()
        }

    def betweenness_centrality(self, graph):
        """Compute raw betweenness centrality using NetworkX, then apply L2 normalization as in NORD."""
        print("开始介数归一化 ")
        G = graph.to_undirected()

        raw_bc = nx.betweenness_centrality(
            G,
            normalized=False,
            weight=None
        )

        total_n_value = math.sqrt(
            sum(value ** 2 for value in raw_bc.values())
        )

        return {
            int(node): (
                round(float(value) / float(total_n_value), 3)
                if total_n_value != 0
                else 0.0
            )
            for node, value in raw_bc.items()
        }

    def eigen_centrality(self, graph, max_iter=3):
        G = graph.to_undirected()

        nodes = sorted([int(n) for n in G.nodes()])
        index = {
            node: idx
            for idx, node in enumerate(nodes)
        }

        n = len(nodes)

        if n == 0:
            raise Exception("cannot compute centrality for the null graph")

        # Construct the adjacency matrix.
        matrix = [
            [0 for _ in range(n)]
            for _ in range(n)
        ]

        for u in nodes:
            for v in G.neighbors(u):
                matrix[index[u]][index[int(v)]] = 1

        # Initial vector [1, 1, ..., 1]^T.
        vector = [
            [1.0]
            for _ in range(n)
        ]

        def matrix_multiplication(m1, m2):
            return [
                [
                    sum(x * y for x, y in zip(m1_r, m2_c))
                    for m2_c in zip(*m2)
                ]
                for m1_r in m1
            ]

        def normalize(vec):
            norm = math.sqrt(
                sum(value[0] ** 2 for value in vec)
            )

            if norm == 0:
                return [
                    [0.0]
                    for _ in vec
                ]

            return [
                [round(float(value[0]) / float(norm), 3)]
                for value in vec
            ]

        # Corresponds to get_normalized_eigen_matrix() in the original source.
        vector = normalize(
            matrix_multiplication(matrix, vector)
        )

        # Corresponds to eigenvector_centrality(max_iter=3) in the original source.
        for _ in range(max_iter):
            eigen_matrix = matrix_multiplication(matrix, vector)
            vector = normalize(eigen_matrix)

        return {
            node: float(vector[index[node]][0])
            for node in nodes
        }

    # ---------- Resource ----------
    def virtual_crb_strength(self, virtual_nodes):
        return {
            int(v["id"]): float(v["cpu"])
            for v in virtual_nodes
        }

    def substrate_crb_strength(self):
        s_nodes = load_substrate_nodes_from_db(self.conn)

        return {
            int(node_id): float(info["cpu"])
            for node_id, info in s_nodes.items()
        }

    def virtual_bw_strength(
            self,
            virtual_nodes,
            virtual_edges
    ):
        result = {
            int(v["id"]): 0.0
            for v in virtual_nodes
        }

        for edge in virtual_edges:
            u = int(edge["source"])
            v = int(edge["target"])
            bw = float(edge["bw"])

            result[u] += bw
            result[v] += bw

        return result

    def substrate_bw_strength(self):

        edges = self.node_function.load_edges_from_db()

        strength = {}

        for (u, v), bw in edges.items():
            strength.setdefault(u, 0.0)
            strength.setdefault(v, 0.0)

            strength[u] += float(bw)
            strength[v] += float(bw)

        return strength

    # ---------- Decision Matrix ----------
    def build_virtual_decision_matrix(
            self,
            virtual_nodes,
            virtual_edges
    ):

        Gv = self.build_virtual_graph(
            virtual_nodes,
            virtual_edges
        )

        crb = self.virtual_crb_strength(virtual_nodes)

        bw = self.virtual_bw_strength(
            virtual_nodes,
            virtual_edges
        )

        degree = self.degree_centrality(Gv)

        betweenness = self.betweenness_centrality(Gv)

        eigen = self.eigen_centrality(Gv)

        decision_matrix = {}

        for node in Gv.nodes():
            decision_matrix[int(node)] = {
                "crb_strength": crb[node],
                "bw_strength": bw[node],
                "degree": degree[node],
                "betweenness": betweenness[node],
                "eigen": eigen[node]
            }

        return decision_matrix

    def build_substrate_decision_matrix(self):

        crb = self.substrate_crb_strength()

        bw = self.substrate_bw_strength()

        degree = self.degree_centrality(self.G_undirected)

        betweenness = self.betweenness_centrality(self.G_undirected)

        eigen = self.eigen_centrality(self.G_undirected)

        decision_matrix = {}

        for node in self.G.nodes():
            decision_matrix[int(node)] = {
                "crb_strength": crb[node],
                "bw_strength": bw[node],
                "degree": degree[node],
                "betweenness": betweenness[node],
                "eigen": eigen[node]
            }

        return decision_matrix