import pickle
import networkx as nx
import pymysql
if __package__:
    from .db_config import connect_database
else:
    from db_config import connect_database
import math
import re
# Read the current substrate network state from the database (updated).
class Network_node_atrribute:
    def __init__(self, conn=None, G=None):
        """conn: MySQL connection object.
                G: NetworkX graph."""
        self.conn = conn
        self.G = G
    # -------------------------------
    # 1. Read link bandwidth from the database.
    # -------------------------------
    def load_edges_from_db(self):

        if self.conn is None:
            raise ValueError("未提供数据库连接 conn")

        edges = {}
        cur = self.conn.cursor(pymysql.cursors.DictCursor)

        sql = "SELECT source_node, destination_node, bw_resource FROM link_single_table"
        cur.execute(sql)
        rows = cur.fetchall()

        for row in rows:
            u = int(row["source_node"])
            v = int(row["destination_node"])
            bw = float(row["bw_resource"])
            edges[(u, v)] = bw

        cur.close()
        return edges


    # -------------------------------
    # 2. Node degree.
    # -------------------------------
    def calc_node_degree(self, m):
        return self.G.degree(m)


    # -------------------------------
    # 3. Strength of incident links.
    # -------------------------------
    def calc_node_adjacent_bw(self, m, edges):
        total_bw = 0
        for (u, v), bw in edges.items():
            if u == m:
                total_bw += bw
        return total_bw


    # -------------------------------
    # 4. Euclidean distance.
    # -------------------------------
    def euclidean_distance_xy(self, m, n):
        xm, ym = self.G.nodes[m]["pos"][0], self.G.nodes[m]["pos"][1]
        xn, yn = self.G.nodes[n]["pos"][0], self.G.nodes[n]["pos"][1]
        return math.sqrt((xm - xn)**2 + (ym - yn)**2)


    # -------------------------------
    # 5. Sum of distances from node m to all nodes (centrality).
    # -------------------------------
    def total_distance_from_node(self, m):
        total_dist = 0.0
        for n in self.G.nodes():
            if n != m:
                total_dist += self.euclidean_distance_xy(m, n)
        return total_dist


    # -------------------------------
    # 6. Link interference coefficient LI between nodes m and n.
    # -------------------------------
    def LI_between_node(self, m, n):
        far_m = self.total_distance_from_node(m)
        d_m = self.calc_node_degree(m)

        far_n = self.total_distance_from_node(n)
        d_n = self.calc_node_degree(n)

        return far_m / d_m + far_n / d_n


    # -------------------------------
    # 7. Sum of LI between m and all nodes.
    # -------------------------------
    def LI_sum_for_node(self, m):
        total_LI = 0
        for n in self.G.nodes():
            if n != m:
                total_LI += self.LI_between_node(m, n)
        return total_LI


    # -------------------------------
    # 8. RB node interference score.
    # -------------------------------
    def RB_node(self, m, Cm, edges):
        c_m = Cm
        s_m = self.calc_node_adjacent_bw(m, edges)
        li_mn = self.LI_sum_for_node(m)
        return c_m * s_m * li_mn

    import networkx as nx

    def NOV_mn(self, m, n,Cm,Cn,edges):
        RB_m = self.RB_node(m,Cm,edges)
        RB_n = self.RB_node(n, Cn, edges)
        dis = self.euclidean_distance_xy(m, n)
        if dis == 0:
            dis = 0.0001

        # Directed shortest path.
        try:
            hop = nx.shortest_path_length(self.G, m, n)
        except nx.NetworkXNoPath:
            hop = 9999  # Assign a very large value when unreachable.

        if hop == 0:
            hop = 1

        NOV = (RB_m*RB_n)/(dis**2)*(hop**2)
        return NOV








if __name__ == "__main__":
    # Connect using the shared database.ini in the project root.
    conn = connect_database()

    # Open the saved network model.
    with open("p_net_directed.pkl", "rb") as f:
        G = pickle.load(f)

    # Output the degree of one node.
    #print_node_degree(G,0)
    # Read link resources from the database.
    #edges = load_edges_from_db(conn)
    # print("Number of parsed edges:", len(edges))

    # Calculate one node's strength.
    #m = 0
    #bw_sum = calc_node_adjacent_bw(m, edges)
    # print(f"Total incident-link bandwidth for node {m} = {bw_sum}")

    # Calculate the distance between two nodes.
    #distance=euclidean_distance_xy(G,1,4)
    # print(f"Distance between nodes 1 and 4 = {distance}")
    # Calculate the degree of node m.
    #node_m_total_distance=total_distance_from_node(G,1)
    # print(f"Centrality of node 1 = {node_m_total_distance}")

    # Calculate the node's LI coefficient.

    #li=LI_between_node(G,0,1)




