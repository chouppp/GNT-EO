from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]

from NFV.NORD.NORD_function import NORDRanker
import pickle
import pymysql
from NFV.db_config import connect_database

from NFV.virtual_net_atttribute import Network_node_atrribute
from NFV.util.normal_method import load_vnet_from_file
if __name__ == "__main__":

    # =========================
    # 1. Database connection.
    # =========================
    conn = connect_database()

    print("✔ 数据库连接成功")

    # =========================
    # 2. Read the substrate network topology.
    # =========================
    with open(str(_PROJECT_ROOT / 'NFV/p_net_directed.pkl'), "rb") as f:
        G = pickle.load(f)

    print("✔ 底层网络读取完成")
    print("底层节点数量:", len(G.nodes()))
    print("底层链路数量:", len(G.edges()))

    # =========================
    # 3. Read residual substrate-link bandwidth.
    # =========================
    node_function = Network_node_atrribute(conn, G)

    substrate_edges = node_function.load_edges_from_db()

    print("✔ 底层链路资源读取完成")
    print("数据库链路数量:", len(substrate_edges))

    # =========================
    # 4. Read the first virtual network.
    # =========================
    vnr_id = 1

    vnet_path = (
        str(_PROJECT_ROOT / f'NFV/vnet_dataset/vinre0.5_node_{vnr_id}.yaml')
    )

    virtual_nodes, virtual_edges = load_vnet_from_file(vnet_path)

    print("✔ 虚拟网络读取完成")
    print("VNR ID:", vnr_id)
    print("虚拟节点数量:", len(virtual_nodes))
    print("虚拟链路数量:", len(virtual_edges))

    # =========================
    # 5. Initialize the NORD attribute calculator.
    # =========================
    nord_attr = NORDRanker(
        conn,
        G
    )

    # =========================
    # 6. Calculate five attributes for virtual nodes.
    # =========================
    virtual_attr_matrix = nord_attr.build_virtual_decision_matrix(
        virtual_nodes=virtual_nodes,
        virtual_edges=virtual_edges
    )

    print("\n========== 虚拟节点属性矩阵 ==========")

    for node_id, attrs in virtual_attr_matrix.items():
        print(
            f"v{node_id}: "
            f"CRB={attrs['crb_strength']:.4f}, "
            f"BW={attrs['bw_strength']:.4f}, "
            f"Degree={attrs['degree']:.4f}, "
            f"Betweenness={attrs['betweenness']:.4f}, "
            f"Eigen={attrs['eigen']:.4f}"
        )

    # =========================
    # 7. Calculate five attributes for substrate nodes.
    # =========================
    substrate_attr_matrix = nord_attr.build_substrate_decision_matrix()

    print("\n========== 底层节点属性矩阵 ==========")

    for node_id, attrs in substrate_attr_matrix.items():
        print(
            f"s{node_id}: "
            f"CRB={attrs['crb_strength']:.4f}, "
            f"BW={attrs['bw_strength']:.4f}, "
            f"Degree={attrs['degree']:.4f}, "
            f"Betweenness={attrs['betweenness']:.4f}, "
            f"Eigen={attrs['eigen']:.4f}"
        )

    conn.close()
    print("\n✔ NORD 属性计算 Demo 运行完成")