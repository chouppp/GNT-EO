import numpy as np
from typing import List, Dict,Optional


class TrafficMatrixCalculator:
    """Traffic matrix calculator for a controller domain."""
    def __init__(self, conn):
        self.conn = conn

    def compute_tm(
        self,
        domain_nodes: List[int],
        directed: bool = True
    ) -> Dict[str, np.ndarray]:
        """Calculate the traffic matrix within a controller domain.
        
                Parameters
                ----------
                domain_nodes: List[int], the domain's node set M.
                directed: bool, whether the traffic matrix distinguishes directions.
        
                Returns
                -------
                result: {'nodes': domain_nodes, 'TM': a |M|-by-|M| NumPy array}."""
        node_index = {node: idx for idx, node in enumerate(domain_nodes)}
        size = len(domain_nodes)

        TM = np.zeros((size, size), dtype=float)

        sql = """
        SELECT source_node, destination_node, current_flow
        FROM link_single_table
        WHERE source_node IN %s
          AND destination_node IN %s
        """

        with self.conn.cursor() as cursor:
            cursor.execute(sql, (tuple(domain_nodes), tuple(domain_nodes)))
            rows = cursor.fetchall()

        for src, dst, flow in rows:
            if src in node_index and dst in node_index:
                i = node_index[src]
                j = node_index[dst]
                TM[i, j] += float(flow)

                # For an undirected traffic matrix.
                #if not directed:
                   # TM[j, i] += float(flow)

        return {
            "nodes": domain_nodes,
            "TM": TM
        }

    @staticmethod
    def compute_network_entropy(TM: np.ndarray, log_base: float = np.e) -> float:
        TM = np.asarray(TM, dtype=float)
        total = TM.sum()
        if total <= 0:
            return 0.0

        P = TM / total
        mask = P > 0  # Sum only terms with p_ij > 0.
        if log_base == np.e:
            return float(-(P[mask] * np.log(P[mask])).sum())
        else:
            return float(-(P[mask] * (np.log(P[mask]) / np.log(log_base))).sum())

    # 3. Calculate total intra-domain traffic F.
    @staticmethod
    def compute_total_flow(TM: np.ndarray) -> float:
        return float(np.sum(TM))

    # 4. Calculate network temperature T = delta F / delta H.
    @staticmethod
    def compute_temperature(
            F_prev: float,
            H_prev: float,
            F_curr: float,
            H_curr: float,
            eps: float = 1e-12
    ) -> Optional[float]:
        dF = F_curr - F_prev
        dH = H_curr - H_prev

        return dF / (dH + eps)

    # Return network temperature for each domain.
    def compute_domain_temperature(
            self,
            domain_nodes: List[int],
            prev_tm: Optional[np.ndarray] = None
    ) -> Dict:
        tm_result = self.compute_tm(domain_nodes)
        TM = tm_result["TM"]

        F_curr = self.compute_total_flow(TM)
        H_curr = self.compute_network_entropy(TM)

        if prev_tm is None:
            return {
                "TM": TM,
                "F": F_curr,
                "H": H_curr,
                "T": None
            }

        F_prev = self.compute_total_flow(prev_tm)
        H_prev = self.compute_network_entropy(prev_tm)

        T = self.compute_temperature(F_prev, H_prev, F_curr, H_curr)

        return {
            "TM": TM,
            "F": F_curr,
            "H": H_curr,
            "T": T
        }