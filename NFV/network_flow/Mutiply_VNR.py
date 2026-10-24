import os

import yaml

from NFV.network_flow.network_flow import VNRFlowSimulator
from typing import Dict, Tuple


class MultiVNRFlowSimulator:
    def __init__(
        self,
        total_s_link_result: dict,
        alpha_range=(0.6, 1),
        link_noise=0,
        seed=None
    ):
        """
        Parameters
        ----------
        total_s_link_result : dict
            {vnr_id: {"slink": s_link_result}}
        """
        self.alpha_range = alpha_range
        self.link_noise = link_noise
        self.seed = seed

        self.vnr_simulators = {}

        # Initialize existing VNRs.
        for vnr_id, info in total_s_link_result.items():
            self.vnr_simulators[vnr_id] = VNRFlowSimulator(
                vne_links=info["slink"],
                alpha_range=self.alpha_range,
                link_noise=self.link_noise,
                seed=self.seed
            )

        self.time = 0

    # =====================================================
    # New feature 1: dynamically add a VNR.
    # =====================================================
    def add_vnr(self, vnr_id: int, s_link_result: list, vnr_id2: int):
        """vnr_id: internal event ID / unique key.
                vnr_id2: actual VNR ID."""
        if vnr_id in self.vnr_simulators:
            raise ValueError(f"VNR {vnr_id} already exists")

        self.vnr_simulators[vnr_id] = VNRFlowSimulator(
            vne_links=s_link_result,
            alpha_range=self.alpha_range,
            link_noise=self.link_noise,
            seed=self.seed,
            vnr_id=vnr_id2
        )

        print("添加成功")

    # =====================================================
    # New feature 2: remove a departing VNR.
    # =====================================================
    def remove_vnr(self, vnr_id: int):
        """Remove a VNR at runtime."""
        if vnr_id in self.vnr_simulators:
            del self.vnr_simulators[vnr_id]

    # =====================================================
    # Preserve the original aggregate-traffic generation logic.
    # =====================================================

    def step_generate_flow(self):
        """Generate aggregate traffic from all VNRFlowSimulator instances."""
        self.time += 1

        global_link_load = {}
        vnr_flow_records = []

        for inner_vnr_id, simulator in self.vnr_simulators.items():
            vnr_flow = simulator.step_generate_flow()

            real_vnr_id = simulator.vnr_id

            for (u, v), bw in vnr_flow.items():
                vnr_flow_records.append({
                    "vnr_id": int(real_vnr_id),
                    "source": int(u),
                    "target": int(v),
                    "flow": float(bw)
                })

            for edge, bw in vnr_flow.items():
                global_link_load[edge] = global_link_load.get(edge, 0.0) + bw

        self.last_vnr_flow_records = vnr_flow_records

        return global_link_load

    def append_flow_record_to_yaml(self,record, yaml_path="vnr_flow_history.yaml"):
        """Append traffic-generation records for all time steps."""
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {
                "records": []
            }

        data.setdefault("records", [])
        data["records"].append(record)

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                sort_keys=False,
                allow_unicode=True
            )

    # =====================================================
    # Preserve traffic generation and database writeback.
    # =====================================================
    def step_and_write(
            self,
            conn,
            commit: bool = True,
            update_reverse_if_missing: bool = False,
            current_time: float = None,
            yaml_path: str = "vnr_flow_history.yaml",
            save_yaml: bool = True
    ) -> dict:
        """One time step:
                1. Generate aggregate traffic across VNRs.
                2. Write to the database.
                3. Optionally save YAML history."""
        # 1. Generate aggregate traffic.
        flow = self.step_generate_flow()

        # 2. Write to the database.
        reset_sql = "UPDATE link_single_table SET current_flow = 0"
        update_sql = """
               UPDATE link_single_table
               SET current_flow = %s
               WHERE source_node = %s AND destination_node = %s
           """

        with conn.cursor() as cursor:
            cursor.execute(reset_sql)
            for (u, v), bw in flow.items():
                cursor.execute(update_sql, (float(bw), int(u), int(v)))
                if cursor.rowcount == 0 and update_reverse_if_missing:
                    cursor.execute(update_sql, (float(bw), int(v), int(u)))

        if commit:
            conn.commit()

        # 3. Save YAML.
        if save_yaml and yaml_path is not None:
            record = {
                "time": float(current_time) if current_time is not None else self.time,
                "step": int(self.time),
                "active_vnr_num": len(self.vnr_simulators),
                "vnr_flows": self.last_vnr_flow_records,
                "global_link_flow": [
                    {"source": int(u), "target": int(v), "flow": float(bw)}
                    for (u, v), bw in flow.items()
                ]
            }
            self.append_flow_record_to_yaml(record, yaml_path)

        return flow