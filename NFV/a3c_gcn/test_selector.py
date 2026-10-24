import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from .a3c_gcn_realtime import apply_resources, event_windows, new_run_directory, run
from . import a3c_gcn_realtime as runtime
from .selector import NodeSelector, map_directed_links
from .training_env import DatabaseSnapshot, EOTrainingEnvironment
from .training_model import DirectionalA3CGCNActor


class SelectorTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weights = self.root / "actor.pt"
        actor = DirectionalA3CGCNActor()
        for p in actor.parameters():
            p.data.zero_()
        torch.save(actor.state_dict(), self.weights)
        self.snapshot = DatabaseSnapshot(
            nodes={i: {"cpu": 10., "max_cpu": 10., "pos": float(i), "pos2": 0.} for i in (2, 9)},
            edges={(2, 9): 10.}, paths={(2, 9): [{"hop": 1, "path": [(2, 9)]}]},
        )
        self.vnodes = [{"id": 5, "cpu": 2.}, {"id": 7, "cpu": 3.}]
        self.vedges = [{"source": 5, "target": 7, "bw": 4.}]

    def test_selector_pure_and_matches_training(self):
        original = copy.deepcopy(self.snapshot)
        selector = NodeSelector(self.weights, self.snapshot, distance_limit=None)
        mapping = selector.select_nodes(1, self.vnodes, self.vedges, self.snapshot.nodes, self.snapshot.edges)
        self.assertEqual(mapping, {5: 2, 7: 9})
        self.assertEqual(self.snapshot, original)
        links = map_directed_links(mapping, self.vedges, self.snapshot.edges, self.snapshot.paths)
        env = EOTrainingEnvironment(self.snapshot, distance_limit=None)
        env.begin_request(1, self.vnodes, self.vedges)
        env.step(2)
        _, _, _, info = env.step(9)
        self.assertEqual(info["node_mapping"], mapping)
        self.assertEqual(links[0]["bw"], info["link_mapping"][0]["bw"])
        self.assertEqual(selector.env.nodes, original.nodes)
        self.assertEqual(selector.env.edges, original.edges)

    def test_link_direction_and_aggregate_contention(self):
        self.assertFalse(map_directed_links({5: 9, 7: 2}, self.vedges,
                                            self.snapshot.edges, self.snapshot.paths))
        demand = [{"source": 5, "target": 7, "bw": 6.}] * 2
        self.assertFalse(map_directed_links({5: 2, 7: 9}, demand,
                                            self.snapshot.edges, self.snapshot.paths))
        self.assertEqual(self.snapshot.edges[(2, 9)], 10.)

    def test_window_order(self):
        events = [{"type": 1, "time": 10}, {"type": 0, "time": 90}, {"type": 1, "time": 100}]
        groups = list(event_windows(events, 100))
        self.assertEqual([e["type"] for e in groups[0][1]], [0, 1])
        self.assertEqual(groups[1][0], 200)

    def test_transaction_rollback(self):
        class FakeConnection:
            rolled_back = committed = False
            rowcount = 0
            calls = 0
            def begin(self): pass
            def cursor(self): return self
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def execute(self, *args):
                self.calls += 1
                self.rowcount = 1 if self.calls == 1 else 0
            def rollback(self): self.rolled_back = True
            def commit(self): self.committed = True
        conn = FakeConnection()
        with self.assertRaises(RuntimeError):
            apply_resources(conn, {2: 2.}, {(2, 9): 4.})
        self.assertTrue(conn.rolled_back)
        self.assertFalse(conn.committed)
        self.assertEqual(conn.calls, 2)  # CPU succeeded but BW failed: rollback both.

    def test_offline_event_replay_and_release(self):
        events = self.root / "events.yaml"
        events.write_text(yaml.safe_dump({"events": [
            {"id": 1, "type": 1, "v_net_id": 3, "time": 0},
            {"id": 9, "type": 1, "v_net_id": 3, "time": 10},
            {"id": 2, "type": 0, "v_net_id": 3, "time": 100},
        ]}), encoding="utf-8")
        config = {"checkpoint": str(self.weights), "event_file": str(events),
                  "output_dir": str(self.root / "output"), "vnet_dir": str(self.root),
                  "filename_pattern": "v_{vnr_id}.yaml", "distance_limit": None}
        with patch.object(runtime, "read_resources", return_value=(
                copy.deepcopy(self.snapshot.nodes), dict(self.snapshot.edges), dict(self.snapshot.edges))), \
             patch.object(runtime, "load_vnet", return_value=(self.vnodes, self.vedges)), \
             patch.object(DatabaseSnapshot, "from_connection", return_value=self.snapshot):
            # No database methods can be called on this sentinel in offline replay.
            result = run(config, object(), write_db=False)
        self.assertEqual(result["successes"], 1)
        self.assertEqual(result["requests"], 2)
        self.assertFalse(result["traffic_simulation"])
        self.assertEqual(result["active_before_cleanup"], 0)
        output = Path(result["output_dir"])
        resources = yaml.safe_load((output / "resource_usage.yaml").read_text())["records"]
        self.assertEqual(resources[0]["cpu"]["used"], 5.)
        self.assertEqual(resources[-1]["cpu"]["used"], 0.)
        self.assertEqual(resources[-1]["bandwidth"]["used"], 0.)
        self.assertEqual(resources[0]["acceptance_rate"], 0.5)
        self.assertEqual(resources[0]["domain_temperatures"], [])
        self.assertEqual(set(resources[0]), {"time", "bandwidth", "cpu", "acceptance_rate", "domain_temperatures"})
        self.assertIsInstance(resources[0]["time"], float)
        mappings = yaml.safe_load((output / "vne_mapping.yaml").read_text())
        self.assertEqual(mappings, [{"id": 1, "vnr_ind": 3, "event_id": 1, "event_time": 0.0,
                                    "links": [{"source": 2, "target": 9, "bw": 4.0, "linkpath": [[2, 9]]}]}])
        self.assertFalse((output / "vnr_flow_history.yaml").exists())
        self.assertEqual(json.loads((output / "experiment_summary.json").read_text())["status"], "completed")

    def test_numbered_outputs_never_overwrite(self):
        root = self.root / "numbered"
        first = new_run_directory(root)
        marker = first / "vne_mapping.yaml"
        marker.write_text("existing data", encoding="utf-8")
        second = new_run_directory(root)
        self.assertEqual(first.name, "1")
        self.assertEqual(second.name, "2")
        self.assertEqual(marker.read_text(), "existing data")

    def test_live_mode_resource_only(self):
        class EngineConnection:
            def cursor(self): return self
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def execute(self, query):
                # Anything outside the resource adapters must only check engines.
                assert "information_schema.TABLES" in query
            def fetchall(self):
                return [("node_single_table", "InnoDB"), ("link_single_table", "InnoDB")]

        nodes, edges = copy.deepcopy(self.snapshot.nodes), dict(self.snapshot.edges)
        def read(_):
            return copy.deepcopy(nodes), dict(edges), dict(self.snapshot.edges)
        def change(_, cpu, bandwidth, release=False):
            sign = 1 if release else -1
            for node, amount in cpu.items(): nodes[node]["cpu"] += sign * amount
            for edge, amount in bandwidth.items(): edges[edge] += sign * amount
        events = self.root / "live.yaml"
        events.write_text(yaml.safe_dump({"events": [
            {"id": 0, "type": 1, "v_net_id": 3, "time": 0},
            {"id": 1, "type": 0, "v_net_id": 3, "time": 100},
        ]}), encoding="utf-8")
        config = {"checkpoint": str(self.weights), "event_file": str(events),
                  "output_dir": str(self.root / "output"), "vnet_dir": str(self.root),
                  "filename_pattern": "v_{vnr_id}.yaml", "distance_limit": None}
        with patch.object(runtime, "read_resources", side_effect=read), \
             patch.object(runtime, "apply_resources", side_effect=change) as updates, \
             patch.object(runtime, "load_vnet", return_value=(self.vnodes, self.vedges)), \
             patch.object(DatabaseSnapshot, "from_connection", return_value=self.snapshot):
            result = run(config, EngineConnection(), write_db=True)
        self.assertEqual(updates.call_count, 2)
        self.assertEqual(nodes, self.snapshot.nodes)
        self.assertEqual(edges, self.snapshot.edges)
        output = Path(result["output_dir"])
        records = yaml.safe_load((output / "resource_usage.yaml").read_text())["records"]
        self.assertEqual(records[0]["bandwidth"]["used"], 4.0)
        self.assertEqual(records[-1]["bandwidth"]["used"], 0.0)
        self.assertFalse((output / "vnr_flow_history.yaml").exists())


if __name__ == "__main__":
    unittest.main()
