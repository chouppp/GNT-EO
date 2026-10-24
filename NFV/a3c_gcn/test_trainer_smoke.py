import json
import tempfile
import unittest
from pathlib import Path

import yaml

from .trainer import TrainConfig, train
from .training_env import DatabaseSnapshot


class TrainerSmokeTest(unittest.TestCase):
    def test_one_worker_one_episode(self):
        snapshot = DatabaseSnapshot(
            nodes={
                0: {"cpu": 10.0, "max_cpu": 10.0, "pos": 0.0, "pos2": 0.0},
                1: {"cpu": 10.0, "max_cpu": 10.0, "pos": 1.0, "pos2": 0.0},
            },
            edges={(0, 1): 10.0, (1, 0): 10.0},
            paths={
                (0, 1): [{"hop": 1, "path": [(0, 1)]}],
                (1, 0): [{"hop": 1, "path": [(1, 0)]}],
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vnet_dir = root / "vnets"
            output_dir = root / "output"
            vnet_dir.mkdir()
            (vnet_dir / "v_1.yaml").write_text(
                """graph [
  node [
    id 0
    pos 0
    pos 0
    cpu 1
  ]
  node [
    id 1
    pos 1
    pos 0
    cpu 1
  ]
  edge [
    source 0
    target 1
    bw 1
  ]
]
""",
                encoding="utf-8",
            )
            events = root / "events.yaml"
            events.write_text(
                yaml.safe_dump({
                    "events": [
                        {"id": 0, "type": 1, "v_net_id": 1, "time": 0.0},
                        {"id": 1, "type": 0, "v_net_id": 1, "time": 1.0},
                    ]
                }),
                encoding="utf-8",
            )
            config = TrainConfig(
                vnet_dir=str(vnet_dir),
                filename_pattern="v_{vnr_id}.yaml",
                event_files=[str(events)],
                output_dir=str(output_dir),
                episodes=1,
                num_workers=1,
                checkpoint_interval=1,
                distance_limit=None,
                seed=7,
            )
            summary = train(snapshot, config)
            self.assertEqual(summary["episodes"], 1)
            self.assertEqual(summary["requests"], 1)
            self.assertEqual(summary["best_episode"]["episode_index"], 1)
            self.assertEqual(
                summary["best_episode"]["acceptance_ratio"],
                summary["overall_acceptance_ratio"],
            )
            greedy = summary["training_greedy_evaluation"]
            self.assertEqual(greedy["requests"], 1)
            self.assertEqual(greedy["successes"], 1)
            self.assertEqual(greedy["acceptance_ratio"], 1.0)
            self.assertEqual(greedy["action_selection"], "greedy_argmax")
            self.assertTrue(greedy["in_sample"])
            self.assertEqual(greedy["failure_reasons"], {})
            self.assertTrue((output_dir / "a3c_gcn_actor_last.pt").exists())
            self.assertTrue((output_dir / "a3c_gcn_actor_best.pt").exists())
            self.assertEqual(summary["best_model_evaluation"]["acceptance_ratio"], 1.0)
            self.assertTrue(summary["model_selection_in_sample"])
            self.assertTrue((output_dir / "checkpoint_evaluations.json").exists())
            self.assertTrue((output_dir / "training_greedy_evaluation.json").exists())
            stored = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["requests"], 1)


if __name__ == "__main__":
    unittest.main()
