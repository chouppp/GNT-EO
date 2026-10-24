import unittest

from .training_env import DatabaseSnapshot, EOTrainingEnvironment


class EOTrainingEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = DatabaseSnapshot(
            nodes={
                0: {"cpu": 10.0, "max_cpu": 10.0, "pos": 0.0, "pos2": 0.0},
                1: {"cpu": 10.0, "max_cpu": 10.0, "pos": 1.0, "pos2": 0.0},
                2: {"cpu": 10.0, "max_cpu": 10.0, "pos": 2.0, "pos2": 0.0},
            },
            edges={(0, 1): 8.0, (1, 2): 8.0},
            paths={
                (0, 1): [{"hop": 1, "path": [(0, 1)]}],
                (0, 2): [{"hop": 2, "path": [(0, 1), (1, 2)]}],
                (1, 2): [{"hop": 1, "path": [(1, 2)]}],
            },
        )

    def test_success_commit_and_release(self):
        env = EOTrainingEnvironment(self.snapshot, distance_limit=None)
        nodes = [{"id": 0, "cpu": 2.0}, {"id": 1, "cpu": 3.0}]
        edges = [{"source": 0, "target": 1, "bw": 4.0}]
        observation = env.begin_request(7, nodes, edges)
        self.assertTrue(observation.action_mask[0])
        observation, reward, done, _ = env.step(0)
        self.assertFalse(done)
        observation, reward, done, info = env.step(2)
        self.assertTrue(done)
        self.assertTrue(info["success"])
        self.assertGreaterEqual(reward, 1.0)
        self.assertEqual(env.nodes[0]["cpu"], 8.0)
        self.assertEqual(env.nodes[2]["cpu"], 7.0)
        self.assertEqual(env.edges[(0, 1)], 4.0)
        self.assertEqual(env.edges[(1, 2)], 4.0)
        self.assertTrue(env.release(7))
        self.assertEqual(env.edges[(0, 1)], 8.0)
        self.assertEqual(env.nodes[0]["cpu"], 10.0)

    def test_direction_is_not_reversed(self):
        env = EOTrainingEnvironment(self.snapshot, distance_limit=None)
        nodes = [{"id": 0, "cpu": 1.0}, {"id": 1, "cpu": 1.0}]
        edges = [{"source": 0, "target": 1, "bw": 1.0}]
        env.begin_request(8, nodes, edges, virtual_order=[0, 1])
        observation, reward, done, info = env.step(2)
        self.assertIsNone(observation)
        self.assertTrue(done)
        self.assertFalse(info["success"])
        self.assertEqual(reward, -1.0)  # no directed path leaving physical node 2


if __name__ == "__main__":
    unittest.main()
