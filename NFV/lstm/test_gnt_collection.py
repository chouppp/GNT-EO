import unittest
from gnt_collection import window_schedule, domain_snapshot


class CollectionTests(unittest.TestCase):
    def test_causal_order(self):
        events = [dict(time=450, type=0), dict(time=410, type=1), dict(time=403, type=1)]
        schedule = window_schedule(events, 400)
        self.assertEqual([t for t, e in schedule], [403,405,410,410,415,420,425,430,435,450])
        self.assertEqual(schedule[2][1]["type"], 1)
        self.assertIsNone(schedule[3][1])
        self.assertEqual([t for t,e in schedule if e is None], list(range(405,436,5)))

    def test_no_arrivals_still_samples(self):
        self.assertEqual(len(window_schedule([], 0)), 7)

    def test_boundary_event_rejected(self):
        with self.assertRaises(ValueError):
            window_schedule([dict(time=500, type=1)], 400)

    def test_snapshot_preserves_decision_history(self):
        class Calculator:
            def compute_domain_temperature(self, nodes, prev_tm):
                return dict(T=prev_tm, F=10, H=1, TM=999)
        previous = [1,2,3,4,5]
        snapshot = domain_snapshot(Calculator(), [[i] for i in range(5)], previous)
        self.assertEqual(previous, [1,2,3,4,5])
        self.assertEqual([d['T'] for d in snapshot], previous)


if __name__ == '__main__':
    unittest.main()
