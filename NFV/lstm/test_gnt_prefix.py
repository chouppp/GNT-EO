"""Synthetic fixtures test the data contract only, never prediction quality."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import yaml
import gnt_dataset as ds
import gnt_prefix as gp


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ds.ROOT/"work")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.domains = [dict(domain_id=i, F=10., H=2., T=100.+i) for i in range(5)]
        self.raw = self.root/"resource_usage.yaml"
        self.raw.write_text(yaml.safe_dump([dict(time=500, domain_temperatures=self.domains)]), encoding="utf-8")
        for t in range(405, 436, 5):
            self.path = gp.record_observation("1", 500, t, t, self.domains, self.root)

    def build(self):
        return gp.build_samples(self.raw, self.path, "1", 500, 500)

    def test_exact_target_and_times(self):
        sample = self.build()
        self.assertEqual(sample.x.shape, (1, 7, 20))
        np.testing.assert_array_equal(sample.history_times[0], np.arange(405, 436, 5))
        np.testing.assert_array_equal(sample.y[0], ds.read_states(self.raw)[1][0, 2])

    def test_future_event_rejected(self):
        with self.assertRaisesRegex(ValueError, "Future events"):
            gp.record_observation("1", 500, 435, 450, self.domains, self.root)

    def test_late_observation_rejected(self):
        with self.assertRaisesRegex(ValueError, "first 35%"):
            gp.record_observation("1", 500, 440, 440, self.domains, self.root)

    def test_missing_sample_rejected(self):
        self.path.write_text("\n".join(self.path.read_text().splitlines()[:-1]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing observations"):
            self.build()

    def test_duplicate_rejected(self):
        gp.record_observation("1", 500, 435, 435, self.domains, self.root)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.build()

    def test_wrong_run_rejected(self):
        self.path.write_text(self.path.read_text().replace('"run": "1"', '"run": "2"'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "run does not match"):
            self.build()

    def test_synthetic_schema_rejected(self):
        record = json.loads(self.path.read_text().splitlines()[0])
        record["schema"] = "synthetic-final-state-substeps"
        with self.assertRaisesRegex(ValueError, "genuine"):
            gp.validate_record(record)


if __name__ == "__main__":
    unittest.main()
