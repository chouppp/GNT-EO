"""Regression tests for temperature conventions, temporal labels, validation isolation, and legacy-result compatibility."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import lstm_predict as lp
import picture

ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = lp.load_all_runs(ROOT / "GNT", ["1", "2", "3"], "lstm_traffic_500_2000.npz")

    def test_clip_before_average_keeps_raw_temperatures(self):
        domains = [{"domain_id": i, "T": t} for i, t in enumerate([50000., -10000., None, 0., 0.])]
        self.assertEqual(lp.average_domain_temperature(domains), 0.)
        self.assertEqual(picture.calc_average_temperature(domains), 0.)
        self.assertEqual(domains[0]["T"], 50000.)

    def test_grouped_validation_is_disjoint_and_deterministic(self):
        candidates = [self.data["1"], self.data["2"]]
        fit, val = lp.split_training_runs(candidates, .15, 42)
        fit2, val2 = lp.split_training_runs(candidates, .15, 42)
        self.assertFalse({d.run for d in fit} & {d.run for d in val})
        self.assertEqual({d.run for d in fit+val}, {"1", "2"})
        self.assertEqual([d.run for d in val], [d.run for d in val2])
        self.assertNotIn("3", [d.run for d in fit+val])

    def test_zero_residual_is_exact_last_value_baseline(self):
        class ZeroModel:
            def predict(self, values, **kwargs):
                return np.zeros((len(values), 1), dtype=np.float32)
        d = self.data["1"]
        prediction = lp.predict_future_mean(ZeroModel(), d.observed, 13, 32)
        np.testing.assert_array_equal(prediction, lp.baseline_predictions(d.observed, 13)["last_value"])

    def test_residual_samples_reconstruct_labels(self):
        d = self.data["1"]
        x, residual = lp.build_future_mean_samples([d], 7, 13)
        _, scale, last = lp.residual_features(d.observed)
        active = np.any(d.observed > 0, axis=1)
        reconstructed = last[active] + residual.ravel() * scale[active] / 100.
        np.testing.assert_allclose(reconstructed, d.target.mean(axis=1, dtype=np.float64)[active], rtol=1e-6)
        self.assertEqual(x.shape, (int(active.sum()), 7, 1))

    def test_perfect_flow_has_zero_error_and_excludes_warmup(self):
        d = self.data["1"]
        result = lp.evaluate_prediction(d, d.target, 400.)
        self.assertEqual(result[0]["window_gnt"]["sample_count"], 16)
        self.assertLess(result[0]["window_gnt"]["MAE"], 1e-7)
        self.assertEqual(result[0]["future_mean"]["MAE"], 0.)

    def test_old_npz_recomputed_with_common_temperature_function(self):
        d = self.data["1"]
        p = lp.load_saved_prediction(ROOT / "GNT/1/lstm_prediction_500_2000.npz", d)
        result = lp.evaluate_prediction(d, p)
        times, actual, predicted, baseline = picture.load_window_comparison(ROOT / "GNT", ROOT / "GNT", ["1"], 500., 2000.)
        self.assertEqual(times.shape, (16,))
        np.testing.assert_allclose(actual[0], lp.gnt_values(result[3])[1:])
        np.testing.assert_allclose(predicted[0], lp.gnt_values(result[4])[1:])

    def test_misaligned_saved_prediction_is_rejected(self):
        d = self.data["1"]
        with np.load(ROOT / "GNT/1/lstm_prediction_500_2000.npz") as z:
            values = {k:z[k] for k in z.files}
        values["base_times"] = values["base_times"] + 100
        (ROOT / "work").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "work") as folder:
            path = Path(folder) / "bad.npz"
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "base_times"):
                lp.load_saved_prediction(path, d)

    def test_missing_windows_and_nonfinite_metrics_are_rejected(self):
        with self.assertRaises(ValueError):
            lp.compute_gnt_records(np.array([400., 600.]), np.zeros((2,1)), np.array(["7"]), np.array(["2"]))
        with self.assertRaises(ValueError):
            picture.calculate_prediction_metrics([[1., np.nan]], [[1., 2.]])

    def test_mean_curve_does_not_replace_per_run_metrics(self):
        report = picture.calculate_prediction_metrics([[1.,1.],[-1.,-1.]], [[-1.,-1.],[1.,1.]])
        self.assertEqual(report["per_run_time"]["MAE"], 2.)
        self.assertEqual(report["mean_curve"]["MAE"], 0.)

    def test_recorded_actual_preserves_original_sixteen_point_curve(self):
        # Verified point by point against original picture.py; never replace ground truth with recomputed window values.
        expected = [936.958, -1176.037, -1591.159, 1075.035, 516.174,
                    303.695, 405.771, 275.205, 1159.131, 111.453,
                    632.127, 193.986, -1047.006, 1517.668, -1128.342, -721.094]
        actual = picture.load_recorded_actual(picture.RAW_BASE_DIR, picture.RUNS, np.arange(500., 2001., 100.))
        np.testing.assert_allclose(actual.mean(axis=0), expected, atol=.00051, rtol=0)

    def test_duplicate_raw_times_average_after_domain_clipping(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "work") as folder:
            path = Path(folder) / "resource_usage.yaml"
            path.write_text("records:\n- time: 500\n  domain_temperatures:\n  - domain_id: 0\n    T: 50000\n"
                            "- time: 500\n  domain_temperatures:\n  - domain_id: 0\n    T: -10000\n", encoding="utf-8")
            times, actual = picture.load_raw_actual_result(path)
            np.testing.assert_array_equal(times, [500.])
            np.testing.assert_array_equal(actual, [0.])

    def test_missing_recorded_actual_does_not_fall_back_to_window_value(self):
        with self.assertRaises(FileNotFoundError):
            picture.load_recorded_actual(ROOT / "work/nonexistent_raw_logs", ["1"], [500.])


if __name__ == "__main__":
    unittest.main()
