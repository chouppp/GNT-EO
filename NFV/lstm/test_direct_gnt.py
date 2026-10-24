"""Regression tests for direct GNT labels, temporal boundaries, and data splits."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import yaml
import gnt_dataset as ds
import gnt_predict as gp
import gnt_evaluate as ev
import picture


class DirectGNTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs=[str(i) for i in range(1,9)]
        cls.data=ds.load_runs(ds.RAW_BASE,cls.runs)
        (ds.ROOT/"work").mkdir(exist_ok=True)

    def test_original_true_curve_is_preserved(self):
        actual=np.stack([self.data[r].y.mean(1)*ds.LIMIT for r in self.runs])
        expected=picture.load_recorded_actual(picture.RAW_BASE_DIR,self.runs,self.data["1"].times)
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-12)

    def test_first_input_is_200_300_400_and_target_is_500(self):
        d=self.data["1"]
        np.testing.assert_array_equal(d.history_times[0],[200,300,400])
        self.assertEqual(d.times[0],500)
        self.assertTrue(np.all(d.history_times<d.times[:,None]))
        self.assertEqual(d.x.shape,(16,3,20))

    def test_changing_target_state_does_not_change_its_input(self):
        source=Path(self.data["1"].source)
        content=yaml.load(source.read_text(encoding="utf-8"),Loader=yaml.CSafeLoader)
        for r in content["records"]:
            if r["time"]==500:
                for d in r["domain_temperatures"]:
                    d.update(F=999999.,H=3.,T=10000.)
        with tempfile.TemporaryDirectory(dir=ds.ROOT/"work") as folder:
            p=Path(folder)/"changed.yaml"
            p.write_text(yaml.safe_dump(content),encoding="utf-8")
            altered=ds.build_samples(p,"1")
        np.testing.assert_array_equal(self.data["1"].x[0],altered.x[0])
        self.assertFalse(np.array_equal(self.data["1"].y[0],altered.y[0]))

    def test_default_groups_are_disjoint(self):
        fold=ds.make_folds(self.runs)[0]
        self.assertEqual(fold["fit"],["1","2","3","4","5"])
        self.assertEqual(fold["validation"],["6"])
        self.assertEqual(fold["test"],["7","8"])
        with self.assertRaises(ValueError):
            ds.make_folds(self.runs,train=["1","2","3","4","5"],validation=["5","6"],test=["7","8"])

    def test_loro_tests_each_run_once(self):
        folds=ds.make_folds(self.runs,"loro")
        self.assertEqual([f["test"][0] for f in folds],self.runs)
        for f in folds:
            self.assertFalse(set(f["fit"])&set(f["test"]+f["validation"]))
            self.assertFalse(set(f["test"])&set(f["validation"]))

    def test_scaler_only_uses_fit_runs(self):
        x,_=ds.concatenate(self.data,["1","2","3","4","5"])
        mean,std=ds.fit_scaler(x)
        expected=x.mean(axis=(0,1))
        np.testing.assert_allclose(mean,expected)
        self.assertTrue(np.all(std>0))
        tx=ds.transform(self.data["8"].x,mean,std)
        np.testing.assert_array_equal(mean,expected)
        self.assertTrue(np.all(np.isfinite(tx)))

    def test_baselines_use_history_only(self):
        d=self.data["1"]
        b=ds.baselines(d.x)
        np.testing.assert_array_equal(b["last_value"],d.x[:,-1,10:15])
        np.testing.assert_allclose(b["history_mean"],d.x[:,:,10:15].mean(1))
        np.testing.assert_array_equal(b["zero"],np.zeros_like(d.y))

    def test_duplicate_time_uses_clipping_before_mean(self):
        with tempfile.TemporaryDirectory(dir=ds.ROOT/"work") as folder:
            p=Path(folder)/"duplicate.yaml"
            p.write_text("records:\n- time: 100\n  domains:\n  - domain_id: 0\n    T: 50000\n"
                         "- time: 100\n  domains:\n  - domain_id: 0\n    T: -10000\n",encoding="utf-8")
            times,states,_=ds.read_states(p)
            self.assertEqual(states[0,2,0],0.)

    def test_missing_window_is_not_filled(self):
        with self.assertRaises(ValueError):
            ds.build_samples(self.data["1"].source,"1",history=5,start=500)

    def test_ridge_handles_constant_features(self):
        x=np.ones((6,3,20),np.float32)
        y=np.tile(np.array([.1,-.1,.2,-.2,0]),(6,1))
        w=gp.fit_ridge(x,y)
        np.testing.assert_allclose(gp.predict_ridge(x,w),y,atol=1e-10)

    def test_evaluation_keeps_individual_errors(self):
        actual=np.array([[100.]*5,[-100.]*5])
        result=ev.evaluate(actual,{"model":-actual},np.array(["1","2"]),np.array([500.,500.]))
        self.assertEqual(result["model"]["per_run_time_gnt"]["MAE"],200.)
        self.assertEqual(result["model"]["mean_curve_auxiliary"]["MAE"],0.)

    def test_duplicate_prediction_keys_rejected(self):
        with self.assertRaises(ValueError):
            ev.evaluate(np.zeros((2,5)),{"zero":np.zeros((2,5))},np.array(["1","1"]),np.array([500.,500.]))

    def test_shared_validation_keeps_test_separate(self):
        original=ds.make_folds(self.runs)
        fold=gp.configure_validation(original,True)[0]
        self.assertEqual(fold["fit"],fold["validation"])
        self.assertEqual(fold["unused_validation_runs"],["6"])
        self.assertEqual(fold["test"],["7","8"])
        self.assertFalse(set(fold["fit"])&set(fold["test"]))
        self.assertFalse(fold["validation_is_independent"])
        self.assertEqual(original[0]["validation"],["6"])

    def test_independent_validation_remains_default(self):
        fold=gp.configure_validation(ds.make_folds(self.runs))[0]
        self.assertEqual(fold["validation"],["6"])
        self.assertTrue(fold["validation_is_independent"])

    def test_diagnostic_cannot_include_test_in_training(self):
        with self.assertRaises(ValueError):
            gp.configure_validation([{"fit":["1"],"validation":["2"],"test":["1"]}],True)


if __name__=="__main__":unittest.main()
