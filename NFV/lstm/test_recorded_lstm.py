import unittest
import numpy as np
import recorded_lstm as rl


class Item:
    def __init__(self, value):
        self.times = np.array([500., 600.])
        self.y = np.full((2,5), value, dtype=float)
        self.x = np.full((2,3,20), value, dtype=float)


class RecordedTests(unittest.TestCase):
    def setUp(self):
        self.data = {str(i):Item(i) for i in range(1,5)}

    def test_profile_excludes_test_run(self):
        result = rl.time_profile(self.data, ["2","3","4"])
        np.testing.assert_allclose(result, 3.)

    def test_leave_one_out_mean_curve_identity(self):
        predictions=[]
        for test in self.data:
            predictions.append(rl.time_profile(self.data,[r for r in self.data if r != test]))
        np.testing.assert_allclose(np.mean(predictions,axis=0),
                                   np.mean([d.y for d in self.data.values()],axis=0))

    def test_training_residual_has_no_self_target_profile(self):
        _, residual = rl.residual_training_data(self.data,["1","2","3"])
        expected=np.concatenate([self.data["1"].y-2.5,
                                 self.data["2"].y-2.,
                                 self.data["3"].y-1.5])
        np.testing.assert_allclose(residual, expected)


if __name__ == "__main__": unittest.main()
