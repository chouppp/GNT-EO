import unittest
import numpy as np
import mean_lstm as ml


class Item:
    def __init__(self,value):
        self.times=np.array([500.,600.,700.,800.,900.,1000.])
        self.x=np.full((6,3,20),value,dtype=float)
        self.y=np.full((6,5),value,dtype=float)


class MeanLSTMTests(unittest.TestCase):
    def test_average_is_computed_before_learning(self):
        data={"1":Item(1),"2":Item(3)}
        times,x,y=ml.average_runs(data,["1","2"])
        np.testing.assert_array_equal(times,data["1"].times)
        np.testing.assert_allclose(x,2.)
        np.testing.assert_allclose(y,2.)

    def test_split_is_strictly_chronological(self):
        times=np.arange(500.,1200.,100.)
        train,val,test=ml.chronological_masks(times,800.,1000.)
        self.assertTrue(np.all(times[train] < times[val].min()))
        self.assertTrue(np.all(times[val] < times[test].min()))
        self.assertFalse(np.any(train & val) or np.any(train & test) or np.any(val & test))

    def test_short_training_period_rejected(self):
        with self.assertRaises(ValueError):
            ml.chronological_masks(np.arange(500.,900.,100.),600.,700.)


if __name__=="__main__": unittest.main()
