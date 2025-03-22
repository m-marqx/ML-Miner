import unittest

from machine_learning.adjust_predicts import adjust_predict_both_side, adjust_predict_one_side
import pandas as pd
import numpy as np

class TestAdjustPredicts(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(33)

    def test_adjust_predict_one_side(self):
        synthetic = pd.Series(
            self.rng.integers(0, 2, 20),
            index=pd.date_range("2020-01-01", periods=20, freq="D"),
            name="predict",
        )

        date_indexes = pd.DatetimeIndex(
            [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-16",
                "2020-01-17",
                "2020-01-18",
                "2020-01-19",
                "2020-01-20",
            ],
            dtype="datetime64[ns]",
            freq="D",
        )

        ref = adjust_predict_one_side(synthetic, 3, 2, 1)
        expect = pd.Series(
            [1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
            , date_indexes,
            name="predict",
            dtype="int64",
        )

        pd.testing.assert_series_equal(ref, expect)

    def test_adjust_predict_both_side(self):
        synthetic = pd.Series(
            self.rng.integers(-1, 2, 20),
            index=pd.date_range("2020-01-01", periods=20, freq="D"),
            name="Predict",
        ).to_frame()

        date_indexes = pd.DatetimeIndex(
            [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-16",
                "2020-01-17",
                "2020-01-18",
                "2020-01-19",
                "2020-01-20",
            ],
            dtype="datetime64[ns]",
            freq="D",
        )

        ref = adjust_predict_both_side(synthetic, 3, 2)
        expect = pd.Series(
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1, 0, 1, 1, 0, 0, 0, -1, 1, 0]
            , date_indexes,
            name="Predict",
            dtype="int64",
        ).to_frame()

        pd.testing.assert_frame_equal(ref, expect)
