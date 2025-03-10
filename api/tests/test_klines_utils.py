import unittest
import pandas as pd
from api.kline_utils import KlineTimes, get_max_interval


class TestKlineTimesIntervalAdjusted(unittest.TestCase):
    def setUp(self):
        self.symbol = "BTCUSDT"
        self.interval = "14h"
        self.kline_times = KlineTimes(self.interval, adjust_interval=True)

    def test_calculate_max_multiplier(self):
        expected_result = 1500
        result = self.kline_times.calculate_max_multiplier()

        self.assertEqual(result, expected_result)

    def test_calculate_max_multiplier_max_days(self):
        result = self.kline_times.calculate_max_multiplier()
        interval_digit = float(self.kline_times.interval[:-1])
        result_days = result * interval_digit / 24
        self.assertLessEqual(result_days, 200)

    def test_get_end_times(self):
        results = pd.Series(self.kline_times.get_end_times())

        expected_results = pd.Series(
            [
                1.5971184e12,
                1.6079184e12,
                1.6187184e12,
                1.6295184e12,
                1.6403184e12,
                1.6511184e12,
                1.6619184e12,
                1.6727184e12,
                1.6835184e12,
                1.6943184e12,
                1.7051184e12,
                1.7159184e12,
                1.7267184e12,
            ]
        )

        results = results.reindex(expected_results.index)

        pd.testing.assert_series_equal(results, expected_results)

    def test_get_max_interval(self):
        results = get_max_interval(self.kline_times.interval)
        expected_results = "2h"

        self.assertEqual(results, expected_results)


class TestKlineTimesNotAdjusted(unittest.TestCase):
    def setUp(self):
        self.symbol = "BTCUSDT"
        self.interval = "14h"
        self.kline_times = KlineTimes(self.interval, adjust_interval=False)

    def test_calculate_max_multiplier(self):
        expected_result = 342
        result = self.kline_times.calculate_max_multiplier()

        self.assertEqual(result, expected_result)

    def test_calculate_max_multiplier_max_days(self):
        result = self.kline_times.calculate_max_multiplier()
        result_days = result * 14 / 24
        self.assertLessEqual(result_days, 200)

    def test_get_end_times(self):
        results = pd.Series(self.kline_times.get_end_times())

        expected_results = pd.Series(
            [
                1.59711840e12,
                1.61435520e12,
                1.63159200e12,
                1.64882880e12,
                1.66606560e12,
                1.68330240e12,
                1.70053920e12,
                1.71777600e12,
            ]
        )

        results = results.reindex(expected_results.index)
        pd.testing.assert_series_equal(results, expected_results)

    def test_get_max_interval(self):
        results = get_max_interval(self.kline_times.interval)
        expected_results = "2h"

        self.assertEqual(results, expected_results)
