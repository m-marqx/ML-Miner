import unittest
import numpy as np
from api.kline_utils import KlineTimes


class TestKlineTimes(unittest.TestCase):
    def setUp(self):
        self.symbol = "BTCUSDT"
        self.interval = "14h"
        self.kline_times = KlineTimes(self.interval)

    def test_calculate_max_multiplier(self):
        expected_result = 342
        result = self.kline_times.calculate_max_multiplier()

        self.assertEqual(result, expected_result)

    def test_calculate_max_multiplier_max_days(self):
        result = self.kline_times.calculate_max_multiplier()
        result_days = result * 14 / 24
        self.assertLessEqual(result_days, 200)

    def test_get_end_times(self):
        results = self.kline_times.get_end_times()[:-1]

        expected_results = np.array(
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

        np.testing.assert_array_equal(results, expected_results)

    def test_get_max_interval(self):
        results = self.kline_times.get_max_interval
        expected_results = "2h"

        self.assertEqual(results, expected_results)
