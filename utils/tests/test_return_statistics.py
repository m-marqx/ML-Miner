import unittest

import pandas as pd
import numpy as np
from utils import Statistics


class TestStatistics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(33)
        size = 2000
        index = pd.date_range("1/1/2010", periods=size)
        simulated_results = rng.random(size).round(4) + rng.choice(
            [-1, 0], size
        )
        self.data = pd.DataFrame(
            simulated_results, columns=["Result"], index=index
        )

        self.statistics = Statistics(self.data)

    def test_calculate_all_statistics_is_percent(self):
        results = Statistics(
            self.data, is_percent=True
        ).calculate_all_statistics()

        expected_data = [
            [-4.03, -0.03, -0.06],
            [-2.78, -0.13, -0.25],
            [-4.3, -0.07, -0.14],
            [-3.89, -0.02, -0.04],
            [-3.04, -0.0, -0.01],
            [-2.61, 0.1, 0.2],
        ]

        expected_index = pd.DatetimeIndex(
            [
                "2010-12-31",
                "2011-12-31",
                "2012-12-31",
                "2013-12-31",
                "2014-12-31",
                "2015-12-31",
            ],
            dtype="datetime64[ns]",
            freq="A-DEC",
        )

        expected_columns = ["Expected_Value", "Sharpe_Ratio", "Sortino_Ratio"]

        expected_results = pd.DataFrame(
            expected_data, columns=expected_columns, index=expected_index
        )

        pd.testing.assert_frame_equal(results, expected_results)

    def test_calculate_all_statistics(self):
        results = Statistics(
            self.data, is_percent=False
        ).calculate_all_statistics()

        expected_data = [
            [-0.04, -0.03, -0.06],
            [-0.03, -0.13, -0.25],
            [-0.04, -0.07, -0.14],
            [-0.04, -0.02, -0.04],
            [-0.03, -0.0, -0.01],
            [-0.03, 0.1, 0.2],
        ]

        expected_index = pd.DatetimeIndex(
            [
                "2010-12-31",
                "2011-12-31",
                "2012-12-31",
                "2013-12-31",
                "2014-12-31",
                "2015-12-31",
            ],
            dtype="datetime64[ns]",
            freq="A-DEC",
        )

        expected_columns = ["Expected_Value", "Sharpe_Ratio", "Sortino_Ratio"]

        expected_results = pd.DataFrame(
            expected_data, columns=expected_columns, index=expected_index
        )

        pd.testing.assert_frame_equal(results, expected_results)

    def test_calculate_estimed_sharpe_ratio(self):
        results = self.statistics.calculate_estimed_sharpe_ratio()

        expected_data = [
            -0.029452474048659968,
            -0.12542684979737836,
            -0.06844183121739818,
            -0.020059967486655516,
            -0.004983801478043558,
            0.0952392421870382,
        ]

        expected_index = pd.DatetimeIndex(
            [
                "2010-12-31",
                "2011-12-31",
                "2012-12-31",
                "2013-12-31",
                "2014-12-31",
                "2015-12-31",
            ],
            dtype="datetime64[ns]",
            freq="A-DEC",
        )

        expected_name = "Result"

        expected_results = pd.Series(
            expected_data, name=expected_name, index=expected_index
        )

        pd.testing.assert_series_equal(results, expected_results)

    def test_calculate_estimed_sortino_ratio(self):
        results = self.statistics.calculate_estimed_sortino_ratio()

        expected_data = [
            -0.06211247137865812,
            -0.25052906501591743,
            -0.13775369222850153,
            -0.040957314030841074,
            -0.009646916972076269,
            0.19511164393651156,
        ]

        expected_index = pd.DatetimeIndex(
            [
                "2010-12-31",
                "2011-12-31",
                "2012-12-31",
                "2013-12-31",
                "2014-12-31",
                "2015-12-31",
            ],
            dtype="datetime64[ns]",
            freq="A-DEC",
        )

        expected_name = "Result"

        expected_results = pd.Series(
            expected_data, name=expected_name, index=expected_index
        )

        pd.testing.assert_series_equal(results, expected_results)

    def test_calculate_expected_value_resampled_is_percent(self):
        results = Statistics(
            self.data, is_percent=True
        ).calculate_expected_value("resampled")

        expected_data = [
            -4.032069588975256,
            -2.784082241501101,
            -4.2990852005234865,
            -3.8890000899362662,
            -3.044325817975389,
            -2.609088508850958,
        ]

        expected_index = pd.DatetimeIndex(
            [
                "2010-12-31",
                "2011-12-31",
                "2012-12-31",
                "2013-12-31",
                "2014-12-31",
                "2015-12-31",
            ],
            dtype="datetime64[ns]",
            freq="A-DEC",
        )

        expected_name = "Expected_Value"

        expected_results = pd.Series(
            expected_data, name=expected_name, index=expected_index
        )

        pd.testing.assert_series_equal(results, expected_results)

    def test_calculate_expected_value_resampled(self):
        results = Statistics(
            self.data, is_percent=False
        ).calculate_expected_value("resampled")

        expected_data = [
            -0.04032069588975256,
            -0.027840822415011013,
            -0.042990852005234864,
            -0.03889000089936266,
            -0.030443258179753883,
            -0.026090885088509583,
        ]

        expected_index = pd.DatetimeIndex(
            [
                "2010-12-31",
                "2011-12-31",
                "2012-12-31",
                "2013-12-31",
                "2014-12-31",
                "2015-12-31",
            ],
            dtype="datetime64[ns]",
            freq="A-DEC",
        )

        expected_name = "Expected_Value"

        expected_results = pd.Series(
            expected_data, name=expected_name, index=expected_index
        )

        pd.testing.assert_series_equal(results, expected_results)

    def test_calculate_expected_value_output_error(self):
        with self.assertRaises(ValueError):
            Statistics(self.data, is_percent=False).calculate_expected_value(
                output="invalid"
            )

        with self.assertRaisesRegex(
            ValueError, "Invalid output type. Use 'complete' or 'resampled'."
        ):
            Statistics(self.data, is_percent=False).calculate_expected_value(
                output="invalid"
            )

    def test_calculate_expected_value_complete_is_percent(self):
        results = Statistics(
            self.data, is_percent=False
        ).calculate_expected_value("complete")

        expected_value = pd.read_parquet(
            "utils/tests/test_data/expected_value.parquet"
        )

        expected_value.index = pd.date_range("1/1/2010", periods=2000)

        pd.testing.assert_frame_equal(results, expected_value)

    def test_calculate_expected_value_complete(self):
        results = Statistics(
            self.data, is_percent=True
        ).calculate_expected_value("complete")

        expected_value = pd.read_parquet(
            "utils/tests/test_data/expected_value_is_percent.parquet"
        )

        expected_value.index = pd.date_range("1/1/2010", periods=2000)

        pd.testing.assert_frame_equal(results, expected_value)
