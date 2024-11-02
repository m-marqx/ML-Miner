import unittest
from unittest.mock import patch
from api.ccxt_api import CcxtAPI
import ccxt
import pandas as pd
from pandas import Timestamp, Timedelta
import numpy as np


class TestCcxtApi(unittest.TestCase):
    def setUp(self) -> None:
        self.ccxt_api = CcxtAPI("BTCUSDT", "2h", ccxt.binance())
        self.exchange = ccxt.binance()

    def test_get_since_value(self):
        with patch.object(self.exchange, "fetch_ohlcv") as mock_fetch_ohlcv:
            mock_fetch_ohlcv.return_value = [
                [1502942400000, 4261.48, 4328.69, 4261.32, 4315.32, 70.415925]
            ]
            result = self.ccxt_api.get_since_value()
            expected_result = 1502942400000
            self.assertEqual(result, expected_result)

    def test_get_all_klines(self):
        mock_first_call = [
            [1502942400000.0, 4261.48, 4328.69, 4261.32, 4315.32, 70.415925],
            [1510149600000.0, 7438.0, 7489.99, 7425.0, 7442.38, 150.754693],
        ]

        mock_second_call = (
            pd.read_parquet("api/tests/test_data/klines.parquet")
            .to_numpy()
            .tolist()
        )

        with patch.object(self.ccxt_api, "_fetch_klines") as mock_fetch_klines:
            mock_fetch_klines.side_effect = [
                mock_first_call,
                mock_second_call,
            ]
            result = self.ccxt_api.get_all_klines(1502978400000).klines_list
            expected_result = (
                pd.read_parquet(
                    "api/tests/test_data/get_all_klines_results.parquet"
                )
                .to_numpy()
                .tolist()
            )
            self.assertListEqual(result, expected_result)

    def test_to_OHLCV(self):
        path = "api/tests/test_data/get_all_klines_results.parquet"

        self.ccxt_api.klines_list = pd.read_parquet(path).to_numpy().tolist()

        results = self.ccxt_api.to_OHLCV().data_frame

        path_results = "api/tests/test_data/to_OHLCV_results.parquet"
        expected_results = pd.read_parquet(path_results).reset_index()
        expected_results["date"] = expected_results["date"].astype(
            "datetime64[ms]"
        )
        expected_results = expected_results.set_index("date")

        pd.testing.assert_frame_equal(results, expected_results)

    def test_date_check(self):
        rng = np.random.default_rng(33)
        simulated_data = rng.integers(0, 100, size=(8, 5))
        simulated_index = pd.date_range("2021-01-01", periods=10, freq="2h")
        simulated_index = simulated_index.drop(simulated_index[5:7])
        self.ccxt_api.data_frame = pd.DataFrame(
            simulated_data,
            index=simulated_index,
            columns=["open", "high", "low", "close", "volume"],
        )
        results = self.ccxt_api.date_check()

        expected_results = pd.DataFrame(
            {
                "open": {Timestamp("2021-01-01 14:00:00"): 11},
                "high": {Timestamp("2021-01-01 14:00:00"): 85},
                "low": {Timestamp("2021-01-01 14:00:00"): 34},
                "close": {Timestamp("2021-01-01 14:00:00"): 12},
                "volume": {Timestamp("2021-01-01 14:00:00"): 1},
                "actual_date": {
                    Timestamp("2021-01-01 14:00:00"): Timestamp(
                        "2021-01-01 14:00:00"
                    )
                },
                "previous_date": {
                    Timestamp("2021-01-01 14:00:00"): Timestamp(
                        "2021-01-01 08:00:00"
                    )
                },
                "timedelta": {
                    Timestamp("2021-01-01 14:00:00"): Timedelta(
                        "0 days 06:00:00"
                    )
                },
            }
        )

        pd.testing.assert_frame_equal(results, expected_results)
