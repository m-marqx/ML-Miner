from unittest import TestCase
from custom_exceptions.invalid_arguments import InvalidArgumentError

from machine_learning.features.onchain_features import OnchainFeatures
import pandas as pd


class TestOnchainFeatures(TestCase):
    def setUp(self):
        self.avg_infos = ["avgfee", "avgfeerate", "avgtxsize"]
        self.feerate_percentiles = ["feerate_percentiles"]
        self.max_infos = ["maxfee", "maxfeerate", "maxtxsize"]
        self.median_infos = ["mediantime", "medianfee", "mediantxsize"]
        self.min_infos = ["minfee", "minfeerate", "mintxsize"]
        self.total_infos = [
            "total_out",
            "total_size",
            "total_weight",
            "totalfee",
        ]

        self.txs_infos = [
            "ins",
            "outs",
            "txs",
            "utxo_increase",
            "utxo_size_inc",
            "utxo_increase_actual",
            "utxo_size_inc_actual",
        ]

        self.all_valid_infos = [
            *self.avg_infos,
            # *self.feerate_percentiles,
            # *self.max_infos,
            *self.median_infos,
            # *self.min_infos,
            *self.total_infos,
            *self.txs_infos,
        ]

        self.all_infos = [
            *self.avg_infos,
            *self.feerate_percentiles,
            *self.max_infos,
            *self.median_infos,
            *self.min_infos,
            *self.total_infos,
            *self.txs_infos,
        ]

        self.onchain_features = OnchainFeatures(None, 1577, 30, False)

    def test_set_bin(self):
        self.onchain_features.set_bins(10)
        self.assertEqual(self.onchain_features.bins, 10)

    def test_calculate_resampled_data(self):
        test_df = self.onchain_features.calculate_resampled_data(
            self.all_valid_infos, "D"
        )

        expected_df = pd.read_parquet(
            "machine_learning/features/tests/onchain_all_features.parquet"
        )

        expected_df = expected_df.asfreq("D").iloc[:-1]

        test_df = test_df.reindex(expected_df.index)

        pd.testing.assert_frame_equal(test_df, expected_df)

    def test_calculate_resampled_data_all_infos_error(self):
        invalid_infos = [*self.min_infos, *self.feerate_percentiles]

        for info in invalid_infos:
            with self.assertRaises(InvalidArgumentError):
                self.onchain_features.calculate_resampled_data([info], "D")

        with self.assertRaises(InvalidArgumentError):
            self.onchain_features.calculate_resampled_data(self.all_infos, "D")

        with self.assertRaises(InvalidArgumentError):
            self.onchain_features.calculate_resampled_data(self.min_infos, "D")

        with self.assertRaises(InvalidArgumentError):
            self.onchain_features.calculate_resampled_data(
                self.feerate_percentiles, "D"
            )

    def test_calculate_std_ratio_feature_short_too_big(self):
        with self.assertRaisesRegex(
            InvalidArgumentError,
            "Short window size must be smaller than long window size.",
        ):
            self.onchain_features.calculate_std_ratio_feature("avgfee", 4, 2)

    def test_calculate_std_ratio_feature_short_too_small(self):
        with self.assertRaisesRegex(
            InvalidArgumentError,
            "Window sizes must be greater than or equal to 2.",
        ):
            self.onchain_features.calculate_std_ratio_feature("avgfee", 0, 4)

        with self.assertRaisesRegex(
            InvalidArgumentError,
            "Window sizes must be greater than or equal to 2.",
        ):
            self.onchain_features.calculate_std_ratio_feature("avgfee", 0, 1)

    def test_calculate_std_ratio_feature_long_equals_to_short(self):
        with self.assertRaisesRegex(
            InvalidArgumentError, "Window sizes must be different."
        ):
            self.onchain_features.calculate_std_ratio_feature("avgfee", 31, 31)

    def test_calculate_std_ratio_feature_window_sizes_is_not_int(self):
        with self.assertRaisesRegex(
            InvalidArgumentError, "Window sizes must be integers"
        ):
            self.onchain_features.calculate_std_ratio_feature(
                "avgfee", 5.5, 11
            )

        with self.assertRaisesRegex(
            InvalidArgumentError, "Window sizes must be integers"
        ):
            self.onchain_features.calculate_std_ratio_feature(
                "avgfee", 5, 11.5
            )

    def test_calculate_std_ratio_feature_is_not_pandas(self):
        with self.assertRaisesRegex(
            InvalidArgumentError,
            "Feature must be a pandas Series or pandas DataFrame.",
        ):
            self.onchain_features.calculate_std_ratio_feature("avgfee", 5, 10)

    def test_calculate_std_ratio_feature_is_pandas_empty(self):
        with self.assertRaisesRegex(
            InvalidArgumentError, "Feature can't be empty"
        ):
            self.onchain_features.calculate_std_ratio_feature(
                pd.Series(), 5, 10
            )

    # def test_calculate_std_ratio_feature_single_column(self):
    #     test_df = self.onchain_features.calculate_std_ratio_feature(
    #         self.onchain_features.onchain_data[["txs"]], 2, 4
    #     )

    #     expected_df = pd.read_parquet(
    #         "machine_learning/features/tests/onchain_std_ratio_feature_single_column.parquet"
    #     ).iloc[:-1].loc["2010":]

    #     test_df = test_df.loc[expected_df.index.unique()]

    #     pd.testing.assert_frame_equal(test_df, expected_df)

    def test_create_std_ratio_feature_multiple_columns(self):
        all_valid_infos = [
            *self.avg_infos,
            "mediantime",
            *self.total_infos,
            *self.txs_infos,
        ]

        test_df = self.onchain_features.create_std_ratio_feature(
            all_valid_infos, "D", 2, 4
        ).dataset

        expected_df = pd.read_parquet(
            "machine_learning/features/tests/onchain_std_ratio_feature.parquet"
        )

        expected_df = expected_df.asfreq("D").iloc[:-1]
        test_df = test_df.reindex(expected_df.index)

        pd.testing.assert_frame_equal(test_df, expected_df)

    def test_create_std_feature_error(self):
        with self.assertRaisesRegex(
            InvalidArgumentError, "feerate_percentiles isn't compatible"
        ):
            self.onchain_features.create_std_ratio_feature(
                self.all_infos, "D", 2, 4
            )
