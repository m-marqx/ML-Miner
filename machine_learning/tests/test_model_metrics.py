import unittest
import warnings

import pandas as pd
import numpy as np

from machine_learning.model_metrics import ModelMetrics
from machine_learning.model_creator import ModelCreator


class TestModelCreator(unittest.TestCase):
    def setUp(self):
        model_df = pd.read_parquet("machine_learning/tests/data/model_df.parquet")
        onchain_df = pd.read_parquet("machine_learning/tests/data/onchain_df.parquet")

        four_years = int(365.25 * 4)

        param_kwargs = dict(
            dataset=model_df,
            onchain_data=onchain_df,
            test_index=four_years,
            train_in_middle=True,
        )

        self.model_creator = ModelCreator(**param_kwargs)

        np.random.seed(33)

        self.hyperparameters = self.model_creator.generate_hyperparameters()
        self.onchain_features = self.model_creator.generate_onchain_features()

        self.results, self.index_splits, self.target_series, self.adj_targets = (
            self.model_creator.create_model(
                max_trades=3,
                off_days=7,
                side=1,
                cutoff_point=5,
                **self.hyperparameters,
            )
        )

        self.model_metrics = ModelMetrics(
            train_in_middle=param_kwargs["train_in_middle"],
            index_splits=self.index_splits,
            results=self.results,
            test_index=four_years,
        )

    def test_generate_hyperparameters(self):
        assert isinstance(self.hyperparameters, dict)

        expected_hyperparams = {
            "iterations": 1000,
            "learning_rate": 0.92,
            "depth": 2,
            "min_child_samples": 1,
            "colsample_bylevel": 0.7799999999999997,
            "subsample": 0.12,
            "reg_lambda": 9,
            "use_best_model": True,
            "eval_metric": "Logloss",
            "random_seed": 23573,
            "silent": True,
        }

        self.assertDictEqual(
            expected_hyperparams, self.hyperparameters
        )

    def test_get_periods(self):
        expected_periods = (
            (pd.Timestamp("2012-01-02 00:00:00"), pd.Timestamp("2016-01-03 00:00:00")),
            (pd.Timestamp("2020-01-04 00:00:00"), pd.Timestamp("2024-12-01 00:00:00")),
        )

        test_periods = self.model_metrics.get_periods(self.index_splits)

        self.assertTupleEqual(
            expected_periods, test_periods
        )

    def test_calculate_model_recommendations(self):
        expected_test_buys, expected_val_buys = self.model_metrics.calculate_model_recommendations()

        test_buys = pd.Series([1037, 424], name="count", index=pd.Index([0, 1], name="Predict", dtype="int32"))
        val_buys = pd.Series([1223, 571], name="count", index=pd.Index([0, 1], name="Predict", dtype="int32"))

        pd.testing.assert_series_equal(
            expected_test_buys, test_buys
        )

        pd.testing.assert_series_equal(
            expected_val_buys, val_buys
        )

    def test_calculate_result_metrics(self):
        expected_result_metrics = {'expected_return_test': 1.0662190316698508,
        'expected_return_val': 0.5299277428729996,
        'precisions_test': 0.5531914893617021,
        'precisions_val': 0.5528169014084507,
        'precisions': (0.5531914893617021, 0.5528169014084507)}

        test_result_metrics = self.model_metrics.calculate_result_metrics(self.target_series, 7)

        self.assertDictEqual(expected_result_metrics, test_result_metrics)

    def test_calculate_drawdowns(self):
        expected_drawdowns = (
            0.5882578556729354,
            0.5095557389140951,
            0.3468939733483762,
            0.2941313190118774,
        )

        test_drawdowns = self.model_metrics.calculate_drawdowns()

        self.assertTupleEqual(expected_drawdowns, test_drawdowns)

    def test_calculate_result_ratios(self):
        expected_result_ratios = {
            "sharpe_test": 1.948,
            "sharpe_val": 0.19,
            "sortino_test": 0.235,
            "sortino_val": 0.396,
        }

        test_result_ratios = self.model_metrics.calculate_result_ratios()

        self.assertDictEqual(expected_result_ratios, test_result_ratios)

    def test_calculate_result_support(self):
        expected_result_support = (-0.026694045174537995, -0.055741360089186176)
        test_result_support = self.model_metrics.calculate_result_support(self.adj_targets, 1)

        self.assertTupleEqual(expected_result_support, test_result_support)

    def test_calculate_total_operations(self):
        test_buys = pd.Series([1037, 424], name="count", index=pd.Index([0, 1], name="Predict", dtype="int32"))
        val_buys = pd.Series([1223, 571], name="count", index=pd.Index([0, 1], name="Predict", dtype="int32"))

        expected_total_operations = (424, 571)
        expected_total_operations_pct = (0.6771617613506731, 0.9119324663472508)

        test_total_operations, test_total_operations_pct = self.model_metrics.calculate_total_operations(
            test_buys,
            val_buys,
            3,
            7,
            1,
        )

        self.assertTupleEqual(expected_total_operations, test_total_operations)
        self.assertTupleEqual(expected_total_operations_pct, test_total_operations_pct)

    def test_calculate_total_operations_too_high_warning(self):
        with warnings.catch_warnings(record=True) as w:
            test_buys = pd.Series(
                [1037, 424],
                name="count",
                index=pd.Index([0, 1], name="Predict", dtype="int32"),
            )
            val_buys = pd.Series(
                [1223, 571],
                name="count",
                index=pd.Index([0, 1], name="Predict", dtype="int32"),
            )

            self.model_metrics.test_index = 1000

            self.model_metrics.calculate_total_operations(
                test_buys,
                val_buys,
                3,
                7,
                1,
            )

            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert (
                "Total operations percentage is higher than 1"
                " (Test: 98.9333%) | (Val: 133.2333%)"
                in str(w[-1].message)
            )

    def test_calculate_total_operations_almost_too_high_warning(self):
        with warnings.catch_warnings(record=True) as w:
            test_buys = pd.Series(
                [1037, 424],
                name="count",
                index=pd.Index([0, 1], name="Predict", dtype="int32"),
            )
            val_buys = pd.Series(
                [1223, 571],
                name="count",
                index=pd.Index([0, 1], name="Predict", dtype="int32"),
            )

            self.model_metrics.test_index = 1500

            self.model_metrics.calculate_total_operations(
                test_buys,
                val_buys,
                3,
                7,
                1,
            )

            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert (
                    "Total operations percentage is higher than 0.8"
                    " (Test: 65.9556%) | (Val: 88.8222%)"
                in str(w[-1].message)
            )

    def test_calculate_ols_metrics(self):
        expected_ols_metrics = (0.49643, 0.734701, 4.16218e-07, 6.64961e-08)
        test_ols_metrics = self.model_metrics.calculate_ols_metrics()

        self.assertTupleEqual(expected_ols_metrics, test_ols_metrics)

    def test_calculate_ols_metrics_warning(self):
        with warnings.catch_warnings(record=True) as w:
            self.model_metrics.results_test.loc[:, "Liquid_Result"] = "test"
            self.model_metrics.results_val.loc[:, "Liquid_Result"] = "test"

            self.model_metrics.calculate_ols_metrics()

            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert (
                "TypeError : can't multiply sequence by non-int of type 'str'"
                in str(w[-1].message)
            )

    def test_set_results_test(self):
        expected_dataset = pd.DataFrame(np.random.rand(10, 10))
        self.model_metrics.set_results_test(expected_dataset)

        pd.testing.assert_frame_equal(
            expected_dataset,
            self.model_metrics.results_test
        )

    def test_set_results_val(self):
        expected_dataset = pd.DataFrame(np.random.rand(10, 10))
        self.model_metrics.set_results_val(expected_dataset)

        pd.testing.assert_frame_equal(
            expected_dataset,
            self.model_metrics.results_val
        )

    def test_set_results(self):
        expected_dataset = pd.DataFrame(np.random.rand(10, 10))
        self.model_metrics.set_results(expected_dataset)

        pd.testing.assert_frame_equal(
            expected_dataset,
            self.model_metrics.results
        )
