import unittest
import pathlib
import pandas as pd
import numpy as np
from machine_learning.model_builder import calculate_model
from pandas import Timestamp, Interval


class TestCalculateModelGeneral(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(33)
        dates = pd.date_range("2020-01-01", periods=100)

        self.dataframe = pd.DataFrame(
            {
                "open": self.rng.random(100),
                "high": self.rng.random(100),
                "low": self.rng.random(100),
                "close": self.rng.random(100),
                "var_1": self.rng.integers(0, 10, 100),
                "var_2": self.rng.integers(0, 5, 100),
                "var_3": self.rng.integers(0, 2, 100),
            },
            index=dates,
        )

        self.dataframe["Target"] = self.dataframe["close"].shift(-1)
        self.dataframe["Target_bin"] = (
            self.dataframe["Target"] > self.dataframe["close"]
        )

        self.features = ["var_1", "var_2", "var_3"]
        self.test_index = 33

        (
            self.mh2,
            self.best_model,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.y_pred_train,
            self.y_pred_test,
            self.all_x,
            self.all_y,
            self.index_splits,
        ) = calculate_model(
            dataset=self.dataframe,
            feats=self.features,
            test_index=self.test_index,
            output="All",
            train_in_middle=False,
        )

    def test_calculate_model_X_train(self):
        expected_result = (
            self.dataframe[self.features]
            .iloc[: self.test_index]
            .astype("int8")
        )

        pd.testing.assert_frame_equal(self.X_train, expected_result)

    def test_calculate_model_X_test(self):
        expected_result = (
            self.dataframe[self.features]
            .iloc[self.test_index : self.test_index * 2]
            .astype("int8")
        )

        pd.testing.assert_frame_equal(self.X_test, expected_result)

    def test_calculate_model_y_train(self):
        expected_result = (
            self.dataframe[["Target_bin"]]
            .iloc[: self.test_index]
            .astype("boolean")
        )

        pd.testing.assert_frame_equal(self.y_train, expected_result)

    def test_calculate_model_y_test(self):
        expected_result = (
            self.dataframe[["Target_bin"]]
            .iloc[self.test_index : self.test_index * 2]
            .astype("boolean")
        )

        pd.testing.assert_frame_equal(self.y_test, expected_result)

    def test_calculate_X_val(self):
        expected_result = pd.DataFrame(
            {
                "var_1": {
                    Timestamp("2020-03-07 00:00:00"): 3,
                    Timestamp("2020-03-08 00:00:00"): 3,
                    Timestamp("2020-03-09 00:00:00"): 8,
                    Timestamp("2020-03-10 00:00:00"): 6,
                    Timestamp("2020-03-11 00:00:00"): 9,
                    Timestamp("2020-03-12 00:00:00"): 6,
                    Timestamp("2020-03-13 00:00:00"): 4,
                    Timestamp("2020-03-14 00:00:00"): 4,
                    Timestamp("2020-03-15 00:00:00"): 2,
                    Timestamp("2020-03-16 00:00:00"): 0,
                    Timestamp("2020-03-17 00:00:00"): 8,
                    Timestamp("2020-03-18 00:00:00"): 1,
                    Timestamp("2020-03-19 00:00:00"): 3,
                    Timestamp("2020-03-20 00:00:00"): 5,
                    Timestamp("2020-03-21 00:00:00"): 4,
                    Timestamp("2020-03-22 00:00:00"): 6,
                    Timestamp("2020-03-23 00:00:00"): 8,
                    Timestamp("2020-03-24 00:00:00"): 7,
                    Timestamp("2020-03-25 00:00:00"): 0,
                    Timestamp("2020-03-26 00:00:00"): 3,
                    Timestamp("2020-03-27 00:00:00"): 2,
                    Timestamp("2020-03-28 00:00:00"): 0,
                    Timestamp("2020-03-29 00:00:00"): 9,
                    Timestamp("2020-03-30 00:00:00"): 3,
                    Timestamp("2020-03-31 00:00:00"): 1,
                    Timestamp("2020-04-01 00:00:00"): 3,
                    Timestamp("2020-04-02 00:00:00"): 8,
                    Timestamp("2020-04-03 00:00:00"): 0,
                    Timestamp("2020-04-04 00:00:00"): 4,
                    Timestamp("2020-04-05 00:00:00"): 3,
                    Timestamp("2020-04-06 00:00:00"): 0,
                    Timestamp("2020-04-07 00:00:00"): 9,
                    Timestamp("2020-04-08 00:00:00"): 3,
                    Timestamp("2020-04-09 00:00:00"): 8,
                },
                "var_2": {
                    Timestamp("2020-03-07 00:00:00"): 4,
                    Timestamp("2020-03-08 00:00:00"): 3,
                    Timestamp("2020-03-09 00:00:00"): 3,
                    Timestamp("2020-03-10 00:00:00"): 3,
                    Timestamp("2020-03-11 00:00:00"): 2,
                    Timestamp("2020-03-12 00:00:00"): 1,
                    Timestamp("2020-03-13 00:00:00"): 1,
                    Timestamp("2020-03-14 00:00:00"): 2,
                    Timestamp("2020-03-15 00:00:00"): 2,
                    Timestamp("2020-03-16 00:00:00"): 3,
                    Timestamp("2020-03-17 00:00:00"): 3,
                    Timestamp("2020-03-18 00:00:00"): 2,
                    Timestamp("2020-03-19 00:00:00"): 2,
                    Timestamp("2020-03-20 00:00:00"): 2,
                    Timestamp("2020-03-21 00:00:00"): 2,
                    Timestamp("2020-03-22 00:00:00"): 4,
                    Timestamp("2020-03-23 00:00:00"): 2,
                    Timestamp("2020-03-24 00:00:00"): 1,
                    Timestamp("2020-03-25 00:00:00"): 3,
                    Timestamp("2020-03-26 00:00:00"): 0,
                    Timestamp("2020-03-27 00:00:00"): 2,
                    Timestamp("2020-03-28 00:00:00"): 2,
                    Timestamp("2020-03-29 00:00:00"): 4,
                    Timestamp("2020-03-30 00:00:00"): 2,
                    Timestamp("2020-03-31 00:00:00"): 1,
                    Timestamp("2020-04-01 00:00:00"): 4,
                    Timestamp("2020-04-02 00:00:00"): 2,
                    Timestamp("2020-04-03 00:00:00"): 4,
                    Timestamp("2020-04-04 00:00:00"): 3,
                    Timestamp("2020-04-05 00:00:00"): 0,
                    Timestamp("2020-04-06 00:00:00"): 2,
                    Timestamp("2020-04-07 00:00:00"): 0,
                    Timestamp("2020-04-08 00:00:00"): 2,
                    Timestamp("2020-04-09 00:00:00"): 3,
                },
                "var_3": {
                    Timestamp("2020-03-07 00:00:00"): 1,
                    Timestamp("2020-03-08 00:00:00"): 1,
                    Timestamp("2020-03-09 00:00:00"): 1,
                    Timestamp("2020-03-10 00:00:00"): 1,
                    Timestamp("2020-03-11 00:00:00"): 1,
                    Timestamp("2020-03-12 00:00:00"): 0,
                    Timestamp("2020-03-13 00:00:00"): 1,
                    Timestamp("2020-03-14 00:00:00"): 0,
                    Timestamp("2020-03-15 00:00:00"): 0,
                    Timestamp("2020-03-16 00:00:00"): 1,
                    Timestamp("2020-03-17 00:00:00"): 0,
                    Timestamp("2020-03-18 00:00:00"): 0,
                    Timestamp("2020-03-19 00:00:00"): 0,
                    Timestamp("2020-03-20 00:00:00"): 0,
                    Timestamp("2020-03-21 00:00:00"): 1,
                    Timestamp("2020-03-22 00:00:00"): 0,
                    Timestamp("2020-03-23 00:00:00"): 1,
                    Timestamp("2020-03-24 00:00:00"): 1,
                    Timestamp("2020-03-25 00:00:00"): 0,
                    Timestamp("2020-03-26 00:00:00"): 1,
                    Timestamp("2020-03-27 00:00:00"): 0,
                    Timestamp("2020-03-28 00:00:00"): 1,
                    Timestamp("2020-03-29 00:00:00"): 1,
                    Timestamp("2020-03-30 00:00:00"): 1,
                    Timestamp("2020-03-31 00:00:00"): 1,
                    Timestamp("2020-04-01 00:00:00"): 0,
                    Timestamp("2020-04-02 00:00:00"): 1,
                    Timestamp("2020-04-03 00:00:00"): 0,
                    Timestamp("2020-04-04 00:00:00"): 1,
                    Timestamp("2020-04-05 00:00:00"): 0,
                    Timestamp("2020-04-06 00:00:00"): 0,
                    Timestamp("2020-04-07 00:00:00"): 1,
                    Timestamp("2020-04-08 00:00:00"): 1,
                    Timestamp("2020-04-09 00:00:00"): 0,
                },
            },
            dtype="int8",
        )

        expected_result.index.freq = "D"

        result = self.all_x.iloc[self.test_index * 2 :]
        pd.testing.assert_frame_equal(result, expected_result)

    def test_calculate_y_val(self):
        expected_result = pd.DataFrame(
            {
                "Target_bin": {
                    Timestamp("2020-03-07 00:00:00"): False,
                    Timestamp("2020-03-08 00:00:00"): False,
                    Timestamp("2020-03-09 00:00:00"): True,
                    Timestamp("2020-03-10 00:00:00"): False,
                    Timestamp("2020-03-11 00:00:00"): False,
                    Timestamp("2020-03-12 00:00:00"): True,
                    Timestamp("2020-03-13 00:00:00"): False,
                    Timestamp("2020-03-14 00:00:00"): False,
                    Timestamp("2020-03-15 00:00:00"): True,
                    Timestamp("2020-03-16 00:00:00"): True,
                    Timestamp("2020-03-17 00:00:00"): False,
                    Timestamp("2020-03-18 00:00:00"): True,
                    Timestamp("2020-03-19 00:00:00"): True,
                    Timestamp("2020-03-20 00:00:00"): False,
                    Timestamp("2020-03-21 00:00:00"): False,
                    Timestamp("2020-03-22 00:00:00"): False,
                    Timestamp("2020-03-23 00:00:00"): True,
                    Timestamp("2020-03-24 00:00:00"): False,
                    Timestamp("2020-03-25 00:00:00"): True,
                    Timestamp("2020-03-26 00:00:00"): False,
                    Timestamp("2020-03-27 00:00:00"): True,
                    Timestamp("2020-03-28 00:00:00"): True,
                    Timestamp("2020-03-29 00:00:00"): False,
                    Timestamp("2020-03-30 00:00:00"): True,
                    Timestamp("2020-03-31 00:00:00"): False,
                    Timestamp("2020-04-01 00:00:00"): False,
                    Timestamp("2020-04-02 00:00:00"): True,
                    Timestamp("2020-04-03 00:00:00"): False,
                    Timestamp("2020-04-04 00:00:00"): True,
                    Timestamp("2020-04-05 00:00:00"): False,
                    Timestamp("2020-04-06 00:00:00"): True,
                    Timestamp("2020-04-07 00:00:00"): False,
                    Timestamp("2020-04-08 00:00:00"): True,
                    Timestamp("2020-04-09 00:00:00"): False,
                }
            },
            dtype="boolean",
        )

        expected_result.index.freq = "D"

        result = self.all_y.iloc[self.test_index * 2 :]

        pd.testing.assert_frame_equal(result, expected_result)

    def test_calculate_index_splits(self):
        start_train_date = pd.Timestamp("2020-01-01")
        end_train_date = pd.Timestamp("2020-02-02")
        start_test_date = pd.Timestamp("2020-02-03")
        end_test_date = pd.Timestamp("2020-03-06")
        start_val_date = pd.Timestamp("2020-03-07")
        end_val_date = pd.Timestamp("2020-04-09")

        expected_result = {
            "train": Interval(
                start_train_date, end_train_date, closed="right"
            ),
            "test": Interval(start_test_date, end_test_date, closed="right"),
            "validation": Interval(
                start_val_date, end_val_date, closed="right"
            ),
        }

        self.assertDictEqual(self.index_splits, expected_result)

    def test_calculate_model_output_all(self):
        expected_result = (
            self.mh2,
            self.best_model,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.y_pred_train,
            self.y_pred_test,
            self.all_x,
            self.all_y,
            self.index_splits,
        )

        results = calculate_model(
            dataset=self.dataframe,
            feats=self.features,
            test_index=self.test_index,
            output="All",
            train_in_middle=False,
        )

        for expected, result in zip(expected_result, results):
            if isinstance(expected, pd.DataFrame):
                pd.testing.assert_frame_equal(expected, result)
            elif isinstance(expected, pd.Series):
                pd.testing.assert_series_equal(expected, result)
            else:
                self.assertEqual(expected, result)

    def test_calculate_model_output_return(self):
        expected_result = self.mh2, self.index_splits

        results = calculate_model(
            dataset=self.dataframe,
            feats=self.features,
            test_index=self.test_index,
            output="Return",
            train_in_middle=False,
        )

        for expected, result in zip(expected_result, results):
            if isinstance(expected, pd.DataFrame):
                pd.testing.assert_frame_equal(expected, result)
            elif isinstance(expected, pd.Series):
                pd.testing.assert_series_equal(expected, result)
            else:
                self.assertEqual(expected, result)

    def test_calculate_model_output_error(self):
        with self.assertRaises(ValueError):
            calculate_model(
                dataset=self.dataframe,
                feats=self.features,
                test_index=self.test_index,
                output="Error",
                train_in_middle=False,
            )

    def test_calculate_model_output_model(self):
        expected_result = self.best_model

        result = calculate_model(
            dataset=self.dataframe,
            feats=self.features,
            test_index=self.test_index,
            output="Model",
            train_in_middle=False,
        )

        self.assertEqual(result, expected_result)

    def test_calculate_model_output_dataset(self):
        expected_result = (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.y_pred_train,
            self.y_pred_test,
            self.all_x,
            self.all_y,
        )

        results = calculate_model(
            dataset=self.dataframe,
            feats=self.features,
            test_index=self.test_index,
            output="Dataset",
            train_in_middle=False,
        )

        for expected, result in zip(expected_result, results):
            if isinstance(expected, pd.DataFrame):
                pd.testing.assert_frame_equal(expected, result)
            elif isinstance(expected, pd.Series):
                pd.testing.assert_series_equal(expected, result)
            else:
                self.assertEqual(expected, result)


class TestCalculateModelStartTrain(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(33)
        dates = pd.date_range("2020-01-01", periods=100)

        self.dataframe = pd.DataFrame(
            {
                "open": self.rng.random(100),
                "high": self.rng.random(100),
                "low": self.rng.random(100),
                "close": self.rng.random(100),
                "var_1": self.rng.integers(0, 10, 100),
                "var_2": self.rng.integers(0, 5, 100),
                "var_3": self.rng.integers(0, 2, 100),
            },
            index=dates,
        )

        self.dataframe["Target"] = self.dataframe["close"].shift(-1)
        self.dataframe["Target_bin"] = (
            self.dataframe["Target"] > self.dataframe["close"]
        )

        self.features = ["var_1", "var_2", "var_3"]
        self.test_index = 33

        (
            self.mh2,
            _,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.y_pred_train,
            self.y_pred_test,
            self.all_x,
            self.all_y,
            self.index_splits,
        ) = calculate_model(
            dataset=self.dataframe,
            feats=self.features,
            test_index=self.test_index,
            output="All",
            train_in_middle=False,
        )

    def test_calculate_model_model_returns(self):
        expected_result_path = pathlib.Path(
            "machine_learning",
            "tests",
            "test_model_returns.parquet",
        )
        expected_result = pd.read_parquet(expected_result_path)
        expected_result.index.freq = "D"

        pd.testing.assert_frame_equal(self.mh2, expected_result)

    def test_calculate_y_pred_train(self):
        expected_result = pd.Series(
            {
                Timestamp("2020-01-01 00:00:00"): 1,
                Timestamp("2020-01-02 00:00:00"): 1,
                Timestamp("2020-01-03 00:00:00"): 1,
                Timestamp("2020-01-04 00:00:00"): 1,
                Timestamp("2020-01-05 00:00:00"): 1,
                Timestamp("2020-01-06 00:00:00"): 1,
                Timestamp("2020-01-07 00:00:00"): 1,
                Timestamp("2020-01-08 00:00:00"): 0,
                Timestamp("2020-01-09 00:00:00"): 1,
                Timestamp("2020-01-10 00:00:00"): 1,
                Timestamp("2020-01-11 00:00:00"): 1,
                Timestamp("2020-01-12 00:00:00"): 0,
                Timestamp("2020-01-13 00:00:00"): 1,
                Timestamp("2020-01-14 00:00:00"): 1,
                Timestamp("2020-01-15 00:00:00"): 1,
                Timestamp("2020-01-16 00:00:00"): 1,
                Timestamp("2020-01-17 00:00:00"): 0,
                Timestamp("2020-01-18 00:00:00"): 1,
                Timestamp("2020-01-19 00:00:00"): 0,
                Timestamp("2020-01-20 00:00:00"): 1,
                Timestamp("2020-01-21 00:00:00"): 1,
                Timestamp("2020-01-22 00:00:00"): 0,
                Timestamp("2020-01-23 00:00:00"): 1,
                Timestamp("2020-01-24 00:00:00"): 1,
                Timestamp("2020-01-25 00:00:00"): 0,
                Timestamp("2020-01-26 00:00:00"): 0,
                Timestamp("2020-01-27 00:00:00"): 1,
                Timestamp("2020-01-28 00:00:00"): 1,
                Timestamp("2020-01-29 00:00:00"): 0,
                Timestamp("2020-01-30 00:00:00"): 0,
                Timestamp("2020-01-31 00:00:00"): 1,
                Timestamp("2020-02-01 00:00:00"): 0,
                Timestamp("2020-02-02 00:00:00"): 1,
            },
            name=1,
            dtype="int32",
        )

        expected_result.index.freq = "D"

        pd.testing.assert_series_equal(self.y_pred_train, expected_result)

    def test_calculate_y_pred_test(self):
        expected_result = pd.Series(
            {
                Timestamp("2020-02-03 00:00:00"): 0,
                Timestamp("2020-02-04 00:00:00"): 1,
                Timestamp("2020-02-05 00:00:00"): 1,
                Timestamp("2020-02-06 00:00:00"): 1,
                Timestamp("2020-02-07 00:00:00"): 0,
                Timestamp("2020-02-08 00:00:00"): 0,
                Timestamp("2020-02-09 00:00:00"): 1,
                Timestamp("2020-02-10 00:00:00"): 1,
                Timestamp("2020-02-11 00:00:00"): 1,
                Timestamp("2020-02-12 00:00:00"): 0,
                Timestamp("2020-02-13 00:00:00"): 0,
                Timestamp("2020-02-14 00:00:00"): 1,
                Timestamp("2020-02-15 00:00:00"): 1,
                Timestamp("2020-02-16 00:00:00"): 1,
                Timestamp("2020-02-17 00:00:00"): 1,
                Timestamp("2020-02-18 00:00:00"): 1,
                Timestamp("2020-02-19 00:00:00"): 1,
                Timestamp("2020-02-20 00:00:00"): 0,
                Timestamp("2020-02-21 00:00:00"): 0,
                Timestamp("2020-02-22 00:00:00"): 1,
                Timestamp("2020-02-23 00:00:00"): 0,
                Timestamp("2020-02-24 00:00:00"): 1,
                Timestamp("2020-02-25 00:00:00"): 1,
                Timestamp("2020-02-26 00:00:00"): 0,
                Timestamp("2020-02-27 00:00:00"): 0,
                Timestamp("2020-02-28 00:00:00"): 1,
                Timestamp("2020-02-29 00:00:00"): 1,
                Timestamp("2020-03-01 00:00:00"): 0,
                Timestamp("2020-03-02 00:00:00"): 0,
                Timestamp("2020-03-03 00:00:00"): 0,
                Timestamp("2020-03-04 00:00:00"): 1,
                Timestamp("2020-03-05 00:00:00"): 0,
                Timestamp("2020-03-06 00:00:00"): 1,
            },
            name=1,
            dtype="int32",
        )

        expected_result.index.freq = "D"

        pd.testing.assert_series_equal(self.y_pred_test, expected_result)


class TestCalculateModelMiddleTrain(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(33)
        dates = pd.date_range("2020-01-01", periods=100)

        self.dataframe = pd.DataFrame(
            {
                "open": self.rng.random(100),
                "high": self.rng.random(100),
                "low": self.rng.random(100),
                "close": self.rng.random(100),
                "var_1": self.rng.integers(0, 10, 100),
                "var_2": self.rng.integers(0, 5, 100),
                "var_3": self.rng.integers(0, 2, 100),
            },
            index=dates,
        )

        self.dataframe["Target"] = self.dataframe["close"].shift(-1)
        self.dataframe["Target_bin"] = (
            self.dataframe["Target"] > self.dataframe["close"]
        )

        self.features = ["var_1", "var_2", "var_3"]
        self.test_index = 33

        (
            self.mh2,
            _,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.y_pred_train,
            self.y_pred_test,
            self.all_x,
            self.all_y,
            self.index_splits,
        ) = calculate_model(
            dataset=self.dataframe,
            feats=self.features,
            test_index=self.test_index,
            output="All",
            train_in_middle=True,
        )

    def test_calculate_model_model_returns(self):
        expected_result_path = pathlib.Path(
            "machine_learning",
            "tests",
            "test_model_returns_middle_train.parquet",
        )

        expected_result = pd.read_parquet(expected_result_path)
        expected_result.index.freq = "D"

        pd.testing.assert_frame_equal(self.mh2, expected_result)

    def test_calculate_y_pred_train(self):
        expected_result = pd.Series(
            {
                Timestamp("2020-01-01 00:00:00"): 0,
                Timestamp("2020-01-02 00:00:00"): 1,
                Timestamp("2020-01-03 00:00:00"): 0,
                Timestamp("2020-01-04 00:00:00"): 0,
                Timestamp("2020-01-05 00:00:00"): 1,
                Timestamp("2020-01-06 00:00:00"): 0,
                Timestamp("2020-01-07 00:00:00"): 1,
                Timestamp("2020-01-08 00:00:00"): 1,
                Timestamp("2020-01-09 00:00:00"): 0,
                Timestamp("2020-01-10 00:00:00"): 0,
                Timestamp("2020-01-11 00:00:00"): 1,
                Timestamp("2020-01-12 00:00:00"): 1,
                Timestamp("2020-01-13 00:00:00"): 1,
                Timestamp("2020-01-14 00:00:00"): 0,
                Timestamp("2020-01-15 00:00:00"): 0,
                Timestamp("2020-01-16 00:00:00"): 0,
                Timestamp("2020-01-17 00:00:00"): 1,
                Timestamp("2020-01-18 00:00:00"): 1,
                Timestamp("2020-01-19 00:00:00"): 0,
                Timestamp("2020-01-20 00:00:00"): 0,
                Timestamp("2020-01-21 00:00:00"): 1,
                Timestamp("2020-01-22 00:00:00"): 1,
                Timestamp("2020-01-23 00:00:00"): 0,
                Timestamp("2020-01-24 00:00:00"): 0,
                Timestamp("2020-01-25 00:00:00"): 0,
                Timestamp("2020-01-26 00:00:00"): 1,
                Timestamp("2020-01-27 00:00:00"): 1,
                Timestamp("2020-01-28 00:00:00"): 0,
                Timestamp("2020-01-29 00:00:00"): 1,
                Timestamp("2020-01-30 00:00:00"): 1,
                Timestamp("2020-01-31 00:00:00"): 0,
                Timestamp("2020-02-01 00:00:00"): 0,
                Timestamp("2020-02-02 00:00:00"): 0,
            },
            name=1,
            dtype="int32",
        )

        expected_result.index.freq = "D"

        pd.testing.assert_series_equal(self.y_pred_train, expected_result)

    def test_calculate_y_pred_test(self):
        expected_result = pd.Series(
            {
                Timestamp("2020-02-03 00:00:00"): 1,
                Timestamp("2020-02-04 00:00:00"): 0,
                Timestamp("2020-02-05 00:00:00"): 0,
                Timestamp("2020-02-06 00:00:00"): 1,
                Timestamp("2020-02-07 00:00:00"): 0,
                Timestamp("2020-02-08 00:00:00"): 1,
                Timestamp("2020-02-09 00:00:00"): 0,
                Timestamp("2020-02-10 00:00:00"): 0,
                Timestamp("2020-02-11 00:00:00"): 0,
                Timestamp("2020-02-12 00:00:00"): 1,
                Timestamp("2020-02-13 00:00:00"): 1,
                Timestamp("2020-02-14 00:00:00"): 0,
                Timestamp("2020-02-15 00:00:00"): 0,
                Timestamp("2020-02-16 00:00:00"): 0,
                Timestamp("2020-02-17 00:00:00"): 1,
                Timestamp("2020-02-18 00:00:00"): 0,
                Timestamp("2020-02-19 00:00:00"): 0,
                Timestamp("2020-02-20 00:00:00"): 1,
                Timestamp("2020-02-21 00:00:00"): 1,
                Timestamp("2020-02-22 00:00:00"): 1,
                Timestamp("2020-02-23 00:00:00"): 0,
                Timestamp("2020-02-24 00:00:00"): 0,
                Timestamp("2020-02-25 00:00:00"): 0,
                Timestamp("2020-02-26 00:00:00"): 1,
                Timestamp("2020-02-27 00:00:00"): 0,
                Timestamp("2020-02-28 00:00:00"): 0,
                Timestamp("2020-02-29 00:00:00"): 0,
                Timestamp("2020-03-01 00:00:00"): 1,
                Timestamp("2020-03-02 00:00:00"): 1,
                Timestamp("2020-03-03 00:00:00"): 1,
                Timestamp("2020-03-04 00:00:00"): 0,
                Timestamp("2020-03-05 00:00:00"): 1,
                Timestamp("2020-03-06 00:00:00"): 1,
            },
            name=1,
            dtype="int32",
        )

        expected_result.index.freq = "D"

        pd.testing.assert_series_equal(self.y_pred_test, expected_result)

class TestCalculateModelCutoff(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(33)
        dates = pd.date_range("2020-01-01", periods=100)

        self.dataframe = pd.DataFrame(
            {
                "open": self.rng.random(100),
                "high": self.rng.random(100),
                "low": self.rng.random(100),
                "close": self.rng.random(100),
                "var_1": self.rng.integers(0, 10, 100),
                "var_2": self.rng.integers(0, 5, 100),
                "var_3": self.rng.integers(0, 2, 100),
            },
            index=dates,
        )

        self.dataframe["Target"] = self.dataframe["close"].shift(-1)
        self.dataframe["Target_bin"] = (
            self.dataframe["Target"] > self.dataframe["close"]
        )

        self.features = ["var_1", "var_2", "var_3"]
        self.test_index = 33

        (
            self.mh2,
            _,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.y_pred_train,
            self.y_pred_test,
            self.all_x,
            self.all_y,
            self.index_splits,
        ) = calculate_model(
            dataset=self.dataframe,
            feats=self.features,
            test_index=self.test_index,
            output="All",
            train_in_middle=False,
            cutoff_point=10,
        )

    def test_calculate_model_model_returns(self):
        expected_result_path = pathlib.Path(
            "machine_learning",
            "tests",
            "test_model_cutoff.parquet",
        )
        expected_result = pd.read_parquet(expected_result_path)
        expected_result.index.freq = "D"

        pd.testing.assert_frame_equal(self.mh2, expected_result)

    def test_calculate_model_error_over_hundred(self):
        with self.assertRaises(ValueError):
            calculate_model(
                dataset=self.dataframe,
                feats=self.features,
                test_index=self.test_index,
                output="All",
                train_in_middle=False,
                cutoff_point=101,
            )

    def test_calculate_model_error_below_zero(self):
        with self.assertRaises(ValueError):
            calculate_model(
                dataset=self.dataframe,
                feats=self.features,
                test_index=self.test_index,
                output="All",
                train_in_middle=False,
                cutoff_point=-1,
            )
