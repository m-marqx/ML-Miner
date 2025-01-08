import warnings

import pandas as pd

from machine_learning.ml_utils import DataHandler
from machine_learning.ols_metrics import OLSMetrics
from custom_exceptions.invalid_arguments import InvalidArgumentError
from utils import Statistics


class ModelMetrics:
    """
    A class to calculate and store various model performance metrics.

    The ModelMetrics class handles calculation of performance metrics
    for both test and validation periods, including returns, drawdowns,
    statistical ratios and operation counts.
    """

    def __init__(
        self,
        train_in_middle: bool,
        index_splits: dict[str, pd.Interval],
        results: pd.DataFrame,
        test_index: int,
    ):
        """
        Initialize the ModelMetrics object.

        Parameters
        ----------
        train_in_middle : bool
            Whether the training period is in the middle of the dataset.
        index_splits : dict[str, pd.Interval]
            Dictionary containing interval splits for
            train/test/validation.
        results : pd.DataFrame
            DataFrame containing model prediction results.
        test_index : int
            Length of the test period in days.
        """
        self.train_in_middle = train_in_middle
        self.test_index = test_index

        self.test_periods, self.val_periods = self.get_periods(index_splits)

        self.results = results

        if self.train_in_middle:
            self.results_test = results.loc[
                self.test_periods[0] : self.test_periods[1]
            ]
        else:
            self.results_test = results.loc[
                self.test_periods[0] : self.test_periods[1]
            ]

        self.results_val = results.loc[
            self.val_periods[0] : self.val_periods[1]
        ]

    def set_results_test(self, results_test: pd.DataFrame):
        """
        Set the test results DataFrame.

        Parameters
        ----------
        results_test : pd.DataFrame
            Updated test results DataFrame.

        Returns
        -------
        ModelMetrics
            The current instance with updated test results.
        """
        self.results_test = results_test
        return self

    def set_results_val(self, results_val: pd.DataFrame):
        """
        Set the validation results DataFrame.

        Parameters
        ----------
        results_val : pd.DataFrame
            Updated validation results DataFrame.

        Returns
        -------
        ModelMetrics
            The current instance with updated validation results.
        """
        self.results_val = results_val
        return self

    def set_results(self, results: pd.DataFrame):
        """
        Set the full results DataFrame.

        Parameters
        ----------
        results : pd.DataFrame
            Updated full results DataFrame.

        Returns
        -------
        ModelMetrics
            The current instance with updated full results.
        """
        self.results = results
        return self

    def get_periods(self, index_splits: dict[str, pd.Interval]) -> tuple[
        tuple[pd.Timestamp, pd.Timestamp],
        tuple[pd.Timestamp, pd.Timestamp]
    ]:
        """
        Extract test and validation periods from the provided index
        splits.

        Parameters
        ----------
        index_splits : dict[str, pd.Interval]
            Dictionary of time intervals for train/test/validation.

        Returns
        -------
        tuple
            A tuple of tuples containing the test and validation
            periods.
        """
        if not isinstance(index_splits, dict):
            raise InvalidArgumentError(
                "index_splits must be a pd.IntervalIndex"
            )
        if self.train_in_middle:
            test_periods = (
                index_splits["train"].left,
                index_splits["train"].right,
            )
        else:
            test_periods = (
                index_splits["test"].left,
                index_splits["test"].right,
            )

        val_periods = (
            index_splits["validation"].left,
            index_splits["validation"].right,
        )

        return test_periods, val_periods

