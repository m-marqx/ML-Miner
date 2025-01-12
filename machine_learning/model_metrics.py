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

    def calculate_model_recommendations(self) -> tuple[pd.Series, pd.Series]:
        """
        Count the model recommendations (Predict values) in test and
        validation sets.

        Returns
        -------
        tuple
            A tuple containing two Series objects for test and
            validation buys.
        """
        test_buys = self.results_test["Predict"].value_counts()
        val_buys = self.results_val["Predict"].value_counts()
        return test_buys, val_buys

    def calculate_result_metrics(
        self,
        target_series: pd.Series,
        target_shift: int,
    ) -> dict[str, float | tuple[float, float]]:
        """
        Compute various performance metrics like precision and expected
        return.

        Parameters
        ----------
        target_series : pd.Series
            The target series for evaluation.
        target_shift : int
            Shift to apply on the validation set.

        Returns
        -------
        dict
            Dictionary containing precision and expected return metrics.
        """
        y_test = target_series.loc[self.test_periods[0] : self.test_periods[1]]
        y_val = target_series.loc[self.val_periods[0] : self.val_periods[1]][
            :-target_shift
        ]

        y_pred_test = (
            self.results[["Predict"]]
            .loc[self.test_periods[0] : self.test_periods[1]]
            .query("Predict != 0")
            .where(self.results["Predict"] == 1, 0)
        )

        y_pred_val = (
            self.results[["Predict"]]
            .loc[self.val_periods[0] : self.val_periods[1]][:-target_shift]
            .query("Predict != 0")
            .where(self.results["Predict"] == 1, 0)
        )

        y_test_adj = y_test.reindex(y_pred_test.index)
        y_val_adj = y_val.reindex(y_pred_val.index)

        result_metrics_test = DataHandler(
            self.results.reindex(y_test_adj.index)
        ).result_metrics(
            "Liquid_Result",
            is_percentage_data=True,
            output_format="Series",
        )

        result_metrics_val = DataHandler(
            self.results.reindex(y_val_adj.index)
        ).result_metrics(
            "Liquid_Result",
            is_percentage_data=True,
            output_format="Series",
        )

        precisions = (
            result_metrics_test["Win_Rate"],
            result_metrics_val["Win_Rate"],
        )

        expected_return = (
            result_metrics_test["Expected_Return"],
            result_metrics_val["Expected_Return"],
        )

        return {
            "expected_return_test": expected_return[0],
            "expected_return_val": expected_return[1],
            "precisions_test": precisions[0],
            "precisions_val": precisions[1],
            "precisions": precisions,
        }

    def calculate_drawdowns(self) -> tuple[float, float, float, float]:
        """
        Calculate drawdowns for both raw and adjusted liquid returns.

        Returns
        -------
        tuple
            Tuple of drawdowns at the 0.95 quantile for test and
            validation sets.
        """
        liquid_return_test = self.results_test["Liquid_Result"].cumprod()
        liquid_return_val = self.results_val["Liquid_Result"].cumprod()

        liquid_return_adj_test = self.results_test[
            "Liquid_Result_pct_adj"
        ].cumprod()

        liquid_return_adj_val = self.results_val[
            "Liquid_Result_pct_adj"
        ].cumprod()

        drawdown_full_test = (
            liquid_return_test.cummax() - liquid_return_test
        ) / liquid_return_test.cummax()

        drawdown_full_val = (
            liquid_return_val.cummax() - liquid_return_val
        ) / liquid_return_val.cummax()

        drawdown_adj_test = (
            liquid_return_adj_test.cummax() - liquid_return_adj_test
        ) / liquid_return_adj_test.cummax()

        drawdown_adj_val = (
            liquid_return_adj_val.cummax() - liquid_return_adj_val
        ) / liquid_return_adj_val.cummax()

        return (
            drawdown_full_test.quantile(0.95),
            drawdown_full_val.quantile(0.95),
            drawdown_adj_test.quantile(0.95),
            drawdown_adj_val.quantile(0.95),
        )

    def calculate_result_ratios(self) -> dict[str, float]:
        """
        Compute performance ratios like Sharpe and Sortino for test and
        validation sets.

        Returns
        -------
        dict
            Dictionary containing Sharpe and Sortino ratios for test and
            validation.
        """
        _, sharpe_test, sortino_test = (
            Statistics(
                dataframe=(
                    self.results_test["Liquid_Result"] - 1
                ).drop_duplicates(),
                time_span="Y",
                risk_free_rate=(1.12) ** (1 / 365.25) - 1,
                is_percent=True,
            )
            .calculate_all_statistics()
            .mean()
        )

        _, sharpe_val, sortino_val = (
            Statistics(
                dataframe=(
                    self.results_val["Liquid_Result"] - 1
                ).drop_duplicates(),
                time_span="Y",
                risk_free_rate=(1.12) ** (1 / 365.25) - 1,
                is_percent=True,
            )
            .calculate_all_statistics()
            .mean()
        )

        return {
            "sharpe_test": sharpe_test,
            "sharpe_val": sharpe_val,
            "sortino_test": sortino_test,
            "sortino_val": sortino_val,
        }

