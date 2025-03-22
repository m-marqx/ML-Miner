from typing import Literal

import numpy as np
import pandas as pd

class DataHandler:
    """
    Class for handling data preprocessing tasks.

    Parameters
    ----------
    dataframe : pd.DataFrame or pd.Series
        The input DataFrame or Series to be processed.

    Attributes
    ----------
    data_frame : pd.DataFrame
        The processed DataFrame.

    Methods
    -------
    get_datasets(feature_columns, test_size=0.5, split_size=0.7)
        Splits the data into development and validation datasets.
    drop_zero_predictions(column)
        Drops rows where the specified column has all zero values.
    get_splits(target, features)
        Splits the DataFrame into training and testing sets.
    get_best_results(target_column)
        Gets the rows with the best accuracy for each unique value in
        the target column.
    result_metrics(result_column=None, is_percentage_data=False,
    output_format="DataFrame")
        Calculates result-related statistics like expected return and win rate.
    fill_outlier(column=None, iqr_scale=1.5, upper_quantile=0.75,
    down_quantile=0.25)
        Removes outliers from a specified column using the IQR method.
    quantile_split(target_input, column=None, method="ratio",
    quantiles=None, log_values=False)
        Splits data into quantiles and analyzes the relationship with a target.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame | pd.Series | np.ndarray,
    ) -> None:
        """
        Initialize the DataHandler object.

        Parameters
        ----------
        dataframe : pd.DataFrame, pd.Series, or np.ndarray
            The input data to be processed. It can be a pandas DataFrame,
            Series, or a numpy array.

        """
        self.data_frame = dataframe.copy()

        if isinstance(dataframe, np.ndarray):
            self.data_frame = pd.Series(dataframe)

    def calculate_targets(self, length=1) -> pd.DataFrame:
        """
        Calculate target variables for binary classification.

        Adds target variables to the DataFrame based on the 'close'
        column:
        - 'Return': Percentage change in 'close' from the previous day.
        - 'Target': Shifted 'Return', representing the future day's
        return.
        - 'Target_bin': Binary classification of 'Target':
            - 1 if 'Target' > 1 (positive return)
            - 0 otherwise.

        Returns
        -------
        pd.DataFrame
            DataFrame with added target variables.

        """
        if isinstance(self.data_frame, pd.Series):
            self.data_frame = pd.DataFrame(self.data_frame)

        self.data_frame['Return'] = self.data_frame["close"].pct_change(length) + 1
        self.data_frame["Target"] = self.data_frame["Return"].shift(-length)
        self.data_frame["Target_bin"] = np.where(
            self.data_frame["Target"] > 1,
            1, 0)

        self.data_frame["Target_bin"] = np.where(
            self.data_frame['Target'].isna(),
            np.nan, self.data_frame['Target_bin']
        )
        return self.data_frame

    def result_metrics(
        self,
        result_column: str | None = None,
        is_percentage_data: bool = False,
        output_format: Literal["dict", "Series", "DataFrame"] = "DataFrame",
    ) -> dict[str, float] | pd.Series | pd.DataFrame:
        """
        Calculate various statistics related to results, including
        expected return, win rate, positive and negative means, and
        payoff ratio.

        Parameters
        ----------
        result_column : str, optional
            The name of the column containing the results (returns) for
            analysis.
            If None, the instance's data_frame will be used as the
            result column.
            (default: None).
        is_percentage_data : bool, optional
            Indicates whether the data represents percentages.
            (default: False).
        output_format : Literal["dict", "Series", "DataFrame"], optional
            The format of the output. Choose from 'dict', 'Series', or
            'DataFrame'
            (default: 'DataFrame').

        Returns
        -------
        Returns the calculated statistics in the specified format:
        - If output_format is `'dict'`, a dictionary with keys:
            - 'Expected_Return': float
                The expected return based on the provided result
                column.
            - 'Win_Rate': float
                The win rate (percentage of positive outcomes) of
                the model.
            - 'Positive_Mean': float
                The mean return of positive outcomes from the
                model.
            - 'Negative_Mean': float
                The mean return of negative outcomes from the
                model.
            - 'Payoff': float
                The payoff ratio, calculated as the positive mean
                divided by the absolute value of the negative mean.
            - 'Observations': int
                The total number of observations considered.
        - If output_format is `Series`, return a pandas Series with
        statistics as rows and a 'Stats' column as the index.

        - If output_format is `DataFrame`, return a pandas DataFrame
        with statistics as rows and a 'Stats' column as the index.

        Raises
        ------
        ValueError
            If output_format is not one of `dict`, `Series`, or
            `DataFrame`.
        ValueError
            If result_column is `None` and the input data_frame is not
            a Series.
        """
        data_frame = self.data_frame.copy()

        if is_percentage_data:
            data_frame = (data_frame - 1) * 100

        if output_format not in ["dict", "Series", "DataFrame"]:
            raise ValueError(
                "output_format must be one of 'dict', 'Series', or "
                "'DataFrame'."
            )

        if result_column is None:
            if isinstance(data_frame, pd.Series):
                positive = data_frame[data_frame > 0]
                negative = data_frame[data_frame < 0]
                positive_mean = positive.mean()
                negative_mean = negative.mean()
            else:
                raise ValueError(
                    "result_column must be provided for DataFrame input."
                )

        else:
            positive = data_frame.query(f"{result_column} > 0")
            negative = data_frame.query(f"{result_column} < 0")
            positive_mean = positive[result_column].mean()
            negative_mean = negative[result_column].mean()

        win_rate = positive.shape[0] / (positive.shape[0] + negative.shape[0])

        expected_return = positive_mean * win_rate - negative_mean * (
            win_rate - 1
        )

        payoff = positive_mean / abs(negative_mean)

        results = {
            "Expected_Return": expected_return,
            "Win_Rate": win_rate,
            "Positive_Mean": positive_mean,
            "Negative_Mean": negative_mean,
            "Payoff": payoff,
            "Observations": positive.shape[0] + negative.shape[0],
        }

        stats_str = "Stats %" if is_percentage_data else "Stats"
        if output_format == "Series":
            return pd.Series(results).rename(stats_str)
        if output_format == "DataFrame":
            return pd.DataFrame(results, index=["Value"]).T.rename_axis(
                stats_str
            )

        return results

def get_recommendation(
    predict_series: pd.Series,
    time_stop: int = 7,
    add_span_tag: bool = False
) -> pd.Series:
    """
    Generate trading recommendations based on the prediction series.

    The function generates trading recommendations based on the
    predictions provided in the input series. The recommendations are
    based on the following rules:

    - Open a long position when the prediction is 1.
    - Close the long position after a specified number of periods

    Parameters
    ----------

    predict_series : pd.Series
        A pandas Series containing the prediction values.
    time_stop : int, optional
        The number of periods to consider for time stop (closing the
        position after a certain number of periods).

        (default: 7)
    add_span_tag : bool, optional
        If True, the recommendations are returned as HTML strings with
        span tags for color formatting. If False, the recommendations
        are returned as plain text. (compatible with itables)

        (default: False)

    Returns
    -------
    pd.Series
        A pandas Series containing the trading recommendations.
    """
    data = predict_series.copy()

    open_positions_df = data.rename("Open_Position")
    close_positions_df = data.shift(time_stop).rename("Close_Position") * -1

    positions_df = pd.concat(
        [open_positions_df, close_positions_df], axis=1
    ).fillna(0)
    positions_df["Position"] = positions_df.sum(axis=1)
    positions_df["Open_trades"] = positions_df["Position"].cumsum()

    positions_df["Open_trades"] = (
        (positions_df["Open_trades"] * 33.3).round(0).astype(int)
    ).astype(str) + "%"

    open_pos = (
        (positions_df["Open_Position"] > 0)
        & (positions_df["Position"] != 0)
    )

    close_pos = (
        (positions_df["Close_Position"] < 0)
        & (positions_df["Position"] != 0)
    )

    open_pos_df = positions_df[open_pos]["Open_Position"].cumsum()
    close_pos_df = positions_df[close_pos]["Close_Position"].cumsum()
    trade_pos_df = open_pos_df.combine_first(close_pos_df)

    positions_df["Trade_position"] = trade_pos_df.astype(str)

    positions_df["Trade_position"] = positions_df["Trade_position"].ffill()
    positions_df["Trade_position"] = positions_df["Trade_position"].replace(
        np.nan, "0"
    )

    if add_span_tag:
        long_color = "<b><span style='color: #00e676'>Open Position</span></b>"
        do_nothing_color = "——————"
        short_color = (
            "<b><span style='color: #ef5350'>Close Position</span></b>"
        )
        start_html = "<b><span style='color: #fff'> | "
        end_html = " |</b></span>"

    else:
        long_color = "Open Position"
        do_nothing_color = "——————"
        short_color = "Close Position"
        start_html = ""
        end_html = ""

    trade_sides = [
        positions_df["Position"] == 1,
        positions_df["Position"] == -1,
    ]

    trade_positions = "(" + positions_df["Trade_position"] + ") "

    pos_size = start_html + positions_df["Open_trades"] + end_html

    trade_results = [
        trade_positions + long_color + pos_size,
        trade_positions + short_color + pos_size,
    ]

    positions_df["Recommendation"] = np.select(
        trade_sides, trade_results, default=do_nothing_color
    )

    return positions_df["Recommendation"]
