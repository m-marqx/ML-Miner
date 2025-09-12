from typing import Literal

import numpy as np
import pandas as pd
from machine_learning.model_builder import adjust_max_trades


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

        self.data_frame["Return"] = (
            self.data_frame["close"].pct_change(length) + 1
        )
        self.data_frame["Target"] = self.data_frame["Return"].shift(-length)
        self.data_frame["Target_bin"] = np.where(
            self.data_frame["Target"] > 1, 1, 0
        )

        self.data_frame["Target_bin"] = np.where(
            self.data_frame["Target"].isna(),
            np.nan,
            self.data_frame["Target_bin"],
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
        start_html = " | "
        end_html = " |"

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

def calculate_returns(
    target: pd.Series,
    y_pred_probs: pd.Series,
    y_true: pd.Series,
    fee: int = 0,
    off_days: int = 7,
    max_trades: int = 3,
    side: int = 1,
    cutoff_point: int = 5,
):
    """
    Calculate trading performance metrics from predicted probabilities
    and true outcomes. This function builds a returns DataFrame from
    model prediction probabilities and the realized target series,
    applies a cutoff to convert probabilities into binary predictions,
    computes gross and fee-aware (liquid) returns, drawdowns and
    drawdown durations, and finally adjusts the resulting trade sequence
    using adjust_max_trades
    (e.g. to enforce off-days / max concurrent trades).

    Parameters
    ----------
    target : pandas.Series
        Series of realized target values aligned to the same index as
        the prediction series. If all values are > 0 the function will
        first convert values that represent gross multipliers
        (e.g. 1.01) into returns by subtracting 1.
    y_pred_probs : pandas.Series
        Series of predicted probabilities (or scores) used to form the
        trading signal. Used to compute the threshold/cutoff and as a
        reference column in the returned DataFrame.
    y_true : pandas.Series
        Series of true binary labels (e.g. 1 for event, 0 otherwise).
        Only used to create a binary `y_true` column in the result for
        comparison/analysis.
    fee : int or float, optional
        Transaction fee expressed in percent (e.g. 0.1 for 0.1%).
        (default: 0)
    off_days : int, optional
        Number of days to enforce between trades when calling
        adjust_max_trades. Passed through to adjust_max_trades.
        (default: 7)
    max_trades : int, optional
        Maximum number of simultaneous/overlapping trades allowed. Passed to
        adjust_max_trades.
        (default: 3)
    side : int, optional
        Trade side filter. If set to 1 (default) only entries where the
        predicted side equals 1 keep their liquid result; other sides
        have Liquid_Result zeroed. Use -1 to select the opposite side
        when relevant.
        (default: 1)
    cutoff_point : int or None, optional
        If truthy, defines a percentile (0 < cutoff_point < 100)
        applied to the subset of predictions above the initial median
        cutoff to determine the final cutoff threshold. Example:
        cutoff_point=5 sets the cutoff to the 5th percentile of
        predictions that are greater than the median. If falsy
        (e.g. 0 or None) the initial cutoff is the median of
        y_pred_probs.
        (default: 5)

    Returns
    -------
    pandas.DataFrame
        A DataFrame (the same object passed to adjust_max_trades)
        containing at least the following columns:
        - y_pred_probs: original prediction scores
        - Predict: binary prediction (1 if score > cutoff else 0)
        - y_true: binary mapped true labels (0/1)
        - target_Return: aligned numeric returns from `target`
        - Position: shifted Predict (previous period signal)
        - Result: gross return when Predict==1 (target_Return * Predict)
        - Liquid_Result: Result net of fee where applicable, otherwise
        0; finally filtered by `side`
        - Period_Return_cum: cumulative sum of target_Return
        - Total_Return: cumulative gross return (Result.cumsum() + 1)
        - Liquid_Return: cumulative fee-adjusted return
        (Liquid_Result.cumsum() + 1)
        - max_Liquid_Return: rolling expanding maximum of Liquid_Return
        (window 365)
        - drawdown: drawdown series computed from Liquid_Return
        vs max_Liquid_Return
        - drawdown_duration: number of consecutive periods in drawdown
        The returned DataFrame is the output of adjust_max_trades(...),
        so any  additional columns produced by that function may also
        be present.

    Raises
    ------
    ValueError
        If `cutoff_point` is provided and is not strictly between 0
        and 100.
    """
    target_series = target.copy()

    if target_series.min() > 0:
        target_series = target_series - 1

    predict = y_pred_probs.copy()

    cutoff = np.median(predict)

    if cutoff_point:
        if cutoff_point >= 100:
            raise ValueError("Cutoff point must be less than 100")
        if cutoff_point <= 0:
            raise ValueError("Cutoff point must be greater than 0")

        predict_mask = predict > cutoff

        cutoff = np.percentile(predict[predict_mask], cutoff_point)

    df_returns = pd.DataFrame(
        {
            "y_pred_probs": y_pred_probs,
        }
    )

    df_returns["Predict"] = np.where(y_pred_probs > cutoff, 1, 0)
    df_returns["y_true"] = np.where(y_true == 1, 1, 0)

    fee_pct = fee / 100

    target_return = target_series.reindex(df_returns.index)

    df_returns["target_Return"] = target_return

    df_returns["Position"] = df_returns["Predict"].shift().fillna(0)

    df_returns["Result"] = df_returns["target_Return"] * df_returns["Predict"]

    df_returns["Liquid_Result"] = np.where(
        (df_returns["Predict"] != 0) & (df_returns["Result"].abs() != 1),
        df_returns["Result"] - fee_pct,
        0,
    )

    df_returns["Period_Return_cum"] = (df_returns["target_Return"]).cumsum()

    df_returns["Total_Return"] = df_returns["Result"].cumsum() + 1
    df_returns["Liquid_Return"] = df_returns["Liquid_Result"].cumsum() + 1

    df_returns["max_Liquid_Return"] = (
        df_returns["Liquid_Return"].expanding(365).max()
    )

    df_returns["max_Liquid_Return"] = np.where(
        df_returns["max_Liquid_Return"].diff(),
        np.nan,
        df_returns["max_Liquid_Return"],
    )

    df_returns["drawdown"] = (
        1 - df_returns["Liquid_Return"] / df_returns["max_Liquid_Return"]
    ).fillna(0)

    drawdown_positive = df_returns["drawdown"] > 0

    df_returns["drawdown_duration"] = drawdown_positive.groupby(
        (~drawdown_positive).cumsum()
    ).cumsum()

    df_returns["Liquid_Result"] = np.where(
        df_returns["Predict"] != side,
        0,
        df_returns["Liquid_Result"],
    )

    return adjust_max_trades(
        data_set=df_returns,
        off_days=off_days,
        max_trades=max_trades,
        pct_adj=0.5,
        side=side,
    )
