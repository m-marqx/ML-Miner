from typing import Literal
import datetime

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

        Parameters:
        -----------
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

        Returns:
        --------
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
    predict_series,
    return_dtype: Literal["string", "normal", "int", "bool"] = "string",
    add_span_tag: bool = False,
):
    """
    Generate trading recommendations based on the prediction series.

    This function takes a series of predictions and generates trading
    recommendations in different formats based on the specified return
    data type.

    Parameters
    ----------
    predict_series : pd.Series
        A pandas Series containing the prediction values.
    return_dtype : {'string', 'normal', 'int', 'bool'}, optional
        The desired return data type for the recommendations.
        - 'string': Returns recommendations as strings
        ('Long', 'Do Nothing', 'Short').
        - 'normal': Returns the original prediction series.
        - 'int': Returns the predictions as integers.
        - 'bool': Returns the predictions as boolean values
        (True for positive, False otherwise).

        (default:'string')
    add_span_tag : bool, optional
        If True, the recommendations are returned as HTML strings with
        span tags for color formatting. If False, the recommendations
        are returned as plain text. (compatible with itables)

        (default: False)

    Returns
    -------
    pd.Series
        A pandas Series containing the trading recommendations in the
        specified format.
    """
    predict = predict_series.copy()
    predict.index = predict.index.date
    predict = predict.rename_axis("date")

    confirmed_signals = pd.Series(predict.iloc[:-1], name=predict.name)

    unconfirmed_signal = pd.Series(
        predict.iloc[-1],
        index=["Unconfirmed"],
    )

    signals = pd.concat([confirmed_signals, unconfirmed_signal])

    if add_span_tag:
        long_color = "<b><span style='color: #00e676'>Open Position</span></b>"
        do_nothing_color = "——————"
        short_color = (
            "<b><span style='color: #ef5350'>Close Position</span></b>"
        )

    else:
        long_color = "Open Position"
        do_nothing_color = "——————"
        short_color = "Close Position"

    match return_dtype:
        case "string":
            recommendation = np.where(
                signals > 0,
                long_color,
                np.where(signals == 0, do_nothing_color, short_color),
            )
            recommendation_series = pd.Series(
                recommendation, index=signals.index
            )

            recommendation_array = np.where(
                (recommendation_series.shift(7) == long_color)
                & (recommendation_series == do_nothing_color),
                short_color,
                recommendation_series,
            )

            return pd.Series(
                recommendation_array,
                index=recommendation_series.index,
                name=predict.name,
            )

        case "normal":
            return signals

        case "int":
            return signals.astype(int)

        case "bool":
            return signals > 0
