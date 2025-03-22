import pandas as pd
import numpy as np
from typing import Literal

class Statistics:
    """A class for calculating strategy statistics.

    Parameters
    ----------
    dataframe : pd.Series or pd.DataFrame
        The input dataframe containing the results of the strategy. If
        `dataframe` is a pd.Series, it should contain a single column
        of results. If it is a pd.DataFrame, it should have a 'Result'
        column containing the results.

    time_span : str, optional
        The time span for resampling the returns. The default is "A"
        (annual).

    risk_free_rate : float, optional
        The risk free rate of the strategy. The default is 0.

    is_percent : bool, optional
        Whether the results are in percentage form. If True, the calculated
        statistics will be multiplied by 100. Default is False.

    Attributes
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing the `Result` column.

    Methods
    -------
    calculate_all_statistics(precision: int = 2) -> pd.DataFrame
        Calculate all strategy statistics.

    calculate_expected_value() -> pd.DataFrame
        Calculate the expected value of the strategy.

    calculate_estimated_sharpe_ratio() -> pd.Series
        Calculate the Sharpe ratio of the strategy.

    calculate_estimated_sortino_ratio() -> pd.Series
        Calculate the Sortino ratio of the strategy.

    """
    def __init__(

        self,
        dataframe: pd.Series | pd.DataFrame,
        time_span: str = "A",
        risk_free_rate: float = 0,
        is_percent: bool = False,
    ):
        """Calculates performance metrics based on the provided data.

        Parameters
        ----------
        dataframe : pandas.Series or pandas.DataFrame
            The input data. If a Series is provided, it is converted to
            a DataFrame with a "Result" column. If a DataFrame is
            provided, it should contain a "Result" column.
        time_span : str, optional
            The time span of the data. Defaults to "A" (annual).
        risk_free_rate : float, optional
            The risk-free rate to be used in performance calculations.
            Defaults to 0 (no risk-free rate).
        is_percent : bool, optional
            Indicates whether the data is in percentage format.
            Defaults to False.

        Raises
        ------
        ValueError
            If an invalid dataframe is provided.

        Notes
        -----
        The risk-free rate should be consistent with the timeframe used
        in the dataframe. If the timeframe is annual and the risk-free
        rate is 2%, the risk_free_rate value should be set as
        `0.00007936507`  (0.02 / 252) if the asset has 252 trading days.

        """
        if isinstance(dataframe, pd.Series):
            self.dataframe = pd.DataFrame({"Result": dataframe})
        elif "Result" in dataframe.columns:
            self.dataframe = dataframe[["Result"]].copy()
        else:
            raise ValueError(
                """
                Invalid dataframe. The dataframe should be a
                pd.Series or a pd.DataFrame with a 'Result' column.
                """
            )

        self.dataframe["Result"] = (
            self.dataframe[["Result"]]
            .query("Result != 0")
            .dropna()
        )

        if is_percent:
            self.dataframe = self.dataframe * 100

        self.time_span = time_span
        self.risk_free_rate = risk_free_rate

    def calculate_all_statistics(self, precision: int = 2):
        """
        Calculate all strategy statistics.

        Parameters
        ----------
        precision : int, optional
            The number of decimal places to round the calculated
            statistics to. Defaults to 2.

        Returns
        -------
        pd.DataFrame
            A dataframe with calculated statistics, including expected
            value, Sharpe ratio, and Sortino ratio.
        """
        stats_df = pd.DataFrame()
        stats_df["Expected_Value"] = self.calculate_expected_value()["Expected_Value"]
        stats_df = stats_df.resample(self.time_span).mean()

        stats_df["Sharpe_Ratio"] = self.calculate_estimed_sharpe_ratio()
        stats_df["Sortino_Ratio"] = self.calculate_estimed_sortino_ratio()

        if self.time_span.endswith("YE"):
            stats_df.index = stats_df.index.year
        if self.time_span.endswith("ME"):
            stats_df.index = stats_df.index.strftime('%m/%Y')
        return round(stats_df, precision)

    def calculate_expected_value(
        self,
        output: Literal["complete", "resampled"] = "complete",
    ) -> pd.DataFrame:
        """
        Calculate the expected value of the strategy.

        Returns
        -------
        pd.DataFrame
            A dataframe with calculated statistics, including gain count,
            loss count, mean gain, mean loss, total gain, total loss,
            total trade, win rate, loss rate, and expected value (EM).

        """
        dataset = self.dataframe.copy()

        gain = dataset["Result"] > 0
        loss = dataset["Result"] < 0

        dataset["Gain_Count"] = np.where(gain, 1, 0)
        dataset["Loss_Count"] = np.where(loss, 1, 0)

        dataset["Gain_Count"] = dataset["Gain_Count"].cumsum()
        dataset["Loss_Count"] = dataset["Loss_Count"].cumsum()

        query_gains = dataset.query("Result > 0")["Result"]
        query_loss = dataset.query("Result < 0")["Result"]

        dataset["Mean_Gain"] = query_gains.expanding().mean()
        dataset["Mean_Loss"] = query_loss.expanding().mean()

        dataset["Mean_Gain"] = dataset["Mean_Gain"].ffill()
        dataset["Mean_Loss"] = dataset["Mean_Loss"].ffill()

        dataset["Total_Gain"] = (
            np.where(gain, dataset["Result"], 0)
            .cumsum()
        )

        dataset["Total_Loss"] = (
            np.where(loss, dataset["Result"], 0)
            .cumsum()
        )

        total_trade = dataset["Gain_Count"] + dataset["Loss_Count"]
        win_rate = dataset["Gain_Count"] / total_trade
        loss_rate = dataset["Loss_Count"] / total_trade

        dataset["Total_Trade"] = total_trade
        dataset["Win_Rate"] = win_rate
        dataset["Loss_Rate"] = loss_rate

        ev_gain = dataset["Mean_Gain"] * dataset["Win_Rate"]
        ev_loss = dataset["Mean_Loss"] * dataset["Loss_Rate"]
        dataset["Expected_Value"] = ev_gain - abs(ev_loss)

        match output:
            case "complete":
                return dataset
            case "resampled":
                return dataset["Expected_Value"].resample(self.time_span).mean()
            case _:
                raise ValueError(
                    "Invalid output type. Use 'complete' or 'resampled'."
                )

    def calculate_estimed_sharpe_ratio(self) -> pd.Series:
        """
        Calculate the Sharpe ratio of the strategy.

        Returns
        -------
        pd.Series
            A series containing the Sharpe ratio values.

        """
        results = self.dataframe["Result"]
        returns_annualized = (
            results
            .resample(self.time_span)
        )

        mean_excess = returns_annualized.mean() - self.risk_free_rate

        sharpe_ratio = mean_excess / returns_annualized.std()

        return sharpe_ratio

    def calculate_estimed_sortino_ratio(self) -> pd.Series:
        """
        Calculate the Sortino ratio of the strategy.

        Returns
        -------
        pd.Series
            A series containing the Sortino ratio values.

        """
        results = self.dataframe["Result"]
        returns_annualized = (
            results
            .resample(self.time_span)
        )

        negative_results = self.dataframe.query("Result < 0")["Result"]
        negative_returns_annualized = (
            negative_results
            .resample(self.time_span)
        )

        mean_excess = returns_annualized.mean() - self.risk_free_rate

        sortino_ratio = mean_excess / negative_returns_annualized.std()

        return sortino_ratio
