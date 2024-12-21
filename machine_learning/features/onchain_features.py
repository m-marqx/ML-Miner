"""
This module provides functionality for processing and analyzing
blockchain on-chain features.

The module contains the OnchainFeatures class which handles loading,
processing, and transforming blockchain data for feature engineering in
machine learning applications. It supports operations like data
resampling and calculating standard deviation ratios.

Classes
-------
OnchainFeatures
    Main class for handling blockchain on-chain data processing and
    feature engineering.

Example
-------
>>> features = OnchainFeatures()
>>> features.create_std_ratio_feature('D', 'txs', 2, 4)
"""

import pathlib
import time

import pandas as pd
from utils.log_handler import create_logger
from custom_exceptions.invalid_arguments import InvalidArgumentError
from machine_learning.model_features import feature_binning

class OnchainFeatures:
    """
    A class for handling and processing on-chain blockchain features.

    This class provides functionality to load, process, and transform
    blockchain on-chain data for feature engineering purposes.

    Parameters
    ----------
    onchain_data : pd.DataFrame | None, optional
        DataFrame containing blockchain on-chain data. If None, data
        will be loaded from default path.
        (default: None)
    verbose : bool, optional
        Flag to control logging verbosity
        (default: False)

    Attributes
    ----------
    onchain_data : pd.DataFrame
        The processed on-chain blockchain data
    logger : Logger
        Logger instance for the class

    Methods
    -------
    create_resampled_feature(column, freq)
        Creates resampled features based on column type and specified
        frequency (e.g., daily, weekly, etc.).

    create_std_ratio_feature(freq, column, short_window, long_window)
        Creates standard deviation ratio features by resampling data at
        specified frequency and calculating ratios between short and
        long term standard deviations

    calculate_std_ratio_feature(feature, short_window, long_window)
        Helper method that calculates the ratio between short-term and
        long-term standard deviations for a feature series

    Example
    -------
    >>> features = OnchainFeatures()
    >>> std_ratio = features.create_std_ratio_feature('D', 'txs', 2, 4)
    """

    def __init__(
        self,
        onchain_data: pd.DataFrame | None = None,
        test_index: int | None = None,
        bins: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize the OnchainFeatures class with blockchain data.

        If no data is provided, loads default BTC block statistics from
        the data/onchain/BTC directory. Converts timestamp to datetime
        and sets it as the DataFrame index.

        Parameters
        ----------
        onchain_data : pd.DataFrame | None, optional
            DataFrame containing blockchain on-chain data. If None, data
            will be loaded from default path. Must have 'time' column in
            Unix timestamp format if provided.
            (default: None)
        verbose : bool, optional
            Flag to control logging verbosity
            (default: False)

        Notes
        -----
        The data is expected to have the following columns:
        - avgfee: Average transaction fee
        - avgfeerate: Average fee rate
        - avgtxsize: Average transaction size
        - height: Block height
        - etc...

        The 'time' column is converted to datetime and set as index.
        """
        if not onchain_data:
            path = pathlib.Path(
                "data",
                "onchain",
                "BTC",
                "block_stats.parquet",
            )
            self.onchain_data = pd.read_parquet(path)

        self.onchain_data["time"] = pd.to_datetime(
            self.onchain_data["time"], unit="s"
        )
        self.onchain_data = self.onchain_data.set_index("time")
        self.dataset = pd.DataFrame()
        self.test_index = test_index
        self.bins = bins

        self.logger = create_logger("OnchainFeatures", verbose)

    def set_bins(self, bins: int):
        """
        Set the number of bins to use for binning the features.

        Parameters
        ----------
        bins : int
            The number of bins to use for binning the features.
        """
        self.bins: int = bins
        return self

    def calculate_resampled_data(self, columns: list, freq: str) -> pd.Series:
        """
        Calculate a resampled data based on column type and frequency.

        Parameters
        ----------
        columns : list
            The column name to resample from onchain_data
        freq : str
            Frequency to resample the data
            (e.g. 'D' for daily, 'W' for weekly)

        Returns
        -------
        pd.Series
            Resampled feature series with specified frequency

        Raises
        ------
        InvalidArgumentError
            If column name is invalid or incompatible

        Notes
        -----
        Different resampling methods are used based on column type:
        - Average: mean() for fees, rates, sizes
        - Maximum: max() for max fees, rates, sizes
        - Median: median() for median times and sizes
        - Minimum: min() for min fees, rates, sizes
        - Sum: sum() for totals and transaction counts
        - Count: count() for heights and hashes
        """
        avg_infos = ["avgfee", "avgfeerate", "avgtxsize"]
        feerate_percentiles = ["feerate_percentiles"]
        max_infos = ["maxfee", "maxfeerate", "maxtxsize"]
        median_infos = ["medianfee", "mediantime", "mediantxsize"]

        # The minimum fee and rate values are equals to 0 in 99% of the time
        # if the frequency <= 1 day
        min_infos = ["minfee", "minfeerate", "mintxsize"]
        total_infos = ["total_out", "total_size", "total_weight", "totalfee"]

        txs_infos = [
            "ins",
            "outs",
            "txs",
            "utxo_increase",
            "utxo_size_inc",
            "utxo_increase_actual",
            "utxo_size_inc_actual",
        ]

        feature = self.onchain_data[columns].copy()
        resampled_feature = feature[columns].resample(freq)

        features = {}
        for col in columns:
            if col in avg_infos:
                features[col] = resampled_feature[col].mean()

            elif col in [*total_infos, *txs_infos, "subsidy"]:
                features[col] = resampled_feature[col].sum()

            elif col in ["height", "blockhash"]:
                features[col] = resampled_feature[col].count()

            elif col in max_infos:
                features[col] = resampled_feature[col].max()

            elif col in median_infos:
                features[col] = resampled_feature[col].median()

            elif col in feerate_percentiles or min_infos:
                raise InvalidArgumentError(f"{col} isn't compatible")

            else:
                raise InvalidArgumentError(f"{col} isn't a valid column.")

        features_df = pd.concat(features, axis=1)
        features_df.columns = [f"{key}_std_ratio" for key in features]
        return features_df

    def calculate_std_ratio_feature(
        self,
        feature: pd.Series | pd.DataFrame,
        short_window: int,
        long_window: int,
    ):
        """
        Calculate the ratio between short-term and long-term standard
        deviations.

        Computes rolling standard deviations over two different window
        sizes and returns their ratio as a feature for machine learning.

        Parameters
        ----------
        feature : pd.Series
            Series containing the feature to calculate the ratio for
        short_window : int, optional
            Size of the shorter rolling window for std calculation
            (default: 2)
        long_window : int, optional
            Size of the longer rolling window for std calculation
            (default: 4)

        Returns
        -------
        pd.DataFrame
            DataFrame containing the ratio of short-term to long-term
            standard deviations for each column in onchain_data
        """
        if short_window > long_window:
            raise InvalidArgumentError(
                "Short window size must be smaller than long window size."
            )

        if short_window < 2 or long_window < 2:
            raise InvalidArgumentError(
                "Window sizes must be greater than or equal to 2."
            )

        if short_window == long_window:
            raise InvalidArgumentError(
                "Window sizes must be different."
            )

        if not isinstance(short_window, int) or not isinstance(long_window, int):
            raise InvalidArgumentError(
                "Window sizes must be integers."
            )

        if not isinstance(feature, pd.Series) and not isinstance(feature, pd.DataFrame):
            raise InvalidArgumentError(
                "Feature must be a pandas Series or pandas DataFrame."
            )

        if feature.empty:
            raise InvalidArgumentError(
                "Feature can't be empty"
            )

        self.logger.info("Creating features for the machine learning model.")

        short_std = feature.rolling(short_window).std()
        long_std = feature.rolling(long_window).std()
        std_ratio = short_std / long_std
        return std_ratio

    def create_std_ratio_feature(
        self,
        column: str | list,
        freq: str,
        short_window: int = 2,
        long_window: int = 4,
    ) -> pd.Series:
        """
        Create a standard deviation ratio feature for a given column at
        a specified frequency.

        Parameters
        ----------
        freq : str
            The frequency to resample the data to
            (e.g., 'D' for daily, 'W' for weekly)
        column : str, optional
            The specific column to calculate the standard deviation
            ratio for
            (default: None)
        short_window : int, optional
            Size of shorter rolling window for std calculation
            (default: 2)
        long_window : int, optional
            Size of longer rolling window for std calculation
            (default: 4)

        Returns
        -------
        pd.Series
            Series containing the ratio of short-term to long-term
            standard deviations

        Raises
        ------
        InvalidArgumentError
            If the specified column is not valid or window parameters
            are invalid
        """
        self.logger.info("Calculating standard deviation ratio feature.")
        start = time.perf_counter()
        feature = self.calculate_resampled_data(column, freq)

        self.dataset = self.calculate_std_ratio_feature(
            feature,
            short_window,
            long_window
        )

        indexes_after_2011 = self.dataset.index >= pd.Timestamp("2011-01-01")
        self.dataset = self.dataset[indexes_after_2011]

        for col in self.dataset.columns:
            self.dataset[f"{col}_feat"] = feature_binning(
                self.dataset[col],
                self.test_index,
                self.bins,
            )

        self.logger.info(
            "Standard deviation ratio feature calculated in %.2f seconds.",
            time.perf_counter() - start,
        )

        return self
