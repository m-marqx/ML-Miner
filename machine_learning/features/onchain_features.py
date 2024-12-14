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
>>> features.create_std_feature('D', 'txs', 2, 4)
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

    create_std_feature(freq, column, short_window, long_window)
        Creates standard deviation ratio features by resampling data at
        specified frequency and calculating ratios between short and
        long term standard deviations

    calculate_std_ratio_feature(feature, short_window, long_window)
        Helper method that calculates the ratio between short-term and
        long-term standard deviations for a feature series

    Example
    -------
    >>> features = OnchainFeatures()
    >>> std_ratio = features.create_std_feature('D', 'txs', 2, 4)
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
        self.dataset = self.onchain_data.copy()
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

    def calculate_resampled_data(self, column: str, freq: str) -> pd.Series:
        """
        Calculate a resampled data based on column type and frequency.

        Parameters
        ----------
        column : str
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

        feature = self.onchain_data[column].copy()

        if column in avg_infos:
            return feature.resample(freq).mean()

        if column in max_infos:
            return feature.resample(freq).max()

        if column in median_infos:
            return feature.resample(freq).median()

        if column in min_infos:
            return feature.resample(freq).min()

        if column in [*total_infos, *txs_infos, "subsidy"]:
            return feature.resample(freq).sum()

        if column in ["height", "blockhash"]:
            return feature.resample(freq).count()

        if column in feerate_percentiles:
            raise InvalidArgumentError(f"{column} isn't compatible")

        raise InvalidArgumentError(f"{column} isn't a valid column.")

