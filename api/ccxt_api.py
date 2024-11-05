import time
import logging
from typing import Literal
import numpy as np
import pandas as pd
import ccxt
from tqdm import tqdm

from utils import interval_to_milliseconds
from api.kline_utils import KlineTimes
from custom_exceptions.invalid_arguments import InvalidArgumentError

class CcxtAPI:
    """
    A class for interacting with the CCXT library to retrieve financial
    market data.

    Parameters
    ----------
    symbol : str
        The trading symbol for the asset pair (e.g., 'BTC/USD').
    interval : str
        The time interval for K-line data
        (e.g., '1h' for 1-hour candles).
    exchange : ccxt.Exchange
        The CCXT exchange object
        (default: ccxt.bitstamp()).
    since : int
        The Unix timestamp of the first candle
        (default: 1325296800000).
    verbose : bool
        If True, print verbose logging messages during data retrieval
        (default: False).

    Attributes
    ----------
    symbol : str
        The trading symbol for the asset pair.
    interval : str
        The time interval for K-line data.
    since : int
        The Unix timestamp of the first candle.
    data_frame : pd.DataFrame
        DataFrame to store the K-line data.
    exchange : ccxt.Exchange
        The CCXT exchange object.
    max_interval : str
        The maximum time interval supported by the asset pair.
    utils : KlineTimes
        An instance of the KlineTimes class for time-related
        calculations.
    max_multiplier : int
        The maximum multiplier calculated based on the time interval.

    Methods:
    --------
    get_since_value_value() -> int or None:
        Search for the Unix timestamp of the first candle in the
        historical K-line data.

    get_all_klines(ignore_unsupported_exchanges=False) -> CcxtAPI:
        Fetch all K-line data for the specified symbol and interval.

    to_OHLCV() -> pd.DataFrame:
        Convert the fetched K-line data into a pandas DataFrame in
        OHLCV format.

    aggregate_klines(
        exchanges=None,
        symbols=None,
        output_format='DataFrame',
        method='mean',
        filter_by_largest_qty=True
    ) -> pd.DataFrame or dict or tuple:
        Aggregate the fetched K-line data into a pandas DataFrame.

    date_check() -> pd.DataFrame:
        Check for irregularities in the K-line data timestamps and
        return a DataFrame with discrepancies.
    """
    def __init__(
        self,
        symbol:str,
        interval:str,
        exchange:ccxt.Exchange = ccxt.bitstamp(),
        since:int = 1325296800000,
        verbose:Literal["Text", "Progress_Bar"] | None = None,
    ) -> None:
        """
        Initialize the CcxtAPI object.

        Parameters
        ----------
        symbol : str
            The trading symbol for the asset pair.
        interval : str
            The time interval for K-line data.
        exchange : ccxt.Exchange
            The CCXT exchange object.
        since : int
            The Unix timestamp of the first candle.
        verbose : bool
            If True, print verbose logging messages during data retrieval
            (default: False).
        """
        self.symbol = symbol
        self.interval = interval
        self.since = since
        self.exchange = exchange
        self.is_progress_bar_verbose = verbose == "Progress_Bar"
        self.utils = KlineTimes(interval, adjust_interval=True)

        self.max_multiplier = (
            int(self.utils.calculate_max_multiplier()) if interval != '1w'
            else None
        )

        self.data_frame: pd.DataFrame = pd.DataFrame()
        self.klines_list = []

        self.logger = logging.getLogger("CCXT_API")
        formatter = logging.Formatter(
            '%(levelname)s %(asctime)s: %(message)s', datefmt='%H:%M:%S'
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.addHandler(handler)
        self.logger.propagate = False

        if verbose == "Text":
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    def _fetch_klines(self, since, limit: int | None = None) -> list:
        return self.exchange.fetch_ohlcv(
            symbol=self.symbol,
            timeframe=self.interval,
            since=since,
            limit=limit,
        )

    def get_since_value(self):
        """
        Search for the Unix timestamp of the first candle in the
        historical K-line data.

        This method iteratively fetches K-line data in reverse
        chronological order and stops when it finds the first candle.
        It can be used to determine the starting point for fetching
        historical data.

        Returns
        -------
        int or None
            The Unix timestamp of the first candle found, or None
            if not found.
        """
        end_times = self.utils.get_end_times(
            self.since,
            self.max_multiplier
        )
        end_times_range = range(0, len(end_times) - 1)

        end_times_range = (
            tqdm(end_times_range) if self.is_progress_bar_verbose
            else end_times_range
        )

        for index in end_times_range:
            klines = self._fetch_klines(
                since=int(end_times[index]),
                limit=self.max_multiplier,
            )
            load_percentage = (index / (len(end_times) - 1)) * 100

            self.logger.info("Finding first candle time [%.2f%%]", load_percentage)

            if len(klines) > 0:
                first_unix_time = klines[0][0]

                self.logger.info("Finding first candle time [100%]")
                self.logger.info("First candle time found: %s\n", first_unix_time)
                break

        return first_unix_time

    def get_all_klines(
        self,
        until: int | None = None,
        ignore_unsupported_exchanges: bool = False
    ):
        """
        Fetch all K-line data for the specified symbol and interval
        using a for loop.

        Parameters
        ----------
        until : None
            The end time for fetching K-line data.
        ignore_unsupported_exchanges : bool, optional
            If True, ignore exchanges that do not support the specified
            symbol.
            (default: False).

        Returns
        -------
        CcxtAPI
            Returns the CcxtAPI object with the fetched K-line data.
        """
        if ignore_unsupported_exchanges:
            not_supported_types = None
        else:
            not_supported_types = (
                type(ccxt.gemini()),
                type(ccxt.huobi()),
                type(ccxt.deribit()),
                type(ccxt.hitbtc()),
            )

        if isinstance(self.exchange, not_supported_types):
            raise InvalidArgumentError(f"{self.exchange} is not supported")

        klines = []
        klines_list = []

        first_call = self._fetch_klines(self.since, self.max_multiplier)

        if first_call:
            first_unix_time = first_call[0][0]
        else:
            first_unix_time = self.get_since_value()
            first_call = self._fetch_klines(first_unix_time, self.max_multiplier)

        last_candle_interval = (
            (
                time.time() * 1000 - interval_to_milliseconds(self.interval)
                if until is None
                else until
            )
        )

        start = time.perf_counter()
        self.logger.info("Starting requests")

        time_value = klines[-1][0] + 1 if klines else first_unix_time
        time_delta = first_call[-1][0] - first_call[0][0]
        step = time_delta + pd.Timedelta(self.interval).value / 1e+6
        end_times = np.arange(time_value, last_candle_interval, step)
        ranges = tqdm(end_times) if self.is_progress_bar_verbose else end_times

        for current_start_time in ranges:
            klines = self._fetch_klines(
                int(current_start_time),
                self.max_multiplier
            )
            if not klines:
                break

            klines_list.extend(klines)

            if klines_list[-1][0] >= last_candle_interval:
                self.logger.info(
                    "Qty: %d - Total: 100%% complete",
                    len(klines_list)
                )

            percentage = (
                (np.where(end_times == current_start_time)[0][0] + 1)
                / end_times.shape[0]
            ) * 100
            self.logger.info(
                "Qty: %d - Total: %.2f%% complete",
                len(klines_list), percentage
            )

        self.logger.info(
            "Requests elapsed time: %s\n",
            time.perf_counter() - start
        )
        self.klines_list = klines_list
        return self

    def to_OHLCV(self):
        """
        Convert the fetched K-line data into a pandas DataFrame in
        OHLCV format.

        Returns
        -------
        pd.DataFrame
            Returns a pandas DataFrame containing OHLCV data.
        """
        ohlcv_columns = ["open", "high", "low", "close", "volume"]

        self.data_frame = pd.DataFrame(
            self.klines_list,
            columns=["date"] + ohlcv_columns
        )

        self.data_frame["date"] = self.data_frame["date"].astype(
            "datetime64[ms]"
        )

        self.data_frame = self.data_frame.set_index("date")
        return self

    def date_check(self) -> pd.DataFrame:
        """
        Check for irregularities in the K-line data timestamps and
        return a DataFrame with discrepancies.

        Returns
        -------
        pd.DataFrame
            Returns a pandas DataFrame with discrepancies in
            timestamps.
        """
        ohlcv_columns = ["open", "high", "low", "close", "volume"]

        time_interval = pd.Timedelta(self.interval)

        date_check_df = self.data_frame.copy()
        date_check_df["actual_date"] = date_check_df.index
        date_check_df["previous_date"] = date_check_df["actual_date"].shift()

        date_check_df = date_check_df[
            ohlcv_columns
            + ["actual_date", "previous_date"]
        ]

        date_check_df["timedelta"] = (
            date_check_df["actual_date"]
            - date_check_df["previous_date"]
        )
        date_check_df = date_check_df.iloc[1:]

        date_check_df = date_check_df[
            date_check_df["timedelta"] != time_interval
        ]

        return date_check_df
