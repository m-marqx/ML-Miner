import itertools

import pandas as pd
import numpy as np

from machine_learning.adjust_predicts import adjust_predict_one_side
from machine_learning.model_creator import ModelCreator
from custom_exceptions.invalid_arguments import InvalidArgumentError


class OnchainModelMiner:
    """
    Class to search for the best model parameters.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing the OHLC data.
    target : pd.Series
        The target series containing the adjusted values.
    max_trades : int, optional
        The maximum number of trades
        (default : 3).
    off_days : int, optional
        The number of days to consider for trade calculation
        (default : 7).
    side : int, optional
        The side of the trade to adjust
        (default : 1).

    Attributes
    ----------
    ohlc : list
        The list of the OHLC columns.
    max_trades : int
        The maximum number of trades.
    off_days : int
        The number of days to consider for trade calculation.
    dataframe : pd.DataFrame
        The input dataframe containing the OHLC data.
    target : pd.Series
        The target series containing the adjusted values.
    ma_types : list
        The list of the moving averages types.
    ma_type_combinations : np.array
        The array of the moving averages type combinations.
    features : list
        The list of the features.
    random_features : np.array
        The array of the random features.
    adj_targets : pd.Series
        The adjusted target series.
    empty_dict : dict
        The empty dictionary.
    feat_parameters : None
        The features variables.

    Methods:
    -------
    search_model(test_index: int, pct_adj: float = 0.5, train_in_middle:
    bool = True)
        Search for the best model parameters.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: pd.Series,
        test_index: int,
        max_trades: int = 3,
        off_days: int = 7,
        side: int = 1,
        bins: int = 10,
        verbose: bool = False,
        train_in_middle: bool = True,
    ):
        """
        Parameters
        ----------
        predict : pd.Series
            The input series containing the predicted values.
        max_trades : int
            The maximum number of trades.
        target_days : int
            The number of days to consider for trade calculation.
        side : int, optional
            The side of the trade to adjust
            (default : 1).
        """
        self.ohlc: list[str] = ["open", "high", "low", "close"]
        self.max_trades: int = max_trades
        self.off_days: int = off_days
        self.side: int = side
        self.test_index: int = test_index

        self.dataframe = dataframe.copy()
        self.target = target.copy()

        self.adj_targets = adjust_predict_one_side(
            self.target,
            max_trades,
            off_days,
            side,
        )

        self.train_in_middle: bool = train_in_middle

        self.empty_dict: dict[str, None] = {
            "feat_parameters": None,
            "hyperparameters": None,
            "metrics_results": None,
            "drawdown_full_test": None,
            "drawdown_full_val": None,
            "drawdown_adj_test": None,
            "drawdown_adj_val": None,
            "expected_return_test": None,
            "expected_return_val": None,
            "precisions_test": None,
            "precisions_val": None,
            "support_diff_test": None,
            "support_diff_val": None,
            "total_operations_test": None,
            "total_operations_val": None,
            "total_operations_pct_test": None,
            "total_operations_pct_val": None,
            "r2_in_2023": None,
            "r2_val": None,
            "ols_coef_2022": None,
            "ols_coef_val": None,
            "test_index": None,
            "total_time": None,
            "return_ratios": None,
            "side": None,
            "max_trades": None,
            "off_days": None,
        }

        self.ma_types: list[str] = ["sma", "ema", "rma"]
        self.ma_type_combinations: np.array = np.array([])
        self.ta_features: np.array = np.array([])
        self.onchain_features: np.array = np.array([])
        self.onchain_rng = None
        self.ta_rng = None
        self.hyperparameter_rng = None
        self.bins = bins
        self.verbose = verbose

        self.model_creator = ModelCreator(
            self.dataframe,
            self.test_index,
            self.bins,
            self.verbose,
            self.train_in_middle,
        )

    def calculate_combinations(self, options_list: list) -> np.array:
        """
        Calculate the combinations.

        Returns
        -------
        np.array
            The list of the combinations.
        """
        combinations_list = []

        for idx, _ in enumerate(options_list, 1):
            if idx > 3:
                break
            combinations_list += list(
                itertools.combinations(options_list, idx)
            )

        return np.array(combinations_list, dtype="object")

