import pathlib
from typing import Literal
import itertools
import time

import pandas as pd
import numpy as np

from custom_exceptions.invalid_arguments import InvalidArgumentError

from machine_learning.features.feature_adapter import FeaturesAdapter
from machine_learning.model_features import ModelFeatures
from machine_learning.features.onchain_features import OnchainFeatures
from machine_learning.model_builder import calculate_model, adjust_max_trades
from machine_learning.adjust_predicts import adjust_predict_one_side
from machine_learning.model_metrics import ModelMetrics

from utils.log_handler import create_logger


class ModelCreator:
    """
    Class for managing and creating machine learning models.

    Attributes
    ----------
    dataset : pd.DataFrame
        Input dataset containing the main data for model creation.
    test_index : int
        Index position to split data into training and testing sets.
    features : pd.DataFrame
        DataFrame containing the features used for model creation.
    train_in_middle : bool
        If True, enables training in the middle of the dataset.
    logger : logging.Logger
        Logger object for logging messages.
    hyperparameter_rng : np.random.Generator
        Random number generator for hyperparameters.
    onchain_rng : np.random.Generator
        Random number generator for onchain features.
    onchain_features : np.array
        Array containing the onchain features.
    model_features_adapter : FeaturesAdapter
        Adapter object for model features.
    onchain_features_adapter : FeaturesAdapter
        Adapter object for onchain features.
    model_features_create_methods : list
        List containing the names of the model features creation methods.
    onchain_features_create_methods : list
        List containing the names of the onchain features creation methods.
    features_dataset : pd.DataFrame
        DataFrame containing the features used for model creation.
    empty_dict : dict
        Dictionary containing the default values for the model results.
    """
    def __init__(
        self,
        dataset: pd.DataFrame,
        onchain_data: pd.DataFrame | None,
        test_index: int,
        verbose: bool = False,
        train_in_middle: bool = False,
    ):
        """
        Initialize ModelCreator class for managing and creating machine
        learning models.

        Parameters
        ----------
        dataset : pd.DataFrame
            Input dataset containing the main data for model creation.
        onchain_data : pd.DataFrame or None
            Dataset containing blockchain-related data. If None, data
            will be loaded from default path.
        test_index : int
            Index position to split data into training and testing sets.
        verbose : bool
            If True, enables detailed logging output.
            (default: False)
        train_in_middle : bool
            If True, enables training in the middle of the dataset.
            (default: False)

        Raises
        ------
        InvalidArgumentError
            If any of the input parameters are of incorrect type.
        """
        if not isinstance(dataset, pd.DataFrame):
            raise InvalidArgumentError("dataset should be a pandas DataFrame")
        if (
            not isinstance(onchain_data, pd.DataFrame)
            and onchain_data is not None
        ):
            raise InvalidArgumentError(
                "onchain_data should be a pandas DataFrame or None"
            )
        if not isinstance(test_index, int):
            raise InvalidArgumentError("test_index should be an integer")
        if not isinstance(verbose, bool):
            raise InvalidArgumentError("verbose should be a boolean")
        if not isinstance(train_in_middle, bool):
            raise InvalidArgumentError("train_in_middle should be a boolean")

        self.dataset = dataset
        self.test_index = test_index
        self.features = pd.DataFrame()
        self.train_in_middle = train_in_middle
        self.logger = create_logger("ModelCreator", verbose)
        self.hyperparameter_rng = None
        self.onchain_rng = None
        self.onchain_features = None

        model_features = ModelFeatures(
            dataset=self.dataset,
            test_index=self.test_index,
            normalize=True,
        )

        self.model_metrics = None

        if onchain_data is None:
            path = pathlib.Path(
                "data",
                "onchain",
                "BTC",
                "block_stats_fragments",
            )

            onchain_data = pd.read_parquet(path)

            onchain_data["time"] = pd.to_datetime(
                onchain_data["time"], unit="s"
            )
            onchain_data = onchain_data.set_index("time")

        onchain_features = OnchainFeatures(
            onchain_data=onchain_data,
            test_index=self.test_index,
        )

        self.onchain_features = onchain_features

        self.model_features_adapter = FeaturesAdapter(model_features)
        self.onchain_features_adapter = FeaturesAdapter(onchain_features)

        self.model_features_create_methods = [
            attr for attr in dir(model_features) if attr.startswith("create")
        ]

        self.onchain_features_create_methods = [
            attr for attr in dir(onchain_features) if attr.startswith("create")
        ]

        self.features_dataset = pd.DataFrame()

        self.empty_dict = {
            "onchain_features": None,
            "hyperparameters": None,
            "model_dates_interval": None,
            "linear_accumulated_return_test": None,
            "linear_accumulated_return_val": None,
            "exponential_accumulated_return_test": None,
            "exponential_accumulated_return_val": None,
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
            "train_in_middle": None,
            "total_time": None,
            "return_ratios": None,
            "side": None,
            "max_trades": None,
            "off_days": None,
        }

    def set_bins(self, bins: int) -> pd.DataFrame:
        """
        Set the number of bins for the features.

        Parameters
        ----------
        bins : int
            The number of bins to set for the features.

        Returns
        -------
        ModelCreator : ModelCreator
            The ModelCreator object with the updated number of bins.

        """
        self.model_features_adapter.run_method("set_bins", bins)
        self.onchain_features_adapter.run_method("set_bins", bins)
        return self

    def feature_docstring(self, method_name: str) -> str:
        """
        Retrieve the docstring for a given feature creation method.

        Parameters
        ----------
        method_name : str
            The name of the feature creation method for which to
            retrieve the docstring.

        Returns
        -------
        str
            The docstring of the specified feature creation method.

        Raises
        ------
        InvalidArgumentError
            If the specified method name is not found in either
            model_features_create_methods or
            onchain_features_create_methods.
        """
        if method_name in [*self.model_features_create_methods]:
            self.model_features_adapter.get_method_docstring(method_name)

        elif method_name in [*self.onchain_features_create_methods]:
            self.onchain_features_adapter.get_method_docstring(method_name)

        else:
            raise InvalidArgumentError(f"method {method_name} not found")

