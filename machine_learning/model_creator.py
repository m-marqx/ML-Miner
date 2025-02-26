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

    def calculate_feature(
        self,
        adapter,
        method_name: str,
        default_columns: list | pd.Index,
        *args,
        **kwargs,
    ) -> None:
        """
        Execute a feature calculation method and store the resulting features.

        This method runs a specified feature calculation method using the provided adapter,
        identifies new columns generated by the method, and adds these columns to the
        instance's feature dataset. New columns are determined by comparing the adapter's
        dataset columns before and after method execution.

        Parameters
        ----------
        adapter : object
            Adapter instance with `run_method` and `dataset` attributes. The `run_method`
            is used to execute the feature calculation, and `dataset` contains the resulting data.
        method_name : str
            Name of the method to execute via the adapter. This method should generate
            new features as columns in the adapter's dataset.
        default_columns : list | pd.Index
            List or Index of column names present in the adapter's dataset before
            executing `method_name`. Used to identify newly created columns.
        *args : tuple
            Variable length argument list passed to `adapter.run_method`.
        **kwargs : dict
            Arbitrary keyword arguments passed to `adapter.run_method`.

        Returns
        -------
        None
            This method does not return a value; it modifies `self.features_dataset` in-place.

        Notes
        -----
        - New columns are added to `self.features_dataset` only if they do not already exist.
        - The method logs the creation of each new feature column at the INFO level.
        """
        self.logger.info("Creating feature: %s", method_name)

        adapter.run_method(method_name, *args, **kwargs)

        new_columns = []
        new_columns += adapter.dataset.columns.difference(
            default_columns, sort=False
        ).tolist()

        for col in new_columns:
            self.features_dataset[col] = adapter.dataset[col]
            self.logger.info("Feature %s created", col)

    def create_feature(
        self,
        method_name: str,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Creates a new feature using the specified method.
        This method creates a new feature by applying the specified
        calculation method to either model features or onchain
        features, depending on the method name.

        Parameters
        ----------
        method_name : str
            Name of the method to create the feature. Must exist in
            either model_features_create_methods or
            onchain_features_create_methods.
        *args : tuple
            Variable length argument list to pass to the calculation
            method.
        **kwargs : dict
            Arbitrary keyword arguments to pass to the calculation
            method.
        Returns
        -------
        pd.DataFrame
            The instance of the class with the new feature added to the
            appropriate features adapter.
        Raises
        ------
        InvalidArgumentError
            If the specified method_name is not found in either
            create_methods list.

        Notes
        -----
        The method automatically determines whether to apply the
        calculation to model features or onchain features based on the
        method_name provided.
        """
        model_features_columns = self.model_features_adapter.dataset.columns

        onchain_features_columns = (
            self.onchain_features_adapter.base_dataset.columns
        )

        if method_name in [*self.model_features_create_methods]:
            self.calculate_feature(
                self.model_features_adapter,
                method_name,
                model_features_columns,
                *args,
                **kwargs,
            )

        elif method_name in [*self.onchain_features_create_methods]:
            self.calculate_feature(
                self.onchain_features_adapter,
                method_name,
                onchain_features_columns,
                *args,
                **kwargs,
            )

        else:
            raise InvalidArgumentError(f"method {method_name} not found")

        return self

    def create_model(
        self,
        max_trades: int = 3,
        off_days: int = 7,
        side: int = 1,
        cutoff_point: int = 5,
        **hyperparams,
    ) -> pd.DataFrame:
        """
        Create and adjust a machine learning model based on provided features.

        This method concatenates the original dataset with the computed features,
        drops rows with missing target values, and builds a model using the
        calculate_model function. The model's predictions are adjusted and the
        final result is obtained with a maximum number of trades and other trading
        parameters adjusted.

        Parameters
        ----------
        max_trades : int, optional
            Maximum number of allowed trades (default is 3).
        off_days : int, optional
            Number of off days between trades (default is 7).
        side : int, optional
            Trading side parameter for adjustments (default is 1).
        cutoff_point : int, optional
            Cutoff point used in the model building process (default is 5).
        **hyperparams : dict
            Additional hyperparameters to pass to the calculate_model function.

        Returns
        -------
        tuple
            A tuple containing:
            - pd.DataFrame: The adjusted model output after applying max trades.
            - index_splits: The index positions used to split the dataset.
            - target_series: The target series used for modeling.
            - adjusted_target: The target adjusted for one side prediction.
        """
        df_columns = self.features_dataset.columns.tolist()
        features = [col for col in df_columns if col.endswith("_feat")]

        dates = self.features_dataset.index[0], self.features_dataset.index[-1]

        data_frame = pd.concat([self.dataset, self.features_dataset], axis=1)
        data_frame = data_frame[dates[0] : dates[1]].dropna(subset=["Target"])

        adjusted_target = adjust_predict_one_side(
            self.dataset["Target_bin"],
            max_trades,
            off_days,
            side,
        )

        mh2, _, _, _, _, _, _, _, _, target_series, index_splits = (
            calculate_model(
                dataset=data_frame,
                feats=features,
                test_index=self.test_index,
                plot=False,
                output="All",
                long_only=False,
                train_in_middle=self.train_in_middle,
                cutoff_point=cutoff_point,
                dev=False,
                **hyperparams,
            )
        )

        mh2["Liquid_Result"] = np.where(
            mh2["Predict"] != side,
            0,
            mh2["Liquid_Result"],
        )

        return (
            adjust_max_trades(mh2, off_days, max_trades, 0.5, side),
            index_splits,
            target_series,
            adjusted_target,
        )

    def calculate_combinations(self, options_list: list) -> np.array:
        """
        Calculate the combinations.

        Parameters
        ----------
        options_list : list
            The list of options to calculate the combinations.

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

    def calculate_onchain_features(self) -> pd.DataFrame:
        """
        Calculate the onchain features.

        Returns
        -------
        pd.DataFrame
            The dataframe containing the onchain features.
        """
        txs_infos = ["txs", "utxo_increase"]
        total_infos = ["total_out", "total_size", "total_weight", "totalfee"]
        avg_infos = ["avgfee", "avgfeerate", "avgtxsize"]
        # max_infos = ["maxfee", "maxfeerate", "maxtxsize"]
        median_infos = ["mediantime"]

        all_infos = [
            *avg_infos,
            # *max_infos,
            *median_infos,
            *total_infos,
            *txs_infos,
        ]

        self.onchain_features = self.calculate_combinations(all_infos)

        return self

    def generate_hyperparameters(self) -> dict:
        """
        Generate a dictionary of hyperparameters for the model.

        Returns
        -------
        dict
            A dictionary containing the following hyperparameters:
            - iterations : int
                The number of iterations for the model.
            - learning_rate : float
                The learning rate for the model.
            - depth : int
                The depth of the model.
            - min_child_samples : int
                The minimum number of samples required to create a new
                node in the model.
            - colsample_bylevel : float
                The fraction of columns to be randomly selected for
                each level in the model.
            - subsample : float
                The fraction of samples to be randomly selected for
                each tree in the model.
            - reg_lambda : int
                The regularization lambda value for the model.
            - use_best_model : bool
                Whether to use the best model found during training.
            - eval_metric : str
                The evaluation metric to be used during training.
            - random_seed : int
                The random seed value for the model.
            - silent : bool
                Whether to print messages during training.
        """
        hyperparameter_seed = np.random.randint(1, 100_000_000)
        self.hyperparameter_rng = np.random.default_rng(hyperparameter_seed)

        return {
            "iterations": 1000,

            "learning_rate": self.hyperparameter_rng.choice(
                np.arange(0.01, 1.01, 0.01)
            ),

            "depth": self.hyperparameter_rng.choice(range(1, 12, 1)),

            "min_child_samples": self.hyperparameter_rng.choice(
                range(1, 21, 1)
            ),

            "colsample_bylevel": self.hyperparameter_rng.choice(
                np.arange(0.1, 1.01, 0.01)
            ),

            "subsample": self.hyperparameter_rng.choice(
                np.arange(0.1, 1.01, 0.01)
            ),

            "reg_lambda": self.hyperparameter_rng.choice(range(1, 206, 1)),

            "use_best_model": True,

            "eval_metric": self.hyperparameter_rng.choice(
                ["Logloss", "AUC", "F1", "Precision", "Recall", "PRAUC"]
            ),

            "random_seed": hyperparameter_seed,
            "silent": True,
        }

    def beta_generate_hyperparameters(self, hyperparams_ranges) -> dict:
        """
        Generate a dictionary of hyperparameters for the model with
        custom ranges.

        This method allows customizing the ranges for various
        hyperparameters used in model creation. Default ranges are used
        for any parameters not specified.

        Parameters
        ----------
        hyperparams_ranges : dict
            Dictionary containing custom ranges for hyperparameters.
            Supported keys are:
            - learning_rate : tuple(float, float)
                Range for learning rate
                (default: (0.01, 1.0))
            - depth : tuple(int, int)
                Range for tree depth
                (default: (1, 12))
            - min_child_samples : tuple(int, int)
                Range for minimum child samples
                (default: (1, 20))
            - colsample_bylevel : tuple(float, float)
                Range for column sampling by level
                (default: (0.1, 1.0))
            - subsample : tuple(float, float)
                Range for subsample ratio
                (default: (0.1, 1.0))
            - reg_lambda : tuple(int, int)
                Range for regularization lambda
                (default: (1, 20))
            - eval_metric : list
                List of evaluation metrics to choose from
                (default:
                ["Logloss", "AUC", "F1", "Precision", "Recall", "PRAUC"])
            - random_seed : tuple(int, int)
                Range for random seed
                (default: (1, 50000))

        Returns
        -------
        dict
            Dictionary containing randomly sampled hyperparameters
            with keys:
            - iterations : int
                The number of iterations for the model.
            - learning_rate : float
                The learning rate for the model.
            - depth : int
                The depth of the model.
            - min_child_samples : int
                The minimum number of samples required to create a new
                node in the model.
            - colsample_bylevel : float
                The fraction of columns to be randomly selected for
                each level in the model.
            - subsample : float
                The fraction of samples to be randomly selected for
                each tree in the model.
            - reg_lambda : int
                The regularization lambda value for the model.
            - use_best_model : bool
                Whether to use the best model found during training.
            - eval_metric : str
                The evaluation metric to be used during training.
            - random_seed : int
                The random seed value for the model.
            - silent : bool
                Whether to print messages during training.

        Notes
        -----
        The method uses numpy's random number generator for sampling
        values within the specified ranges. Float parameters use a
        step size of 0.01 while integer parameters use a step size of 1.
        """
        eval_metrics = ["Logloss", "AUC", "F1", "Precision", "Recall", "PRAUC"]

        hyperparams_ranges["learning_rate"] = hyperparams_ranges.get(
            "learning_rate", (0.01, 1.0)
        )

        hyperparams_ranges["depth"] = hyperparams_ranges.get(
            "depth", (1, 12)
        )

        hyperparams_ranges["min_child_samples"] = hyperparams_ranges.get(
            "min_child_samples", (1, 20)
        )

        hyperparams_ranges["colsample_bylevel"] = hyperparams_ranges.get(
            "colsample_bylevel", (0.1, 1.0)
        )

        hyperparams_ranges["subsample"] = hyperparams_ranges.get(
            "subsample", (0.1, 1.0)
        )

        hyperparams_ranges["reg_lambda"] = hyperparams_ranges.get(
            "reg_lambda", (1, 205)
        )

        hyperparams_ranges["eval_metric"] = hyperparams_ranges.get(
            "eval_metric", eval_metrics
        )

        hyperparams_ranges["random_seed"] = hyperparams_ranges.get(
            "random_seed", (1, 100_000_000)
        )

        float_step = 0.01
        int_step = 1

        hyperparameter_seed = np.random.randint(1, 100_000_000)
        self.hyperparameter_rng = np.random.default_rng(hyperparameter_seed)

        return {
            "iterations": 1000,

            "learning_rate": self.hyperparameter_rng.choice(
                np.arange(
                    hyperparams_ranges["learning_rate"][0],
                    hyperparams_ranges["learning_rate"][1] + float_step,
                    float_step
                )
            ),

            "depth": self.hyperparameter_rng.choice(
                range(
                    hyperparams_ranges["depth"][0],
                    hyperparams_ranges["depth"][1] + int_step,
                    int_step
                )
            ),

            "min_child_samples": self.hyperparameter_rng.choice(
                range(
                    hyperparams_ranges["min_child_samples"][0],
                    hyperparams_ranges["min_child_samples"][1] + int_step,
                    int_step
                )
            ),

            "colsample_bylevel": self.hyperparameter_rng.choice(
                np.arange(
                    hyperparams_ranges["colsample_bylevel"][0],
                    hyperparams_ranges["colsample_bylevel"][1] + float_step,
                    float_step,
                )
            ),

            "subsample": self.hyperparameter_rng.choice(
                np.arange(
                    hyperparams_ranges["subsample"][0],
                    hyperparams_ranges["subsample"][1] + float_step,
                    float_step,
                )
            ),

            "reg_lambda": self.hyperparameter_rng.choice(
                range(
                    hyperparams_ranges["reg_lambda"][0],
                    hyperparams_ranges["reg_lambda"][1] + int_step,
                    int_step,
                )
            ),

            "use_best_model": True,

            "eval_metric": self.hyperparameter_rng.choice(
                hyperparams_ranges["eval_metric"]
            ),

            "random_seed": hyperparameter_seed,
            "silent": True,
        }

    def generate_onchain_features(self) -> dict:
        """
        Generate the onchain feature.

        Returns
        -------
        dict
            The dictionary containing the features used.
        """
        try:
            self.calculate_onchain_features()

            onchain_seed = np.random.randint(1, 100_000_000)
            self.onchain_rng = np.random.default_rng(onchain_seed)

            if len(self.onchain_features) > 1:
                reduced_onchain_features = self.onchain_rng.choice(
                    self.onchain_features
                )
            else:
                reduced_onchain_features = reduced_onchain_features[0]

            features_used = {}

            for feature in reduced_onchain_features:
                length_windows = self.onchain_rng.choice(
                    range(2, 30),
                    2,
                    replace=False,
                )

                short_window = int(length_windows.min())
                long_window = int(length_windows.max())

                bins = self.onchain_rng.integers(10, 31)
                self.set_bins(bins)

                self.create_feature(
                    "create_std_ratio_feature",
                    [feature],
                    "D",
                    short_window,
                    long_window,
                )

                features_used[feature] = {
                    "short_window": short_window,
                    "long_window": long_window,
                    "bins": bins,
                }

            features_used["onchain_seed"] = onchain_seed

            return features_used
        except Exception as e:
            print(
                f"""
                onchain features : {reduced_onchain_features}
                seed : {onchain_seed}
                """
            )
            raise type(e)(f"Error creating onchain features: {e}") from e

    def create_onchain_features(
        self,
        feature: str,
        short_window: int,
        long_window: int,
        bins: int,
    ):
        """
        Create the onchain feature using a standard ratio feature
        calculation.

        This method sets the number of bins for the onchain features,
        and then creates a new feature column by applying the
        "create_std_ratio_feature" method. The feature is created for
        the specified feature name using the provided short and long
        window lengths.

        Parameters
        ----------
        feature : str
            The name of the feature for which the onchain standard
            ratio is to be created.
        short_window : int
            The size of the short window used in the calculation.
        long_window : int
            The size of the long window used in the calculation.
        bins : int
            The number of bins to use when configuring the feature.

        Returns
        -------
        self : ModelCreator
            Returns the current instance with the newly created onchain
            feature added.
        """
        self.set_bins(bins)

        self.create_feature(
            "create_std_ratio_feature",
            [feature],
            "D",
            short_window,
            long_window,
        )
        return self

    def create_multiple_onchain_features(
        self,
        onchain_features_params: dict,
    ) -> pd.DataFrame:
        """
        Create multiple onchain features based on provided parameters.

        This method processes a dictionary of onchain feature parameters
        and creates features for each specified feature using the
        create_onchain_features method.

        Parameters
        ----------
        onchain_features_params : dict
            Dictionary containing parameters for onchain features
            creation.
            Each feature should have a nested dictionary with keys:
                - 'short_window': Window size for short-term calculation
                - 'long_window': Window size for long-term calculation
                - 'bins': Number of bins for feature discretization
            May optionally contain an 'onchain_seed' key which will be
            ignored.

        Returns
        -------
        pd.DataFrame
            The current ModelCreator instance with newly created
            features added.

        Notes
        -----
        The method will ignore the 'onchain_seed' key if present in the
        parameters dictionary. All numeric parameters are converted to
        integers before use.
        """
        onchain_features: dict[str, float] = (
            pd.Series(onchain_features_params)
            .dropna()
            .to_dict()
        )

        if "onchain_seed" in onchain_features.keys():
            onchain_features.pop("onchain_seed")

        for feature in onchain_features:
            feature_params: dict[str, float] = onchain_features[feature]
            short_window: int = int(feature_params["short_window"])
            long_window: int = int(feature_params["long_window"])

            bins: int = int(feature_params["bins"])

            self.create_onchain_features(
                feature=feature,
                short_window=short_window,
                long_window=long_window,
                bins=bins,
            )

        return self

    def new_generate_onchain_features(self) -> dict:
        """
        Generate the onchain feature.

        Returns
        -------
        dict
            The dictionary containing the features used.
        """
        try:
            self.calculate_onchain_features()

            onchain_seed = np.random.randint(1, 100_000_000)
            self.onchain_rng = np.random.default_rng(onchain_seed)

            if len(self.onchain_features) > 1:
                reduced_onchain_features = self.onchain_rng.choice(
                    self.onchain_features
                )
            else:
                reduced_onchain_features = reduced_onchain_features[0]

            features_used = {}

            for feature in reduced_onchain_features:
                length_windows = self.onchain_rng.choice(
                    range(2, 30),
                    2,
                    replace=False,
                )

                short_window = int(length_windows.min())
                long_window = int(length_windows.max())

                bins = self.onchain_rng.integers(10, 31)

                self.create_onchain_features(
                    feature,
                    short_window,
                    long_window,
                    bins,
                )

                features_used[feature] = {
                    "short_window": short_window,
                    "long_window": long_window,
                    "bins": bins,
                }

            features_used["onchain_seed"] = onchain_seed

            return features_used
        except Exception as e:
            print(
                f"""
                onchain features : {reduced_onchain_features}
                seed : {onchain_seed}
                """
            )
            raise type(e)(f"Error creating onchain features: {e}") from e

    def beta_generate_onchain_features(self, seed) -> dict:
        """
        Generate the onchain feature.

        Returns
        -------
        dict
            The dictionary containing the features used.
        """
        try:
            self.calculate_onchain_features()

            onchain_seed = int(seed)
            self.onchain_rng = np.random.default_rng(onchain_seed)

            if len(self.onchain_features) > 1:
                reduced_onchain_features = self.onchain_rng.choice(
                    self.onchain_features
                )
            else:
                reduced_onchain_features = reduced_onchain_features[0]

            features_used = {}

            for feature in reduced_onchain_features:
                length_windows = self.onchain_rng.choice(
                    range(2, 30),
                    2,
                    replace=False,
                )

                short_window = int(length_windows.min())
                long_window = int(length_windows.max())

                bins = self.onchain_rng.integers(10, 31)

                self.create_onchain_features(
                    feature,
                    short_window,
                    long_window,
                    bins,
                )

                features_used[feature] = {
                    "short_window": short_window,
                    "long_window": long_window,
                    "bins": bins,
                }

            features_used["onchain_seed"] = onchain_seed

            return features_used
        except Exception as e:
            print(
                f"""
                onchain features : {reduced_onchain_features}
                seed : {onchain_seed}
                """
            )
            raise type(e)(f"Error creating onchain features: {e}") from e

    def create_onchain_catboost_model(
        self,
        onchain_features: dict,
        max_trades: int = 3,
        off_days: int = 7,
        side: int = 1,
        cutoff_point: int = 5,
        return_type: Literal["results", "metrics"] = "results",
        **hyperparameters,
    ) -> dict:
        """
        Create a CatBoost model using generated onchain features.

        This method creates onchain features based on the provided parameters,
        builds a CatBoost model, and returns either the raw model results or
        a dictionary of evaluation metrics and additional model details based
        on the specified return type.

        Parameters
        ----------
        onchain_features : dict
            Dictionary containing onchain feature parameters which are used to
            create new onchain features.
        max_trades : int, optional
            Maximum number of allowed trades
            (default: 3).
        off_days : int, optional
            Number of off days between trades
            (default: 7).
        side : int, optional
            Trading side parameter used for adjusting predictions
            (default: 1).
        cutoff_point : int, optional
            Cutoff point used in the model creation process
            (default: 5).
        return_type : {"results", "metrics"}, optional
            Specifies the kind of output to return. If "results", the
            method returns the raw model results. If "metrics", it
            returns a dictionary containing evaluation metrics and other
            model information
            (default: "results").
        **hyperparameters : dict
            Additional hyperparameters to pass to the model hyperparams

        Returns
        -------
        dict
            If return_type is "results", returns the raw model results as a DataFrame.
            If return_type is "metrics", returns a dictionary containing evaluation metrics,
            model dates interval, accumulated returns, drawdowns, total operations, and additional
            model parameters.

        Raises
        ------
        Exception
            Raises an exception with extra debugging information if an error occurs
            during model creation.
        """
        start: float = time.perf_counter()

        self.create_multiple_onchain_features(onchain_features)

        results, index_splits, target_series, adj_targets = (
            self.create_model(
                max_trades=max_trades,
                off_days=off_days,
                side=side,
                cutoff_point=cutoff_point,
                **hyperparameters,
            )
        )

        if return_type == "results":
            print("Split dates: ", index_splits)
            return results

        try:
            model_dates_interval = pd.Interval(
                results.index[0],
                results.index[-1],
                closed='both',
            )

            self.model_metrics = ModelMetrics(
                self.train_in_middle,
                index_splits,
                results,
                self.test_index,
            )

            accumulated_returns = (
                self.model_metrics.calculate_accumulated_returns("all")
            )

            if min(accumulated_returns) <= 1:
                return self.empty_dict

            linear_return_test = accumulated_returns[0]
            linear_return_val = accumulated_returns[1]
            exponential_return_test = accumulated_returns[2]
            exponential_return_val = accumulated_returns[3]

            test_buys, val_buys = (
                self.model_metrics.calculate_model_recommendations()
            )

            metrics_results = self.model_metrics.calculate_result_metrics(
                target_series,
                7,
            )

            if min(metrics_results["precisions"]) < 0.52:
                return self.empty_dict

            total_operations, total_operations_pct = (
                self.model_metrics.calculate_total_operations(
                    test_buys,
                    val_buys,
                    max_trades,
                    off_days,
                    side,
                )
            )

            drawdowns = self.model_metrics.calculate_drawdowns()

            return_ratios = self.model_metrics.calculate_result_ratios()

            support_diffs = (
                self.model_metrics
                .calculate_result_support(adj_targets, side)
            )

            # get the results from the bear market that started in 2022
            bearmarket_2022 = results.loc["2021-08-11":"2023-01-01"]

            r2_test, r2_val, ols_coef_test, ols_coef_val = (
                self.model_metrics.set_results_test(
                    bearmarket_2022
                ).calculate_ols_metrics()
            )

            return {
                "onchain_features": onchain_features,
                "hyperparameters": hyperparameters,
                "metrics_results": [metrics_results],
                "model_dates_interval": model_dates_interval,
                "linear_accumulated_return_test": linear_return_test,
                "linear_accumulated_return_val": linear_return_val,
                "exponential_accumulated_return_test": exponential_return_test,
                "exponential_accumulated_return_val": exponential_return_val,
                "drawdown_full_test": drawdowns[0],
                "drawdown_full_val": drawdowns[1],
                "drawdown_adj_test": drawdowns[2],
                "drawdown_adj_val": drawdowns[3],
                "expected_return_test": metrics_results["expected_return_test"],
                "expected_return_val": metrics_results["expected_return_val"],
                "precisions_test": metrics_results["precisions_test"],
                "precisions_val": metrics_results["precisions_val"],
                "support_diff_test": support_diffs[0],
                "support_diff_val": support_diffs[1],
                "total_operations_test": total_operations[0],
                "total_operations_val": total_operations[1],
                "total_operations_pct_test": total_operations_pct[0],
                "total_operations_pct_val": total_operations_pct[1],
                "r2_in_2023": r2_test,
                "r2_val": r2_val,
                "ols_coef_2022": ols_coef_test,
                "ols_coef_val": ols_coef_val,
                "test_index": self.test_index,
                "train_in_middle": self.train_in_middle,
                "total_time": time.perf_counter() - start,
                "return_ratios": return_ratios,
                "side": side,
                "max_trades": max_trades,
                "off_days": off_days,
            }

        except Exception as e:
            raise type(e)(
                f"Error creating model: {e}"
                + f"\n\n{onchain_features}"
                + f"\n{hyperparameters}"
            ) from e

    def generate_onchain_catboost_model(
        self,
        max_trades: int = 3,
        off_days: int = 7,
        side: int = 1,
        cutoff_point: int = 5,
    ) -> dict:
        """
        Create the CatBoost model.

        Returns
        -------
        dict
            The dictionary containing the hyperparameters and the
            onchain features used.
        """
        start: float = time.perf_counter()

        hyperparameters = self.generate_hyperparameters()
        onchain_features = self.generate_onchain_features()

        try:
            results, index_splits, target_series, adj_targets = (
                self.create_model(
                    max_trades=max_trades,
                    off_days=off_days,
                    side=side,
                    cutoff_point=cutoff_point,
                    **hyperparameters,
                )
            )

            self.model_metrics = ModelMetrics(
                self.train_in_middle,
                index_splits,
                results,
                self.test_index,
            )

            results_test = self.model_metrics.results_test
            results_val = self.model_metrics.results_val

            test_buys, val_buys = (
                self.model_metrics.calculate_model_recommendations()
            )

            is_hold_results = (
                len(test_buys) <= 1
                or len(val_buys) <= 1
                or len(results_test["Liquid_Result"].value_counts()) <= 1
                or len(results_val["Liquid_Result"].value_counts()) <= 1
            )

            if is_hold_results:
                return self.empty_dict

            metrics_results = self.model_metrics.calculate_result_metrics(
                target_series,
                7,
            )

            if min(metrics_results["precisions"]) < 0.52:
                return self.empty_dict

            # get the results from the bear market that started in 2022
            bearmarket_2022 = results.loc["2021-08-11":"2023-01-01"]

            r2_test, r2_val, ols_coef_test, ols_coef_val = (
                self.model_metrics.set_results_test(
                    bearmarket_2022
                ).calculate_ols_metrics()
            )

            total_operations, total_operations_pct = (
                self.model_metrics.calculate_total_operations(
                    test_buys,
                    val_buys,
                    max_trades,
                    off_days,
                    side,
                )
            )

            drawdowns = self.model_metrics.calculate_drawdowns()

            return_ratios = self.model_metrics.calculate_result_ratios()

            support_diffs = self.model_metrics.calculate_result_support(
                adj_targets, side
            )

            return {
                "onchain_features": onchain_features,
                "hyperparameters": hyperparameters,
                "metrics_results": [metrics_results],
                "model_dates_interval": None,
                "linear_accumulated_return_test": None,
                "linear_accumulated_return_val": None,
                "exponential_accumulated_return_test": None,
                "exponential_accumulated_return_val": None,
                "drawdown_full_test": drawdowns[0],
                "drawdown_full_val": drawdowns[1],
                "drawdown_adj_test": drawdowns[2],
                "drawdown_adj_val": drawdowns[3],
                "expected_return_test": metrics_results["expected_return_test"],
                "expected_return_val": metrics_results["expected_return_val"],
                "precisions_test": metrics_results["precisions_test"],
                "precisions_val": metrics_results["precisions_val"],
                "support_diff_test": support_diffs[0],
                "support_diff_val": support_diffs[1],
                "total_operations_test": total_operations[0],
                "total_operations_val": total_operations[1],
                "total_operations_pct_test": total_operations_pct[0],
                "total_operations_pct_val": total_operations_pct[1],
                "r2_in_2023": r2_test,
                "r2_val": r2_val,
                "ols_coef_2022": ols_coef_test,
                "ols_coef_val": ols_coef_val,
                "test_index": self.test_index,
                "train_in_middle": self.train_in_middle,
                "total_time": time.perf_counter() - start,
                "return_ratios": return_ratios,
                "side": side,
                "max_trades": max_trades,
                "off_days": off_days,
            }

        except Exception as e:
            raise type(e)(
                f"Error creating model: {e}"
                + f"\n\n{onchain_features}"
                + f"\n{hyperparameters}"
            ) from e

    def beta_generate_onchain_catboost_model(
            self,
            hyperparams_ranges: dict,
            max_trades: int = 3,
            off_days: int = 7,
            side: int = 1,
            cutoff_point: int = 5,
        ) -> dict:
        """
            Create the CatBoost model.

            Returns
            -------
            dict
                The dictionary containing the hyperparameters and the
                onchain features used.
            """
        start: float = time.perf_counter()

        hyperparameters = self.beta_generate_hyperparameters(
            hyperparams_ranges
        )

        onchain_features = self.generate_onchain_features()

        metrics = self.create_onchain_catboost_model(
                onchain_features,
                max_trades,
                off_days,
                side,
                cutoff_point,
                "metrics",
                **hyperparameters,
            )

        self.logger.info("Total time: %s", time.perf_counter() - start)
        return metrics
