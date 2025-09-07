import time

import pandas as pd

from machine_learning.model_creator import ModelCreator
from machine_learning.model_metrics import ModelMetrics
from machine_learning.ml_utils import calculate_returns
from machine_learning.adjust_predicts import adjust_predict_one_side


class MLStacker:
    def __init__(
        self,
        dataset,
        onchain_data,
        test_index,
        train_in_middle,
        max_trades,
        off_days,
        side,
        cutoff_point,
        best_model_dict: dict,
        verbose: bool = False,
    ):
        """
        Initialize the MLStacker class.

        Parameters
        ----------
        dataset : pd.DataFrame
            The main dataset containing features and target columns.
        onchain_data : pd.DataFrame
            Additional on-chain data to be used for modeling.
        test_index : int or list
            Index or indices specifying the test set split.
        train_in_middle : bool
            Whether to train the model using a middle split of the
            dataset.
        max_trades : int
            Maximum number of trades allowed in the strategy.
        off_days : int
            Number of days to remain inactive after a trade.
        side : str
            Trading side, e.g., 'buy' or 'sell'.
        cutoff_point : float
            Cutoff threshold for predictions or filtering.
        best_model_dict : dict
            Dictionary containing the best model parameters and
            configurations.
        verbose : bool, optional
            If True, enables verbose output during model training and
            evaluation
            (default: False).

        Attributes
        ----------
        dataset : pd.DataFrame
            Stores the input dataset.
        onchain_data : pd.DataFrame
            Stores the on-chain data.
        test_index : int or list
            Stores the test index.
        train_in_middle : bool
            Stores the train-in-middle flag.
        target_series : pd.Series
            Target values from the dataset.
        y_true : pd.Series
            Binary target values from the dataset.
        adj_targets : np.ndarray
            Adjusted target values based on trading constraints.
        max_trades : int
            Stores the maximum trades.
        off_days : int
            Stores the off days.
        side : str
            Stores the trading side.
        cutoff_point : float
            Stores the cutoff point.
        best_model_dict : dict
            Stores the best model configuration.
        verbose : bool
            Stores the verbosity flag.
        model_creator : ModelCreator
            Instance for creating models.
        model_metrics : None or dict
            Stores model metrics after evaluation.
        empty_dict : dict
            Template dictionary for storing model results and metrics.
        """
        self.dataset = dataset
        self.onchain_data = onchain_data
        self.test_index = test_index
        self.train_in_middle = train_in_middle
        self.target_series = self.dataset["Target"]
        self.y_true = self.dataset['Target_bin']
        self.adj_targets = adjust_predict_one_side(
            self.y_true,
            max_trades,
            off_days,
            side,
        )
        self.max_trades = max_trades
        self.off_days = off_days
        self.side = side
        self.cutoff_point = cutoff_point
        self.best_model_dict = best_model_dict

        self.verbose = verbose
        self.model_creator = ModelCreator(
            dataset, onchain_data, test_index, verbose, train_in_middle
        )

        self.model_metrics = None

        self.empty_dict = {
            "onchain_model_1_hyperparameters": None,
            "onchain_model_2_hyperparameters": None,
            "onchain_model_1_features": None,
            "onchain_model_2_features": None,
            "metrics_results": None,
            "model_dates_interval": None,
            "linear_accumulated_return_test": None,
            "linear_accumulated_return_val": None,
            "exponential_accumulated_return_test": None,
            "exponential_accumulated_return_val": None,
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

    def run_model_onchain(self):
        hyperparams = self.model_creator.beta_generate_hyperparameters({})
        params = self.model_creator.generate_onchain_features()
        return self.train_onchain_model(hyperparams, params)

    def train_onchain_model(
        self,
        hyperparams,
        params,
    ):
        """
        Train a model using on-chain feature generation and return the
        trained model along with corresponding feature/label datasets
        and metadata.

        Parameters
        ----------
        hyperparams : dict
            Hyperparameters forwarded to self.model_creator.get_model()
            used to configure the model training/fitting procedure
            (e.g., model-specific settings, regularization, etc.).
        params : dict
            Parameters passed to
            self.model_creator.create_multiple_onchain_features() that
            control how on-chain features are generated or updated
            (e.g., window sizes, aggregation rules).

        Returns
        -------
        dict
            A dictionary containing the following keys:
            - "model": object
                The trained model returned as the first element of
                self.model_creator.get_model(...).
            - "all_x": pandas.DataFrame
                DataFrame of feature columns (columns whose names end
                with "_feat") selected from
                self.model_creator.features_dataset and filtered to the
                indexes originally stored in
                self.best_model_dict['all_x'].
            - "all_y": array-like or pandas.Series
                Target labels returned as the third element of
                self.model_creator.get_model(...).
            - "index_splits": object
                Index split information (e.g., train/validation/test
                index sets) returned as the fourth element of
                self.model_creator.get_model(...).
            - "hyperparams": dict
                The same hyperparams argument passed through for
                traceability.
            - "feat_params": dict
                The same params argument passed through
                (feature-generation parameters).

        Raises
        ------
        KeyError
            If 'all_x' is not present in self.best_model_dict.
        AttributeError
            If self.model_creator does not implement the required
            methods or attributes (create_multiple_onchain_features,
            get_model, or features_dataset).
        IndexError
            If the stored indexes in self.best_model_dict['all_x'].index
            are not present in self.model_creator.features_dataset.
        """
        indexes = self.best_model_dict['all_x'].index

        self.model_creator.create_multiple_onchain_features(params)

        model_dict = self.model_creator.get_model(
            max_trades=self.max_trades,
            off_days=self.off_days,
            side=self.side,
            cutoff_point=self.cutoff_point,
            **hyperparams
        )

        feat_columns = [col for col in self.model_creator.features_dataset.columns if col.endswith('_feat')]

        return {
            "model": model_dict[0],
            "all_x": self.model_creator.features_dataset.loc[indexes, feat_columns],
            "all_y": model_dict[2],
            "index_splits": model_dict[3],
            "hyperparams": hyperparams,
            "feat_params": params,
        }

    def create_weighted_models(self):
        """
        Creates and returns two weighted on-chain models.

        This method runs the on-chain model generation process twice and
        returns the resulting models. The models can be used for further
        stacking or ensemble learning.

        Returns
        -------
        tuple
            A tuple containing two on-chain models generated by
            `run_model_onchain`.
        """
        onchain_model_1 = self.run_model_onchain()
        onchain_model_2 = self.run_model_onchain()
        return onchain_model_1, onchain_model_2

    def get_positive_proba(self, model_dict):
        """
        Extract positive class probabilities from a trained model.

        Parameters
        ----------
        model_dict : dict
            Dictionary containing the trained model and feature data.
            Expected keys:
                - 'model': trained classifier with predict_proba method
                - 'all_x': pandas DataFrame containing input features

        Returns
        -------
        pandas.Series
            Series containing probabilities for the positive class
            (class 1), indexed by the same index as the input features.
        """
        proba = model_dict['model'].predict_proba(model_dict['all_x'])
        return pd.Series(proba[:, 1], index=model_dict['all_x'].index)

