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

    def create_weighted_predictions(
        self,
        model_dict_1: dict,
        model_dict_2: dict,
        weights: None | list = None,
    ):
        """
        Create weighted ensemble predictions from multiple models.
        Combines predictions from the best model and two additional test
        models using specified weights to create a weighted ensemble
        prediction.

        Parameters
        ----------
        model_dict_1 : dict
            Dictionary containing the first test model and
            its predictions.
        model_dict_2 : dict
            Dictionary containing the second test model and
            its predictions.
        weights : list or None, optional
            List of three weights for [best_model, model_1, model_2].

            Default is [0.7, 0.15, 0.15].
        Returns
        -------
        pandas.DataFrame
            DataFrame containing individual weighted probabilities
            and their sum in 'summed_proba' column.
        """
        weights = weights or [0.7, 0.15, 0.15]

        best_model = (
            self.get_positive_proba(self.best_model_dict) * weights[0]
        ).rename("best_model")
        test_model_1 = (
            self.get_positive_proba(model_dict_1) * weights[1]
        ).rename("test_model_2")
        test_model_2 = (
            self.get_positive_proba(model_dict_2) * weights[2]
        ).rename("test_model_3")

        sum_probas = pd.concat(
            [best_model, test_model_1, test_model_2], axis=1
        )

        sum_probas["summed_proba"] = sum_probas.sum(axis=1)
        return sum_probas

    def mine_onchain_model(self, weights: None | list = None):
        """
        Mine and evaluate a weighted ensemble of on-chain models.

        This method creates two on-chain models, combines their
        predictions using specified weights, calculates trading returns,
        and evaluates performance metrics. If evaluation fails due to
        poor performance or errors, returns empty results.

        Parameters
        ----------
        weights : list or None, optional
            List of three weights for [best_model, model_1, model_2].
            Default is [0.7, 0.15, 0.15].

        Returns
        -------
        dict
            Dictionary containing model hyperparameters, feature
            parameters, performance metrics, accumulated returns,
            drawdowns, trading statistics, and evaluation metadata.
            Returns empty_dict if models fail performance thresholds
            or evaluation encounters errors.

        Raises
        ------
        Exception
            Re-raises any exception encountered during model evaluation
            with additional context about the failed models'
            hyperparameters and feature parameters.
        """
        start = time.perf_counter()
        weights = weights or [0.7, 0.15, 0.15]
        onchain_model_1, onchain_model_2 = self.create_weighted_models()

        sum_probas = self.create_weighted_predictions(
            onchain_model_1,
            onchain_model_2,
            weights=weights,
        )

        target_rollback = -self.off_days

        results = calculate_returns(
            target=self.dataset["Target"].iloc[:target_rollback],
            y_pred_probs=sum_probas["summed_proba"].iloc[:target_rollback],
            y_true=self.y_true.iloc[:target_rollback],
            max_trades=self.max_trades,
            off_days=self.off_days,
            side=self.side,
            cutoff_point=self.cutoff_point,
        )

        try:
            model_dates_interval = pd.Interval(
                results.index[0],
                results.index[-1],
                closed="both",
            )

            self.model_metrics = ModelMetrics(
                self.train_in_middle,
                self.best_model_dict["index_split"],
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
                self.y_true,
                7,
            )

            if min(metrics_results["precisions"]) < 0.52:
                return self.empty_dict

            total_operations, total_operations_pct = (
                self.model_metrics.calculate_total_operations(
                    test_buys=test_buys,
                    val_buys=val_buys,
                    max_trades=self.max_trades,
                    off_days=self.off_days,
                    side=self.side,
                )
            )

            drawdowns = self.model_metrics.calculate_drawdowns()

            return_ratios = self.model_metrics.calculate_result_ratios()

            support_diffs = self.model_metrics.calculate_result_support(
                self.adj_targets, self.side
            )

            # get the results from the bear market that started in 2022
            bearmarket_2022 = results.loc["2021-08-11":"2023-01-01"]

            r2_test, r2_val, ols_coef_test, ols_coef_val = (
                self.model_metrics.set_results_test(
                    bearmarket_2022
                ).calculate_ols_metrics()
            )

            return {
                "onchain_model_1_hyperparameters": onchain_model_1[
                    "hyperparams"
                ],
                "onchain_model_2_hyperparameters": onchain_model_2[
                    "hyperparams"
                ],
                "onchain_model_1_features": onchain_model_1["feat_params"],
                "onchain_model_2_features": onchain_model_2["feat_params"],
                "weights": weights,
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
                "expected_return_test": metrics_results[
                    "expected_return_test"
                ],
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
                "side": self.side,
                "max_trades": self.max_trades,
                "off_days": self.off_days,
            }

        except Exception as e:
            raise type(e)(
                f"Error creating model: {e}"
                + "\nFirst On-chain Model:"
                + f"\n{onchain_model_1['hyperparams']}"
                + f"\n{onchain_model_1['feat_params']}"
                + "\nSecond On-chain Model:"
                + f"\n{onchain_model_2['hyperparams']}"
                + f"\n{onchain_model_2['feat_params']}"
            ) from e

    def calculate_result_metrics(
        self,
        results,
    ):
        """
        Calculate comprehensive model performance metrics and statistics.

        This method computes various performance metrics including
        accumulated returns, drawdowns, precision scores, trading
        operations, and regression statistics for model evaluation.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame containing model results with datetime index
            and performance data for metric calculations.

        Returns
        -------
        dict
            Dictionary containing comprehensive model metrics:
            - metrics_results: list with calculated metric results
            - model_dates_interval: pd.Interval of result date range
            - linear/exponential_accumulated_return_test/val: float
                Accumulated returns for test and validation sets
            - drawdown_full/adj_test/val: float
                Maximum drawdown values for different configurations
            - expected_return_test/val: float
                Expected return values for test and validation
            - precisions_test/val: float
                Precision scores for test and validation sets
            - support_diff_test/val: float
                Support difference metrics
            - total_operations_test/val: int
                Total number of trading operations
            - total_operations_pct_test/val: float
                Percentage of total operations
            - r2_in_2023/val: float
                R-squared values for 2023 and validation periods
            - ols_coef_2022/val: float
                OLS coefficients for 2022 and validation periods
            - test_index: int/list
                Test index configuration
            - train_in_middle: bool
                Training configuration flag
            - return_ratios: dict
                Calculated return ratio metrics
            - side/max_trades/off_days: str/int
                Trading configuration parameters

            Returns empty_dict if accumulated returns <= 1 or
            minimum precision < 0.52.
        """
        model_dates_interval = pd.Interval(
            results.index[0],
            results.index[-1],
            closed='both',
        )

        self.model_metrics = ModelMetrics(
            self.train_in_middle,
            self.best_model_dict["index_split"],
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
            self.y_true,
            7,
        )

        if min(metrics_results["precisions"]) < 0.52:
            return self.empty_dict

        total_operations, total_operations_pct = (
            self.model_metrics.calculate_total_operations(
                test_buys=test_buys,
                val_buys=val_buys,
                max_trades=self.max_trades,
                off_days=self.off_days,
                side=self.side,
            )
        )

        drawdowns = self.model_metrics.calculate_drawdowns()

        return_ratios = self.model_metrics.calculate_result_ratios()

        support_diffs = (
            self.model_metrics
            .calculate_result_support(self.adj_targets, self.side)
        )

        # get the results from the bear market that started in 2022
        bearmarket_2022 = results.loc["2021-08-11":"2023-01-01"]

        r2_test, r2_val, ols_coef_test, ols_coef_val = (
            self.model_metrics.set_results_test(
                bearmarket_2022
            ).calculate_ols_metrics()
        )

        return {
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
            "return_ratios": return_ratios,
            "side": self.side,
            "max_trades": self.max_trades,
            "off_days": self.off_days,
        }
