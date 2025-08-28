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

