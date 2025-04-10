import os
import json
from ast import literal_eval

import pytz
from datetime import datetime

import pandas as pd
import klib

from machine_learning.model_builder import model_creation
from machine_learning.ml_utils import DataHandler, get_recommendation


class ModelPipeline:
    def __init__(self, table_name, database_url):
        btc = pd.read_sql(table_name, con=database_url, index_col="date")
        target_length = 7

        self.model_df = DataHandler(btc).calculate_targets(target_length)
        self.model_df = klib.convert_datatypes(self.model_df)
        self.model_df["Target_bin"] = self.model_df["Target_bin"].replace(
            {0: -1}
        )

    def create_model(self, configs_dataset, dataframe):
        """
        Create a machine learning model based on the provided 
        configuration parameters.

        Parameters
        ----------
        configs_dataset : pandas.Series
            Series containing configuration parameters including 
            hyperparameters,
            feature parameters, test_index, train_in_middle, side, 
            max_trades, and off_days.
        dataframe : pandas.DataFrame
            The input dataframe containing the features and target 
            variables.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing model results including predictions and
            performance metrics.
        """
        hyperparams = literal_eval(configs_dataset.loc["hyperparameters"])[0]
        hyperparams["iterations"] = 1000

        feat_params = literal_eval(configs_dataset.loc["feat_parameters"])[0]
        test_index = int(configs_dataset.loc["test_index"])
        train_in_mid = configs_dataset.loc["train_in_middle"]
        side = int(configs_dataset.loc["side"])
        max_trades = int(configs_dataset.loc["max_trades"])
        off_days = int(configs_dataset.loc["off_days"])

        model_results, _, _, _ = model_creation(
            feat_params,
            hyperparams,
            test_index,
            dataframe,
            dev=False,
            train_in_middle=train_in_mid,
            cutoff_point=5,
            side=side,
            max_trades=max_trades,
            off_days=off_days,
        )

        return model_results
