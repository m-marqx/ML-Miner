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

    def prepare_model(self):
        """
        Prepare and create a machine learning model using configurations
        from environment variables. This method retrieves model 
        configurations, features, and hyperparameters from environment
        variables, combines them into a unified configuration series,
        and passes them to the model creation function.

        Returns
        -------
        object
            The machine learning model created by the `create_model` 
            method using the specified configurations and the model
            dataframe.

        Notes
        -----
        The method relies on environment variables with specific keys:
        - "33139_configs": Contains general model configuration
        - "33139_features": Contains feature parameters
        - "33139_hyperparams": Contains model hyperparameters
        """
        ml_configs = literal_eval(json.loads(os.getenv("33139_configs")))
        ml_features = {
            "feat_parameters": json.loads(os.getenv("33139_features"))
        }
        ml_hyperparams = {
            "hyperparameters": json.loads(os.getenv("33139_hyperparams"))
        }

        configs_series = pd.Series(
            {**ml_features, **ml_hyperparams, **ml_configs}
        )

        return self.create_model(configs_series, self.model_df)

    def get_model_recommendations(self, span_tag: bool = True):
        """
        Generate and format model recommendations with appropriate time
        indexing. This method prepares a model, extracts prediction 
        probabilities and positions, and formats the recommendations 
        with proper timezone handling.  The last index is set to the 
        current time, and may be highlighted in red if it  doesn't 
        match the expected time format (20:59:59).

        Parameters
        ----------
        span_tag : bool, default=True
            Whether to add HTML span tags to the recommendation text 
            for styling.

        Returns
        -------
        pandas.Series
            A series containing model recommendations indexed by 
            formatted datetime strings in the 'America/Sao_Paulo' 
            timezone. The index format is 'YYYY-MM-DD HH:MM:SS'.

        Notes
        -----
        - The returned recommendations are indexed with timestamps 
        ending at 23:59:59 UTC and then converted to 'America/Sao_Paulo'
        timezone.

        - If the last index doesn't have time 20:59:59, it will be 
        highlighted in red using HTML span tags regardless of the 
        `span_tag` parameter.
        """
        model = self.prepare_model()
        recommendation_ml = model[["y_pred_probs", "Predict", "Position"]]

        recommendations = get_recommendation(
            recommendation_ml["Position"].loc["15-09-2024":],
            add_span_tag=span_tag,
        ).rename(f"model_33139")

        recommendations.index = (
            recommendations.index.tz_localize("UTC")
            + pd.Timedelta(hours=23, minutes=59, seconds=59)
        ).strftime("%Y-%m-%d %H:%M:%S")

        last_index = pd.Timestamp(datetime.now(pytz.timezone("UTC"))).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        recommendations.index = (
            pd.DatetimeIndex(
                recommendations.index[:-1].tolist() + [last_index]
            )
            .tz_localize("UTC")
            .tz_convert("America/Sao_Paulo")
        )

        last_index_hour = recommendations.index[-1].hour
        last_index_minute = recommendations.index[-1].minute
        last_index_second = recommendations.index[-1].second

        recommendations.index = recommendations.index.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        if (
            last_index_hour != 20
            and last_index_minute != 59
            and last_index_second != 59
        ):
            span_tag = "<span style='color: red'>"
            close_span_tag = "</span>"
            last_index = span_tag + recommendations.index[-1] + close_span_tag
            recommendations.index = recommendations.index[:-1].tolist() + [
                last_index
            ]

        return recommendations
