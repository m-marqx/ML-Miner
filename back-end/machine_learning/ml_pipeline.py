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
