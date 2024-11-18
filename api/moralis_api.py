import logging

import pandas as pd
import numpy as np

from moralis import evm_api


class MoralisAPI:
    """
    A class to interact with the Moralis API for retrieving transaction
    data.

    Parameters
    ----------
    verbose : bool
        If True, sets the logger to INFO level.
    api_key : str
        The API key for the Moralis API.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for logging information.
    api_key : str
        The API key for the Moralis API.

    Methods
    -------
    process_transaction_data(data)
        Processes transaction data for a given transaction.
    get_swaps(wallet)
        Retrieves all swaps for a given wallet address.
    get_swaps_data(swaps)
        Retrieves all swaps data for a given wallet address.
    get_account_swaps(wallet)
        Retrieves all swaps for a given wallet address.
    """

    def __init__(self, verbose: bool, api_key: str):
        """
        Initialize the MoralisAPI object.

        Parameters
        ----------
        verbose : bool
            If True, sets the logger to INFO level.
        api_key : str
            The API key for the Moralis API.
        """
        self.api_key = api_key

        self.logger = logging.getLogger("moralis_API")
        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s: %(message)s", datefmt="%H:%M:%S"
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.addHandler(handler)
        self.logger.propagate = False

        if verbose:
            self.logger.setLevel(logging.INFO)

    def process_transaction_data(self, data: list) -> list:
        """
        Processes transaction data for a given transaction.

        Parameters
        ----------
        data : list
            The transaction data to process.

        Returns
        -------
        list
            The processed transaction data.

        Raises
        ------
        ValueError
            If the data has less than 2 elements.
        """
        if len(data) == 2:
            return data

        if len(data) > 2:
            df = pd.DataFrame(data)
            default_columns = df.columns.tolist()
            value_columns = [
                "value",
                "value_formatted",
            ]

            df[value_columns] = df[value_columns].astype(float)
            df = df.groupby("direction").agg(
                {
                    col: "sum" if col in value_columns else "first"
                    for col in default_columns
                }
            )

            ordened_df = df.loc[["send", "receive"]][default_columns]

            return [ordened_df.iloc[x].to_dict() for x in range(df.shape[0])]

        raise ValueError("data has less than 2 elements")

    def get_transactions(self, wallet: str) -> list:
        """
        Retrieves all swaps for a given wallet address.

        Parameters
        ----------
        wallet : str
            The wallet address to retrieve swaps for.
        """
        self.logger.info("Retrieving transactions for wallet: %s", wallet)

        params = {
            "chain": "polygon",
            "order": "DESC",
            "address": wallet,
        }

        txn_infos = evm_api.wallets.get_wallet_history(
            api_key=self.api_key,
            params=params,
        )["result"]

        swaps = []

        excluded_categories = [
            "contract interaction",
            "token receive",
            "airdrop",
            "receive",
            "approve",
        ]

        for txn in txn_infos:
            is_not_spam = not txn["possible_spam"]
            in_excluded_categories = txn["category"] in excluded_categories

            if is_not_spam and not in_excluded_categories:
                swaps.append(txn)

        self.logger.info("Retrieved %d transactions", len(swaps))

        return swaps

