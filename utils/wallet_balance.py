import ccxt
import pandas as pd
import numpy as np

from crypto_explorer import MoralisAPI, CcxtAPI


class WalletBalanceAnalyzer:
    """
    A class for analyzing wallet balances and tracking USD values over time.

    This class integrates with the Moralis API to fetch wallet transaction data
    and token balances, processes the data to calculate total USD values, and
    stores the results in a database.

    Parameters
    ----------
    wallet_address : str
        The wallet address to analyze.
    moralis_api_key : str
        The API key for accessing Moralis services.
    database_connection_url : str
        The database connection URL for storing results.
    enable_update : bool
        Flag indicating whether to update existing data.

    Attributes
    ----------
    wallet : str
        The wallet address being analyzed.
    api_key : str
        The Moralis API key.
    database_url : str
        The database connection URL.
    update : bool
        Update flag for data operations.
    moralis_api : MoralisAPI
        Instance of the Moralis API client.
    excluded_categories : list of str
        Transaction categories to exclude from analysis.
    extra_columns : list of str
        Additional columns for filtering operations.
    """

    def __init__(
        self,
        wallet_address: str,
        moralis_api_key: str,
        database_connection_url: str,
        enable_update: bool,
    ):
        self.wallet = wallet_address
        self.api_key = moralis_api_key
        self.database_url = database_connection_url
        self.update = enable_update
        self.moralis_api = None
        self.excluded_categories = ["contract interaction", "approve"]
        self.extra_columns = ["airdrop", "receive"]
        self.moralis_api = MoralisAPI(verbose=True, api_key=self.api_key)


    def set_wallet(self, wallet_address: str) -> "WalletBalanceAnalyzer":
        """
        Set the wallet address for analysis.

        Parameters
        ----------
        wallet_address : str
            The new wallet address.

        Returns
        -------
        WalletBalanceAnalyzer
            Self for method chaining.
        """
        self.wallet = wallet_address
        return self

