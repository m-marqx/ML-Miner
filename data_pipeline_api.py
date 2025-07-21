import os
import logging
import ccxt
import pandas as pd
import numpy as np
from crypto_explorer import CcxtAPI, MoralisAPI
from machine_learning.ml_pipeline import ModelPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipelineAPI:
    """
    API class for data pipeline operations with conditional database updates.

    This class provides methods to fetch, process, and update cryptocurrency
    wallet data, BTC price data, and machine learning model recommendations.
    All methods support conditional database updates based on the `update` parameter.

    Attributes
    ----------
    wallet : str
        Polygon wallet address from environment variable.
    api_key : str
        Moralis API key from environment variable.
    database_url : str
        Database URL from environment variable.
    """

    def __init__(self):
        self.wallet = os.getenv("polygon_wallet")
        self.api_key = os.getenv("moralis_api_key")
        self.database_url = os.getenv("DATABASE_URL")

        if not all([self.wallet, self.api_key, self.database_url]):
            raise ValueError("Missing required environment variables: polygon_wallet, moralis_api_key, DATABASE_URL")

    def process_total_usd(self, dataframe: pd.DataFrame, token: str = "WBTC"):
        """
        Process total USD value in the DataFrame.
        This function calculates the total USD value by summing relevant
        columns and multiplying WBTC by its USD price.

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame containing wallet balances with columns for various
            tokens and their USD prices.

        Returns
        -------
        pd.DataFrame
            DataFrame with an additional column 'total_usd' containing the
            total USD value.
        """
        usd_columns = [
            col
            for col in dataframe.columns
            if any(sub in col for sub in ["USD", "DAI"])
        ]

        return (
            dataframe[usd_columns].sum(axis=1)
            + dataframe[token] * dataframe["usdPrice"]
        ).fillna(0)

    def get_wallet_transactions(self, update: bool = True) -> pd.DataFrame:
        """
        Fetch new wallet transactions and update wallet balances.

        Retrieves new transactions from the Moralis API starting from the last
        recorded block height, processes token balances for each block, and
        optionally updates the database with the new data.

        Parameters
        ----------
        update : bool, default True
            If True, saves processed data to database. If False, returns
            DataFrame without database update.

        Returns
        -------
        pd.DataFrame
            DataFrame containing wallet balance data with columns including
            token balances, USD prices, block timestamps, and aggregated values.
            Index is 'height' (block number).

        Raises
        ------
        ValueError
            If required environment variables are missing.

        Notes
        -----
        Excludes transactions with categories "contract interaction" and "approve".
        Progress is logged during block processing.
        """
        logger.info("Starting wallet transactions update...")

        # Initialize Moralis API
        moralis_api = MoralisAPI(verbose=True, api_key=self.api_key)

        # Read existing wallet data
        try:
            wallet_data = pd.read_sql("wallet_balances", con=self.database_url)
            last_block = wallet_data['height'].iloc[-2]
        except:
            last_block = 61881005  # Default to a known block if no data exists
            wallet_data = pd.DataFrame(columns=["height"])

        excluded_categories = [
            "contract interaction",
            "approve",
        ]

        response = moralis_api.fetch_paginated_transactions(
            wallet_address=self.wallet,
            excluded_categories=excluded_categories,
            order='ASC',
            from_block=int(last_block),
        )

        # Fetch new wallet blocks
        try:
            wallet_blocks = moralis_api.get_wallet_blocks(
                wallet_address=self.wallet,
                from_block=int(last_block),
            )

            if wallet_blocks is None or len(wallet_blocks) == 0:
                logger.info("No new blocks found since last update")
                return wallet_data if "height" not in wallet_data.columns else wallet_data.set_index("height")

            wallet_blocks = (
                pd.Series(wallet_blocks)
                .sort_values(ascending=True)
                .tolist()
            )

        except Exception as e:
            logger.error(f"Error fetching wallet blocks: {str(e)}")
            return wallet_data if "height" not in wallet_data.columns else wallet_data.set_index("height")

        token_balances = []

        # Process each block
        for block in wallet_blocks:
            try:
                moralis_api.logger.info(f"Getting token balances for block {block}.")

                temp_df = moralis_api.get_wallet_token_balances(self.wallet, block).T
                token_price = moralis_api.fetch_token_price(block)

                temp_df['usdPrice'] = token_price['usdPrice']
                temp_df['blockTimestamp'] = pd.Timestamp(
                    int(token_price['blockTimestamp']),
                    unit='ms',
                )

                token_balances.append(temp_df)

                progress = len(token_balances) / len(wallet_blocks)
                progress_abs = f"{len(token_balances)} / {len(wallet_blocks)}"
                moralis_api.logger.info(f"Progress: {progress:.2%} - {progress_abs}")

            except Exception as e:
                moralis_api.logger.warning(f"Error processing block {block}: {str(e)}")
                continue

        if not token_balances:
            logger.info("No token balance data collected")
            return (
                wallet_data if "height" not in wallet_data.columns
                else wallet_data.set_index("height")
            )

        wallet_data = (
            wallet_data.set_index("height")
            if "height" in wallet_data.columns
            else wallet_data
        )

        new_wallet_data = self._process_wallet_data(token_balances, response)

        concated_wallet_data = pd.concat([wallet_data, new_wallet_data])

        concated_wallet_data['total_usd'] = self.process_total_usd(
            concated_wallet_data,
            token="WBTC",
        )

        concated_wallet_data['formatted_total_usd'] = (
            concated_wallet_data["total_usd"]
            + concated_wallet_data["value_aggregated"]
        )

        concated_wallet_data[['value_aggregated', 'formatted_total_usd']] = (
            concated_wallet_data[['value_aggregated', 'formatted_total_usd']]
            .ffill()
        )

        if update:
            concated_wallet_data.to_sql(
                "wallet_balances",
                con=self.database_url,
                if_exists="replace",
                index=True,
                index_label="height",
            )
            logger.info("Wallet balances updated in database")
            return concated_wallet_data
        else:
            logger.info("Returning wallet balances DataFrame")
            return concated_wallet_data

    def update_wallet_usd(self, wallet_data: pd.DataFrame | None = None, update: bool = True) -> pd.DataFrame:
        """
        Update wallet USD values table.

        Extracts USD value information from wallet balance data and creates
        a simplified table with timestamps and total USD values.

        Parameters
        ----------
        wallet_data : pd.DataFrame
            Wallet data DataFrame with 'blockTimestamp' and 'formatted_total_usd' columns.
            If None, reads from database 'wallet_balances' table.
            (default: None)
        update : bool
            If True, saves processed data to database. If False, returns
            DataFrame without database update.
            (default: True)

        Returns
        -------
        pd.DataFrame
            DataFrame containing wallet USD data with columns:
            - 'blockTimestamp': timestamp of the block
            - 'wallet_usd': total USD value of wallet
            Index is 'height' (block number).
        """
        logger.info("Starting wallet USD update...")

        if wallet_data is None:
            wallet_data = pd.read_sql("wallet_balances", con=self.database_url, index_col="height")

        wallet_usd = wallet_data[['blockTimestamp', 'formatted_total_usd']]
        wallet_usd = wallet_usd.rename(columns={"formatted_total_usd": "wallet_usd"})

        if update:
            wallet_usd.reset_index().to_sql(
                "wallet_usd",
                con=self.database_url,
                if_exists="replace",
                index=False,
            )
            logger.info("Wallet USD data updated in database")
            return wallet_usd
        else:
            logger.info("Returning wallet USD DataFrame")
            return wallet_usd

