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
            dataframe[usd_columns].sum(axis=1).fillna(0) +
            dataframe[token].fillna(0) * dataframe["usdPrice"]
        )

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

    def update_wallet_usd(
        self,
        wallet_data: pd.DataFrame | None = None,
        update: bool = True,
    ) -> pd.DataFrame:
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

    def update_btc_data(self, update: bool = True) -> pd.DataFrame:
        """
        Update BTC price data.

        Fetches new BTC/USDT price data from Binance using CCXT API,
        starting from the last recorded timestamp in the database.

        Parameters
        ----------
        update : bool, default True
            If True, saves processed data to database. If False, returns
            DataFrame without database update.

        Returns
        -------
        pd.DataFrame
            DataFrame containing BTC OHLCV data with date index.
            Columns include open, high, low, close, and volume.

        Notes
        -----
        Uses 1-day timeframe for BTC data. Excludes the last 2 rows from
        existing data to avoid incomplete candles.
        """
        logger.info("Starting BTC data update...")

        try:
            database_df = pd.read_sql("btc", con=self.database_url, index_col="date").iloc[:-2]
        except:
            first_time = pd.to_datetime("2012-01-02 00:00:00").timestamp() * 1000
            ccxt_api = CcxtAPI(
                "BTC/USD",
                "1d",
                ccxt.bitstamp(),
                since=int(first_time),
                verbose="Text",
            )

            database_df = ccxt_api.get_all_klines().to_OHLCV().data_frame.loc[:"2020"]
            database_df.index.name = "date"
            database_df["updatedAt"] = pd.Timestamp.now(tz='UTC')

        last_time = pd.to_datetime(database_df.index[-2]).timestamp() * 1000
        logger.info(f"Last recorded BTC timestamp: {last_time}")

        # Fetch new BTC data
        ccxt_api = CcxtAPI(
            "BTC/USDT",
            "1d",
            ccxt.binance(),
            since=int(last_time),
            verbose="Text",
        )

        new_data = ccxt_api.get_all_klines().to_OHLCV().data_frame
        new_data['updatedAt'] = pd.Timestamp.now(tz='UTC')
        btc = new_data.combine_first(database_df).drop_duplicates()

        if update:
            btc.to_sql(
                "btc",
                con=self.database_url,
                if_exists="replace",
                index=True,
                index_label="date",
            )
            logger.info("BTC data updated in database")
            return btc
        else:
            logger.info("Returning BTC DataFrame")
            return btc

    def update_model_recommendations(self, update: bool = True) -> pd.DataFrame:
        """
        Update model recommendations.

        Generates trading recommendations using the machine learning pipeline
        and processes them into structured format with position, side, and capital columns.

        Parameters
        ----------
        update : bool, default True
            If True, saves processed data to database. If False, returns
            DataFrame without database update.

        Returns
        -------
        pd.DataFrame
            DataFrame containing model recommendations with columns:
            - 'position': trading position recommendation
            - 'side': trading side and direction
            - 'capital': recommended capital allocation
            Index is date.

        Notes
        -----
        Recommendations are processed by splitting the raw recommendation text
        and extracting position, side, and capital information.
        """
        logger.info("Starting model recommendations update...")

        try:
            model_pipeline = ModelPipeline("btc", database_url=self.database_url)

            # Get recommendations with better error handling
            try:
                raw_recommendations = model_pipeline.get_model_recommendations(False)
                logger.info(f"Raw recommendations type: {type(raw_recommendations)}")
                logger.info(f"Raw recommendations shape/length: {getattr(raw_recommendations, 'shape', len(raw_recommendations) if hasattr(raw_recommendations, '__len__') else 'unknown')}")

                if raw_recommendations is None or (hasattr(raw_recommendations, 'empty') and raw_recommendations.empty):
                    logger.warning("No recommendations returned from model pipeline")
                    # Return empty DataFrame with expected structure
                    empty_df = pd.DataFrame(columns=['position', 'side', 'capital'])
                    if update:
                        empty_df.to_sql(
                            "model_recommendations",
                            con=self.database_url,
                            if_exists="replace",
                            index=True,
                            index_label="date",
                        )
                    return empty_df

            except Exception as e:
                logger.error(f"Error getting model recommendations: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                # Return empty DataFrame with expected structure
                empty_df = pd.DataFrame(columns=['position', 'side', 'capital'])
                if update:
                    empty_df.to_sql(
                        "model_recommendations",
                        con=self.database_url,
                        if_exists="replace",
                        index=True,
                        index_label="date",
                    )
                return empty_df

            clean_recomendations = (
                raw_recommendations
                .rename("Clean")
                .to_frame()
            )[::-1]

            if clean_recomendations.empty:
                logger.warning("Clean recommendations DataFrame is empty")
                empty_df = pd.DataFrame(columns=['position', 'side', 'capital'])
                if update:
                    empty_df.to_sql(
                        "model_recommendations",
                        con=self.database_url,
                        if_exists="replace",
                        index=True,
                        index_label="date",
                    )
                return empty_df

            logger.info(f"Processing {len(clean_recomendations)} recommendations")

            clean_recomendations_splitted = clean_recomendations["Clean"].str.split(" ", expand=True)
            clean_recomendation = clean_recomendations_splitted[0].rename("position").to_frame()
            clean_recomendation["side"] = (
                clean_recomendations_splitted[1].fillna("").astype(str)
                + " "
                + clean_recomendations_splitted[2].fillna("").astype(str).fillna("")
            )
            clean_recomendation["capital"] = clean_recomendations_splitted[4].fillna("")

            if update:
                clean_recomendation.to_sql(
                    "model_recommendations",
                    con=self.database_url,
                    if_exists="replace",
                    index=True,
                    index_label="date",
                )
                logger.info("Model recommendations updated in database")
                return clean_recomendation
            else:
                logger.info("Returning model recommendations DataFrame")
                return clean_recomendation

        except Exception as e:
            logger.error(f"Error in update_model_recommendations: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty DataFrame to prevent pipeline failure
            empty_df = pd.DataFrame(columns=['position', 'side', 'capital'])
            if update:
                try:
                    empty_df.to_sql(
                        "model_recommendations",
                        con=self.database_url,
                        if_exists="replace",
                        index=True,
                        index_label="date",
                    )
                except Exception as db_error:
                    logger.error(f"Failed to save empty recommendations to database: {str(db_error)}")
            return empty_df

