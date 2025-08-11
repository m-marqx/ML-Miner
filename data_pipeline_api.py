import os
import logging
import ccxt
import pandas as pd
import numpy as np
import time

from crypto_explorer import CcxtAPI, MoralisAPI, QuickNodeAPI
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

    def update_onchain_data(self, update: bool = True, batch_size: int = 10000) -> pd.DataFrame:
        """
        Update Bitcoin onchain blockchain data.

        Fetches new Bitcoin block statistics from QuickNode API starting from the last
        recorded block height and appends to the current session file. Creates a new file
        only when Docker container restarts (detected by absence of session marker).

        Parameters
        ----------
        update : bool, default True
            If True, saves processed data to parquet files. If False, returns
            DataFrame without file update.
        batch_size : int, default 10000
            Number of blocks to process in each batch. Smaller batches for frequent updates.

        Returns
        -------
        pd.DataFrame
            DataFrame containing block statistics data with columns including
            block height, fees, transaction data, and other blockchain metrics.
            Index is 'height' (block number).

        Notes
        -----
        New Bitcoin blocks are generated approximately every 10 minutes.
        This method appends to the same file during a Docker session to avoid
        creating too many small files. Uses sequential processing for Docker compatibility.
        """
        logger.info("Starting onchain data update...")

        try:
            # Initialize QuickNode API
            api_keys = [os.getenv(f"quicknode_endpoint_{x}") for x in range(1, 11)]
            api_keys = [key for key in api_keys if key is not None]

            if not api_keys:
                logger.error("No QuickNode API keys found in environment variables")
                return pd.DataFrame()

            quant_node_api = QuickNodeAPI(api_keys, 0)

            # Get current blockchain height
            try:
                highest_height = quant_node_api.get_blockchain_info()["blocks"]
                logger.info(f"Current blockchain height: {highest_height}")
            except Exception as e:
                logger.error(f"Error getting blockchain info: {str(e)}")
                return pd.DataFrame()

            # Set up file paths
            data_folder = "data/onchain/BTC/block_stats_fragments/incremental"
            session_marker_file = f"{data_folder}/.session_marker"

            if not os.path.exists(data_folder):
                os.makedirs(data_folder, exist_ok=True)
                logger.info("Created incremental folder")

            # Determine if this is a new Docker session
            is_new_session = not os.path.exists(session_marker_file)

            # Find existing data and determine last height and current file
            last_height = 0
            current_file_path = None

            try:
                # Get all incremental files
                incremental_files = [f for f in os.listdir(data_folder) if f.startswith("incremental_block_stats_") and f.endswith(".parquet")]

                if incremental_files:
                    # Sort files by number and get the latest
                    file_numbers = [int(f.split("_")[-1].split(".")[0]) for f in incremental_files]
                    latest_file_num = max(file_numbers)
                    latest_file = f"{data_folder}/incremental_block_stats_{latest_file_num}.parquet"

                    try:
                        latest_data = pd.read_parquet(latest_file)
                        if not latest_data.empty:
                            last_height = latest_data['height'].max() + 1
                            logger.info(f"Found existing data, last height: {last_height - 1}")

                            # Use the same file if it's the same Docker session
                            if is_new_session:
                                # New session: create new file
                                new_file_number = latest_file_num + 1
                                current_file_path = f"{data_folder}/incremental_block_stats_{new_file_number}.parquet"
                                logger.info(f"New Docker session detected, creating new file: {current_file_path}")
                            else:
                                # Same session: append to existing file
                                current_file_path = latest_file
                                logger.info(f"Same Docker session, appending to existing file: {current_file_path}")
                    except Exception as e:
                        logger.warning(f"Could not read latest file {latest_file}: {str(e)}")

                if current_file_path is None:
                    # No existing files found, create the first one
                    current_file_path = f"{data_folder}/incremental_block_stats_0.parquet"
                    logger.info(f"No existing files found, creating first file: {current_file_path}")

            except Exception as e:
                logger.warning(f"Error finding existing files: {str(e)}")
                current_file_path = f"{data_folder}/incremental_block_stats_0.parquet"

            # Calculate blocks to fetch (limit to batch_size for incremental updates)
            blocks_to_fetch = min(highest_height - last_height, batch_size)

            if blocks_to_fetch <= 0:
                logger.info("No new blocks to fetch")
                # Create/update session marker even if no new blocks
                if update:
                    try:
                        with open(session_marker_file, 'w') as f:
                            f.write(str(time.time()))
                    except Exception as e:
                        logger.warning(f"Could not create session marker: {str(e)}")
                return pd.DataFrame()

            logger.info(f"Fetching {blocks_to_fetch} blocks from height {last_height} to {last_height + blocks_to_fetch - 1}")

            # Fetch block data sequentially (Docker-friendly)
            start_time = time.perf_counter()

            try:
                batch_data = []
                for block_height in range(last_height, last_height + blocks_to_fetch):
                    try:
                        block_stats = quant_node_api.get_block_stats(block_height)
                        if block_stats:
                            batch_data.append(block_stats)

                        # Log progress every 10 blocks for user feedback
                        if (block_height - last_height + 1) % 10 == 0 or block_height == last_height + blocks_to_fetch - 1:
                            progress = block_height - last_height + 1
                            logger.info(f"Progress: {progress}/{blocks_to_fetch} blocks fetched ({progress/blocks_to_fetch*100:.1f}%)")

                    except Exception as e:
                        logger.warning(f"Failed to fetch block {block_height}: {str(e)}")
                        continue

                if not batch_data:
                    logger.warning("No block data received")
                    return pd.DataFrame()

                # Convert to DataFrame
                new_block_df = pd.DataFrame(batch_data)

                # Convert time to datetime
                if 'time' in new_block_df.columns:
                    new_block_df['time'] = pd.to_datetime(new_block_df['time'], unit='s')

                elapsed_time = time.perf_counter() - start_time
                blocks_per_second = len(batch_data) / elapsed_time if elapsed_time > 0 else 0

                logger.info(f"Fetched {len(batch_data)} blocks in {elapsed_time:.2f} seconds ({blocks_per_second:.2f} blocks/sec)")
                logger.info(f"Block range: {new_block_df['height'].min()} - {new_block_df['height'].max()}")

                # Save to file if update is True
                if update:
                    try:
                        # Check if we need to append or create new file
                        if os.path.exists(current_file_path) and not is_new_session:
                            # Append to existing file
                            existing_data = pd.read_parquet(current_file_path)
                            combined_data = pd.concat([existing_data, new_block_df], ignore_index=True)
                            # Remove duplicates based on height (just in case)
                            combined_data = combined_data.drop_duplicates(subset=['height'], keep='last')
                            combined_data = combined_data.sort_values('height')
                            combined_data.to_parquet(current_file_path)
                            logger.info(f"Appended {len(new_block_df)} blocks to existing file: {current_file_path}")
                        else:
                            # Create new file or overwrite
                            new_block_df.to_parquet(current_file_path)
                            logger.info(f"Saved {len(new_block_df)} blocks to new file: {current_file_path}")
                        
                        # Create/update session marker
                        with open(session_marker_file, 'w') as f:
                            f.write(str(time.time()))
                            
                    except Exception as e:
                        logger.error(f"Error saving onchain data: {str(e)}")

                return new_block_df.set_index('height') if 'height' in new_block_df.columns else new_block_df

            except Exception as e:
                logger.error(f"Error fetching block data: {str(e)}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error in update_onchain_data: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def run_full_pipeline(self, update: bool = True) -> dict:
        """
        Run the complete data pipeline.

        Executes all pipeline steps in sequence: wallet transactions update,
        wallet USD data update, BTC data update, model recommendations update,
        and onchain blockchain data update.

        Parameters
        ----------
        update : bool, default True
            If True, saves all processed data to database/files. If False, returns
            all DataFrames without database/file updates.

        Returns
        -------
        dict
            Dictionary containing all processed DataFrames and execution status.
            Keys include:
            - 'wallet_balances': wallet balance DataFrame
            - 'wallet_usd': wallet USD DataFrame
            - 'btc_data': BTC price DataFrame
            - 'model_recommendations': recommendations DataFrame
            - 'onchain_data': Bitcoin blockchain statistics DataFrame
            - 'status': execution status ('success', 'partial_success', or 'error')
            - 'errors': list of error messages if any step fails

        Notes
        -----
        If any step fails, the pipeline continues with remaining steps and returns
        partial_success status. Only returns error status if most steps fail.
        """
        logger.info("Starting full pipeline execution...")

        results = {}
        errors = []

        # Step 1: Update wallet transactions
        try:
            logger.info("Step 1: Updating wallet transactions...")
            wallet_data = self.get_wallet_transactions(update=update)
            results["wallet_balances"] = wallet_data
            logger.info("✓ Step 1 completed successfully")
        except Exception as e:
            error_msg = f"Step 1 (wallet transactions) failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            results["wallet_balances"] = pd.DataFrame()

        # Step 2: Update wallet USD data
        try:
            logger.info("Step 2: Updating wallet USD data...")
            wallet_usd = self.update_wallet_usd(update=update)
            results["wallet_usd"] = wallet_usd
            logger.info("✓ Step 2 completed successfully")
        except Exception as e:
            error_msg = f"Step 2 (wallet USD) failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            results["wallet_usd"] = pd.DataFrame()

        # Step 3: Update BTC data
        try:
            logger.info("Step 3: Updating BTC data...")
            btc_data = self.update_btc_data(update=update)
            results["btc_data"] = btc_data
            logger.info("✓ Step 3 completed successfully")
        except Exception as e:
            error_msg = f"Step 3 (BTC data) failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            results["btc_data"] = pd.DataFrame()

        # Step 4: Update model recommendations
        try:
            logger.info("Step 4: Updating model recommendations...")
            recommendations = self.update_model_recommendations(update=update)
            results["model_recommendations"] = recommendations
            logger.info("✓ Step 4 completed successfully")
        except Exception as e:
            error_msg = f"Step 4 (model recommendations) failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            results["model_recommendations"] = pd.DataFrame()

        # Step 5: Update onchain data
        try:
            logger.info("Step 5: Updating onchain blockchain data...")
            onchain_data = self.update_onchain_data(update=update)
            results["onchain_data"] = onchain_data
            logger.info("✓ Step 5 completed successfully")
        except Exception as e:
            error_msg = f"Step 5 (onchain data) failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            results["onchain_data"] = pd.DataFrame()

        # Determine overall status
        if not errors:
            logger.info("Full pipeline execution completed successfully")
            results["status"] = "success"
        else:
            logger.warning(f"Pipeline completed with {len(errors)} errors")
            results["status"] = "partial_success" if len(errors) < 5 else "error"
            results["errors"] = errors

        return results

    def _process_wallet_data(self, token_balances: list, response: list) -> pd.DataFrame:
        """
        Process wallet data from token balances and transaction responses.

        Combines token balance data with transaction responses, calculates
        USD values, processes transfers, and creates aggregated metrics.

        Parameters
        ----------
        token_balances : list of pd.DataFrame
            List of DataFrames containing token balance data for each block.
        response : list of dict
            List of transaction response dictionaries from Moralis API.

        Returns
        -------
        pd.DataFrame
            Processed DataFrame with wallet data including:
            - Token balances and prices
            - Total USD values
            - Aggregated transaction values
            - Block timestamps
            Index is 'height' (block number).

        Notes
        -----
        This is a private method that handles complex data processing including:
        - Merging token balances with transaction data
        - Calculating USD values for different tokens
        - Processing transfer directions (receive/send)
        - Creating cumulative aggregated values
        """
        if not token_balances or len(token_balances) == 0:
            # Return empty DataFrame with expected structure if no token balances
            logger.warning("No token balances provided to process")
            return pd.DataFrame()

        try:
            response_data = pd.concat(token_balances)
        except Exception as e:
            logger.error(f"Error concatenating token balances: {str(e)}")
            return pd.DataFrame()

        response_data = response_data.reset_index()
        response_data = response_data.rename(columns={"index": "block_number"})

        new_data = response_data.copy()

        usd_columns = [col for col in new_data.columns if any(sub in col for sub in ["USD", "DAI"])]
        new_data["total_usd"] = (new_data[usd_columns].sum(axis=1) + new_data["WBTC"] * new_data["usdPrice"]).fillna(0)

        relevant_columns = response_data.columns.tolist() + ["total_usd"]

        # Only merge with response data if it's not empty
        if response and len(response) > 0:
            response_df = pd.DataFrame(response)

            # response_blocks = response_df['block_number'].tolist()
            # new_data = new_data.query("block_number == @response_blocks")

            new_data = new_data.merge(
                response_df,
                on="block_number",
                how="left",
                suffixes=("", "_response"),
            )

            not_swap_df = new_data.query("category != 'token swap'")

            if not_swap_df.empty:
                logger.info("No non-swap data found in the new_data.")

            extra_columns = ["airdrop", "receive"]
            not_swap_df = not_swap_df.set_index("block_number")

            transfer_data = not_swap_df.query("category not in @extra_columns")

            if transfer_data.empty:
                logger.info("No transfer data found in the not_swap_df.")
            else:
                transfer_data = pd.DataFrame(
                    pd.DataFrame(transfer_data["erc20_transfers"].tolist())[0].tolist(),
                    index=transfer_data.index,
                )

                usd_symbols = [
                    col
                    for col in transfer_data["token_symbol"].unique().tolist()
                    if any(sub in col for sub in ["USD", "DAI"])
                ]

                transfer_data["value_formatted"] = transfer_data[
                    "value_formatted"
                ].astype("float64")

                transfer_data["value_formatted"] = np.where(
                    transfer_data["direction"] == "receive",
                    -transfer_data["value_formatted"],
                    transfer_data["value_formatted"],
                )

                not_swap_df["value_formatted"] = np.nan

                transfer_data_agg = transfer_data.groupby(transfer_data.index).sum()

                not_swap_df.loc[transfer_data_agg.index, "value_formatted"] = transfer_data_agg

                not_swap_df = not_swap_df[not_swap_df["value_formatted"].isna() | ~not_swap_df.duplicated(subset="value_formatted")]

                not_swap_df = not_swap_df.sort_values(
                    ascending=True, by="blockTimestamp"
                )
                not_swap_df["value_aggregated"] = not_swap_df["value_formatted"].fillna(0).cumsum()

                not_swap_df["formatted_total_usd"] = not_swap_df["total_usd"] + not_swap_df["value_aggregated"]
                new_data = new_data.merge(
                    not_swap_df["value_aggregated"],
                    on="block_number",
                    how="left",
                )

                new_data["value_aggregated"] = new_data["value_aggregated"].ffill()
                new_data["formatted_total_usd"] = new_data["total_usd"] + new_data["value_aggregated"]
                relevant_columns = response_data.columns.tolist() + ["total_usd", "value_aggregated"]
        else:
            # If no response data, set default values
            logger.info("No response data provided, setting default values")
            new_data["value_aggregated"] = np.nan
            new_data["formatted_total_usd"] = new_data["total_usd"]
            relevant_columns = response_data.columns.tolist() + ["total_usd", "value_aggregated"]

        new_data = new_data[relevant_columns]
        new_data = new_data.rename(columns={"block_number": "height"})
        new_data = new_data.set_index("height")

        return new_data

