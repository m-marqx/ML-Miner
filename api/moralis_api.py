import time

import pandas as pd
import numpy as np

from moralis import evm_api
from custom_exceptions.invalid_arguments import InvalidArgumentError
from utils.log_handler import create_logger

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

    def __init__(self, verbose: bool, api_key: str, chain: str = "polygon"):
        """
        Initialize the MoralisAPI object.

        Parameters
        ----------
        api_key : str
            The API key for the Moralis API.
        chain : str
            The blockchain to retrieve data for.
        logger : logging.Logger
            Logger instance for logging information.
        """
        self.api_key = api_key
        self.chain = chain
        self.logger = create_logger("moralis_api", verbose)

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

    def get_transactions(
        self,
        wallet: str,
        excluded_categories: list | None = None,
        **kwargs,
    ) -> list:
        """
        Retrieves transaction history for the specified wallet address 
        while filtering out transactions that are marked as spam or 
        belong to any of the excluded categories.

        Parameters
        ----------
        wallet : str
            The wallet address for which to retrieve the transaction 
            history.
        excluded_categories : list or None, optional
            A list of transaction categories to exclude. If None, a 
            default list of categories including "contract interaction",
            "token receive", "airdrop", "receive", "approve", and "send"
            will be used.
        **kwargs : dict
            Additional keyword arguments to filter transactions, such 
            as:
                - **from_block**: int
                    The minimum block number to start retrieving 
                    transactions.
                - **to_block**: int
                    The maximum block number to stop retrieving 
                    transactions.
                - **from_date**: str
                    The start date 
                    (in seconds or a momentjs-compatible datestring).
                - **to_date**: str
                    The end date 
                    (in seconds or a momentjs-compatible datestring).
                - **include_internal_transactions**: bool
                    Whether to include internal transactions in the 
                    results.
                - **nft_metadata**: bool
                    Whether to include NFT metadata in the results.
                - **cursor**: str
                    A pagination cursor returned from previous 
                    responses.
                - **order**: str
                    The order of transactions, either "ASC" for 
                    ascending or "DESC" for descending.
                - **limit**: int
                    The maximum number of transactions to retrieve.

        Returns
        -------
        list
            A list of transaction dictionaries that have been filtered to exclude
            spam and the specified categories.

        Side Effects
        ------------
        Logs the start and completion of the transaction retrieval process.
        """
        self.logger.info("Retrieving transactions for wallet: %s", wallet)

        params = {**kwargs}
        params['chain'] = kwargs.get('chain', self.chain)
        params['address'] = kwargs.get('address', wallet)
        params['order'] = kwargs.get('order', 'DESC')

        txn_infos = evm_api.wallets.get_wallet_history(
            api_key=self.api_key,
            params=params,
        )["result"]

        transactions = []

        if excluded_categories is None:
            excluded_categories = [
                "contract interaction",
                "token receive",
                "airdrop",
                "receive",
                "approve",
                "send",
            ]

        for txn in txn_infos:
            is_not_spam = not txn["possible_spam"]
            in_excluded_categories = txn["category"] in excluded_categories

            if is_not_spam and not in_excluded_categories:
                transactions.append(txn)

        self.logger.info("Retrieved %d transactions", len(transactions))

        return transactions

    def get_swaps(self, swaps: list, add_summary: bool = False) -> list:
        """
        Retrieves all swaps data for a given wallet address.

        Parameters
        ----------
        swaps : list
            The swaps to retrieve data for.

        Returns
        -------
        list
            A list of dictionaries, each containing details of a swap
            transaction.
        """
        swaps_data = []

        infos_df = pd.DataFrame(swaps)
        infos_df["transaction_fee"] = infos_df["transaction_fee"].astype(float)
        infos_df["summary"] = infos_df["summary"]

        for idx, x in enumerate(swaps):
            try:
                swap = self.process_transaction_data(x["erc20_transfers"])

            except ValueError as exc:
                erc20_transfer_direction = x["erc20_transfers"][0]["direction"]

                if erc20_transfer_direction == "send":
                    x = x["erc20_transfers"] + x["native_transfers"]

                elif erc20_transfer_direction == "receive":
                    x = x["native_transfers"] + x["erc20_transfers"]

                else:
                    raise ValueError("unknown direction") from exc

                swap = self.process_transaction_data(x)

            swap.extend([{"txn_fee": infos_df.loc[idx, "transaction_fee"]}])

            if add_summary:
                swap.extend([{"summary": infos_df.loc[idx, "summary"]}])

            swaps_data.append(swap)

        return swaps_data

    def get_account_swaps(self, wallet: str, coin_name: bool = False, add_summary: bool = False) -> pd.DataFrame:
        """
        Retrieves all swaps for a given wallet address.

        Parameters
        ----------
        wallet : str
            The wallet address to retrieve swaps for.
        coin_name : bool
            Whether to include the names of the coins being swapped.


        Returns
        -------
        pandas.DataFrame
            A DataFrame containing details of all swaps for the given
            wallet address.
        """
        swaps_list = self.get_transactions(wallet)
        swaps_data = self.get_swaps(swaps_list, add_summary)

        swap_columns = ["token_symbol", "value_formatted"]
        from_df = pd.DataFrame(pd.DataFrame(swaps_data)[0].tolist())[
            swap_columns
        ]
        from_df = from_df.rename(
            columns={
                "token_symbol": "from_coin_name",
                "value_formatted": "from",
            }
        )

        to_df = pd.DataFrame(pd.DataFrame(swaps_data)[1].tolist())[
            swap_columns
        ]
        to_df = to_df.rename(
            columns={"token_symbol": "to_coin_name", "value_formatted": "to"}
        )

        fee_df = pd.DataFrame(pd.DataFrame(swaps_data)[2].tolist())

        columns_name = [
            "from",
            "to",
            "USD Price",
            "from_coin_name",
            "to_coin_name",
            "txn_fee",
        ]

        data_dfs = [from_df, to_df, fee_df]

        if add_summary:
            columns_name.append("summary")
            summary_df = pd.DataFrame(pd.DataFrame(swaps_data)[3].tolist())
            data_dfs.append(summary_df)

        swaps_df = pd.concat(data_dfs, axis=1)

        swaps_df[["from", "to"]] = swaps_df[["from", "to"]].astype(float)

        swaps_df["USD Price"] = np.where(
            swaps_df["to_coin_name"].str.startswith("USD"),
            swaps_df["to"] / swaps_df["from"],
            swaps_df["from"] / swaps_df["to"],
        )

        swaps_df = swaps_df[columns_name]

        if not coin_name:
            coin_name_columns = ["from_coin_name", "to_coin_name"]
            swaps_df = swaps_df.drop(columns=coin_name_columns)

        return swaps_df

    def get_token_price(self, block_number):
        """
        Retrieves the token price at a specified block number using the 
        Moralis API.

        Parameters
        ----------
        block_number : int
            The block number at which to fetch the token price.

        Returns
        -------
        pandas.Series
            A Series containing the token price data as returned by the 
            Moralis API.
        """
        params = {
            "chain": self.chain,
            "to_block": block_number,
            "address": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
        }

        result = evm_api.token.get_token_price(
            api_key=self.api_key,
            params=params,
        )

        return pd.Series(result)