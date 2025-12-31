"""
BTC Single-Core Updater for GitHub Actions.

This script fetches Bitcoin blockchain data and OHLCV data sequentially,
designed to run in CI/CD environments without multicore parallelization.
"""

import os
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd
import numpy as np

from crypto_explorer import QuickNodeAPI, CcxtAPI


def get_date_suffix() -> str:
    """Get current UTC date as suffix for file naming."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def update_ohlcv_data() -> None:
    """Update BTC/USDT OHLCV data from Binance."""
    ohlcv_path = "data/cryptodata/dynamic_btc.parquet"

    old_data = pd.read_parquet(ohlcv_path)
    last_time = pd.to_datetime(old_data.index[-3]).timestamp() * 1000

    ccxt_api = CcxtAPI(
        "BTC/USDT", "1d", ccxt.binance(), since=int(last_time), verbose="Text"
    )

    new_data = (
        ccxt_api
        .get_all_klines()
        .to_OHLCV()
        .data_frame
    )

    btc = new_data.combine_first(old_data).drop_duplicates()
    btc.to_parquet(ohlcv_path)
    print(f"OHLCV data updated: {len(btc)} records")


def update_onchain_data(quant_node_api: QuickNodeAPI, max_blocks: int = 5) -> None:
    """
    Update on-chain block stats data sequentially.

    Parameters
    ----------
    quant_node_api : QuickNodeAPI
        QuickNode API instance for fetching block stats.
    max_blocks : int, optional
        Maximum number of blocks to fetch per run, by default 5.
    """
    start_time = time.perf_counter()

    # Get current blockchain height
    highest_height = quant_node_api.get_blockchain_info()["blocks"]
    print(f"Current blockchain height: {highest_height}")

    # Read existing data
    data = pd.read_parquet("data/onchain/BTC/block_stats_fragments")

    # Validate data integrity
    max_height_diff = data["height"].diff().max()
    if max_height_diff > 1:
        raise ValueError("There are missing blocks in the data")

    # Normalize time column
    data["time"] = np.where(data["time"] < 1e10, data["time"] * 1000, data["time"])
    data = data.set_index("time")

    # Setup incremental folder
    incremental_folder = "data/onchain/BTC/block_stats_fragments/incremental"
    if not os.path.exists(incremental_folder):
        os.makedirs(incremental_folder)
        print("Incremental folder created!")

    # Date-suffixed file path
    date_suffix = get_date_suffix()
    file_path = f"{incremental_folder}/incremental_block_stats_{date_suffix}.parquet"

    # Determine starting height
    last_height = int(data["height"].iloc[-1]) + 1
    print(f"Last saved height: {last_height - 1}")

    # Calculate batch range
    total_blocks = highest_height
    blocks_to_fetch = total_blocks - last_height

    if blocks_to_fetch <= 0:
        print("No new blocks to fetch. Data is up to date.")
        return

    batch_end = last_height + blocks_to_fetch
    print(f"Fetching blocks {last_height} to {batch_end - 1} ({blocks_to_fetch} blocks)")

    # Sequential block fetching
    batch_data = []
    for block_height in range(last_height, batch_end):
        try:
            block_stats = quant_node_api.get_block_stats(block_height)
            if block_stats:
                batch_data.append(block_stats)

            # Progress logging
            progress = block_height - last_height + 1
            print(f"Progress: {progress}/{blocks_to_fetch} - Block {block_height} fetched")

        except Exception as e:
            print(f"Failed to fetch block {block_height}: {e}")
            continue

    if not batch_data:
        print("No new blocks were successfully fetched.")
        return

    # Create DataFrame from fetched data
    new_onchain_data = pd.DataFrame(batch_data)

    # Append to existing file or create new one
    if os.path.exists(file_path):
        existing_data = pd.read_parquet(file_path)
        combined_data = pd.concat([existing_data, new_onchain_data], ignore_index=True)
        combined_data = combined_data.drop_duplicates(subset=["height"], keep="last")
    else:
        combined_data = new_onchain_data

    # Save to file (overwrite with combined data)
    combined_data.to_parquet(file_path)

    # Statistics
    elapsed_time = time.perf_counter() - start_time
    print(f"\n=== Update Complete ===")
    print(f"Blocks saved: {len(batch_data)}")
    print(f"Total records in today's file: {len(combined_data)}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"File saved: {file_path}")


def main():
    """Main function to run the BTC data updater."""
    print(f"=== BTC Single-Core Updater ===")
    print(f"Run time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # Load API keys from environment
    api_keys = []
    for x in range(1, 11):
        api_key = os.getenv(f"quicknode_endpoint_{x}")
        if api_key:
            api_keys.append(api_key)

    if not api_keys:
        raise ValueError("No QuickNode API keys found in environment variables")

    print(f"Loaded {len(api_keys)} API key(s)")

    # Initialize QuickNode API
    quant_node_api = QuickNodeAPI(api_keys, 0)

    # Update OHLCV data
    # print("\n--- Updating OHLCV Data ---")
    # update_ohlcv_data()

    # Update on-chain data (limit to 5 blocks for 30-min cron)
    print("\n--- Updating On-Chain Data ---")
    update_onchain_data(quant_node_api, max_blocks=5)

    print("\n=== Update Complete ===")


if __name__ == "__main__":
    main()
