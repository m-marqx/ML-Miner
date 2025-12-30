import os
import time

import ccxt
import pandas as pd
from joblib import Parallel, delayed

from crypto_explorer import QuickNodeAPI, CcxtAPI

def main():
    api_keys = []

    for x in range(1, 11):
        api_key = os.getenv(f"quicknode_endpoint_{x}")

        if api_key:
            api_keys.append(api_key)

    quant_node_api = QuickNodeAPI(api_keys, 0)
    last_height = 0
    batch_data = []
    highest_height = quant_node_api.get_blockchain_info()["blocks"]

    old_data = pd.read_parquet("data/cryptodata/dynamic_btc.parquet")
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
    btc.to_parquet("data/cryptodata/dynamic_btc.parquet")

    data = pd.read_parquet("data/onchain/BTC/block_stats_fragments")

    max_height_diff = data["height"].diff().max()

    if max_height_diff > 1:
        raise ValueError("There are missing blocks in the data")

    data["time"] = pd.to_datetime(data["time"], unit="s")
    data = data.set_index("time")

    new_file_folder = "data/onchain/BTC/block_stats_fragments/incremental"

    if not os.path.exists(new_file_folder):
        os.makedirs(new_file_folder)
        print("new folder created!")

    file_path = f"{new_file_folder}/incremental_block_stats.parquet"

    if file_path:
        last_height = data["height"].iloc[-1] + 1
        print(f"last height: {last_height}")

    total_blocks = highest_height

    print(highest_height)

    initial_size = len(batch_data) + last_height
    batch_end = initial_size + 10000
    print(last_height < total_blocks)
    print(batch_end < total_blocks)
    print(last_height)
    print(total_blocks)
    print(batch_end)

    while (last_height < total_blocks) or (batch_end < total_blocks):
        if batch_end == total_blocks:
            break

        initial_size = len(batch_data) + last_height
        batch_end = initial_size + 10000

        if batch_end > total_blocks:
            batch_end = total_blocks

        batch_data += Parallel(n_jobs=8, verbose=10000)(
            delayed(quant_node_api.get_block_stats)(x)
            for x in range(initial_size, batch_end)
        )

        onchain_data = pd.DataFrame(batch_data)

        elapsed_time = time.perf_counter() - start
        saved_blocks_delta = batch_end - last_height

        request_sec = saved_blocks_delta / elapsed_time
        restant_time = (total_blocks - batch_end) / request_sec

        print(f"Ultimo salvamento: {time.ctime()}")
        print(f"Api Key Index: {quant_node_api.default_api_key_idx}")
        print(f"\nBlocos salvos: {batch_end}")
        print(f"\nTempo decorrido: {elapsed_time}")
        print(f"Tempo restante: {restant_time}")

        btc_data = pd.read_parquet(file_path)

        pd.concat([btc_data, onchain_data]).to_parquet(file_path)


main()
if __name__ == "__main__":
    while True:
        current_time = time.gmtime()
        print(time.strftime("%d/%m/%Y %H:%M:%S", current_time))
        if current_time.tm_hour == 0 and current_time.tm_min == 0:
            print("Updating BTC data")
            main()
            print("BTC data updated")
            time.sleep(60)  # Sleep for 1 minute to avoid multiple executions
        else:
            print("Sleeping for 20 seconds")
            time.sleep(20)  # Check every 30 seconds
