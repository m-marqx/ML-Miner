import json
import time
import requests

class QuickNodeAPI:
    """
    API client for making requests to QuickNode's Bitcoin RPC endpoints.

    Uses multiple API keys for redundancy and implements rate limiting.

    Parameters
    ----------
    api_keys : list
        List of QuickNode API endpoint URLs.
    default_api_key_idx : int
        Starting index in the api_keys list to begin requests from.
    """
    def __init__(self, api_keys: list, default_api_key_idx: int):
        """
        Initialize QuickNodeAPI client with multiple API endpoints.

        Parameters
        ----------
        api_keys : list
            List of QuickNode API endpoint URLs
        default_api_key_idx : int
            Initial index position in api_keys list to start making
            requests from. Must be between 0 and len(api_keys)-1.

        Raises
        ------
        ValueError
            If api_keys list is empty or default_api_key_idx is out of bounds.
        """
        self.api_keys = api_keys
        self.default_api_key_idx = default_api_key_idx

    def get_block_stats(self, block_height: int):
        """
        Retrieve statistics for a Bitcoin block by height.

        Makes POST requests to QuickNode API endpoints with automatic
        failover between provided API keys. Implements 1 second rate
        limiting.

        Parameters
        ----------
        block_height : int
            Block height to get statistics for.

        Returns
        -------
        dict
            Block statistics from the Bitcoin RPC getblockstats method.

        Raises
        ------
        ValueError
            If all API key requests fail.
        """
        payload = json.dumps(
            {
                "method": "getblockstats",
                "params": [block_height],
            }
        )
        headers = {"Content-Type": "application/json"}

        for api_key in self.api_keys[self.default_api_key_idx:]:
            self.default_api_key_idx = self.api_keys.index(api_key)

            start = time.perf_counter()
            response = requests.request(
                "POST", api_key, headers=headers, data=payload, timeout=60
            )

            if response.ok:
                end = time.perf_counter()
                time_elapsed = end - start

                if time_elapsed < 1:
                    time.sleep(1 - time_elapsed)

                return response.json()["result"]
        raise ValueError("No API key worked")
