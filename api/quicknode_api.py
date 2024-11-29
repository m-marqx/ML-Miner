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

