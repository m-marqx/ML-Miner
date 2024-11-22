import logging
import pandas as pd

from api.moralis_api import MoralisAPI
from api.blockscout_api import BlockscoutAPI

from abc import ABC, abstractmethod


class APIHandler(ABC):
    def __init__(self):
        self._next_handler = None
        self.verbose = False
        self.api_key = ""
        self.logger = logging.getLogger("API_Handler")
        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s: %(message)s", datefmt="%H:%M:%S"
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.addHandler(handler)
        self.logger.propagate = False

    def set_next_api(
        self,
        handler: MoralisAPI | BlockscoutAPI,
    ) -> MoralisAPI | BlockscoutAPI:
        self._next_handler = handler
        return handler

    def handle(self, wallet: str, coin_name: bool = False):
        try:
            return self.get_account_swaps(wallet, coin_name)
        except Exception as e:
            if self._next_handler:
                self.logger.warning("Moralis API failed")
                self.logger.warning("Error: %s", e)
                return self._next_handler.handle(wallet, coin_name)
            raise e

    @abstractmethod
    def get_account_swaps(self, wallet: str, coin_name: bool = False):
        ...
