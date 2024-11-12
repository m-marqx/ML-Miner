import unittest
from unittest.mock import patch, Mock
from api.blockscout_api import BlockscoutAPI


class TestBlockscoutAPI(unittest.TestCase):
    def setUp(self):
        self.api = BlockscoutAPI(verbose=False)

    @patch("requests.get")
    def test_get_transactions(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "8676632"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "13320"},
                },
            ]
        }

        mock_get.return_value = mock_response

        result = self.api.get_transactions(
            "0x1",
            False,
        )

        expected_result = {
            "from": 8.676632,
            "to": 0.0001332,
            "USD Price": 65139.879879879874,
        }

        self.assertDictEqual(result, expected_result)

    @patch("requests.get")
    def test_get_transactions_coin_names(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "8676632"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "13320"},
                },
            ]
        }

        mock_get.return_value = mock_response

        result = self.api.get_transactions(
            "0x1",
            True,
        )

        expected_result = {
            "from": 8.676632,
            "to": 0.0001332,
            "USD Price": 65139.879879879874,
            "from_coin_name": "USDT",
            "to_coin_name": "WBTC",
        }

        self.assertDictEqual(result, expected_result)

    @patch("requests.get")
    def test_get_transactions_no_coin_names(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7757769"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12859"},
                },
            ]
        }

        mock_get.return_value = mock_response

        result = self.api.get_transactions(
            "0x1",
            False,
        )

        expected_result = {
            'from': 7.757769,
            'to': 0.00012859,
            'USD Price': 60329.489073800454,
        }

        self.assertDictEqual(result, expected_result)

    @patch("requests.get")
    def test_get_account_transactions_no_coin_names(self, mock_get):
        mock_response_1 = Mock()
        mock_response_1.json.return_value = {
            "items": [
                {
                    "hash": "0xf0",
                    "method" : "processRouteWithTransferValueOutput",
                    "fee" : {'type': 'actual', 'value': '7022820013343358'},
                    "status": "ok",
                },
                {
                    "hash": "0xf1",
                    "method" : "processRouteWithTransferValueOutput",
                    "fee" : {'type': 'actual', 'value': '4945950006759465'},
                    "status": "ok",
                },
                {
                    "hash": "0xf2",
                    "method" : "processRouteWithTransferValueOutput",
                    "fee" : {'type': 'actual', 'value': '6832756486003916'},
                    "status": "ok",
                },
                {
                    "hash": "0xf3",
                    "method" : "processRouteWithTransferValueOutput",
                    "fee" : {'type': 'actual', 'value': '7861015823084520'},
                    "status": "ok",
                },
                {
                    "hash": "0xf4",
                    "method" : "processRouteWithTransferValueOutput",
                    "fee" : {'type': 'actual', 'value': '4802190080676792'},
                    "status": "ok",
                },
                {
                    "hash": "0xf5",
                    "method" : "processRouteWithTransferValueOutput",
                    "fee" : {'type': 'actual', 'value': '9602190080676792'},
                    "status": "error",
                },
            ]
        }

        mock_response_2 = Mock()
        mock_response_2.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7757769"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12859"},
                },
            ]
        }

        mock_response_3 = Mock()
        mock_response_3.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7740000"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12831"},
                },
            ]
        }

        mock_response_4 = Mock()
        mock_response_4.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12602"},
                },
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7594096"},
                },
            ]
        }

        mock_response_5 = Mock()
        mock_response_5.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12647"},
                },
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7903673"},
                },
            ]
        }

        mock_response_6 = Mock()
        mock_response_6.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7664795"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12589"},
                },
            ]
        }

        mock_get.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_response_3,
            mock_response_4,
            mock_response_5,
            mock_response_6,
        ]

        result = self.api.get_account_transactions(
            "0x1",
            False,
        )

        expected_result = [
            {
                "from": 7.757769,
                "to": 0.00012859,
                "USD Price": 60329.489073800454,
                "txn_fee": 0.007022820013343358,
            },
            {
                "from": 7.74,
                "to": 0.00012831,
                "USD Price": 60322.656067336924,
                "txn_fee": 0.004945950006759465,
            },
            {
                "from": 0.00012602,
                "to": 7.594096,
                "USD Price": 60261.03793048723,
                "txn_fee": 0.006832756486003916,
            },
            {
                "from": 0.00012647,
                "to": 7.903673,
                "USD Price": 62494.44927650827,
                "txn_fee": 0.00786101582308452,
            },
            {
                "from": 7.664795,
                "to": 0.00012589,
                "USD Price": 60884.85979823655,
                "txn_fee": 0.004802190080676792,
            },
        ]

        self.assertListEqual(result, expected_result)

    @patch("requests.get")
    def test_get_account_transactions_coin_names(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [
                {
                    "hash": "0xf0",
                    "method" : "processRouteWithTransferValueOutput",
                    "fee" : {'type': 'actual', 'value': '7022820013343358'},
                    "status": "ok",
                },
                {
                    "hash": "0xf1",
                    "method" : "processRouteWithTransferValueOutput",
                    "fee" : {'type': 'actual', 'value': '4945950006759465'},
                    "status": "ok",
                },
                {
                    "hash": "0xf2",
                    "method" : "processRouteWithTransferValueOutput",
                    "fee" : {'type': 'actual', 'value': '6832756486003916'},
                    "status": "ok",
                },
                {
                    "hash": "0xf3",
                    "method" : "processRouteWithTransferValueOutput",
                    "fee" : {'type': 'actual', 'value': '7861015823084520'},
                    "status": "ok",
                },
                {
                    "hash": "0xf4",
                    "method" : "processRouteWithTransferValueOutput",
                    "fee" : {'type': 'actual', 'value': '4802190080676792'},
                    "status": "ok",
                },
                {
                    "hash": "0xf5",
                    "method" : "processRouteWithTransferValueOutput",
                    "fee" : {'type': 'actual', 'value': '9602190080676792'},
                    "status": "error",
                },
            ]
        }

        mock_response_1 = Mock()
        mock_response_1.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7757769"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12859"},
                },
            ]
        }

        mock_response_2 = Mock()
        mock_response_2.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7740000"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12831"},
                },
            ]
        }

        mock_response_3 = Mock()
        mock_response_3.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12602"},
                },
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7594096"},
                },
            ]
        }

        mock_response_4 = Mock()
        mock_response_4.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12647"},
                },
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7903673"},
                },
            ]
        }

        mock_response_5 = Mock()
        mock_response_5.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7664795"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12589"},
                },
            ]
        }

        mock_get.side_effect = [
            mock_response,
            mock_response_1,
            mock_response_2,
            mock_response_3,
            mock_response_4,
            mock_response_5,
        ]

        result = self.api.get_account_transactions(
            "0x1",
            True,
        )

        expected_result = [
            {
                "from": 7.757769,
                "to": 0.00012859,
                "USD Price": 60329.489073800454,
                "from_coin_name": "USDT",
                "to_coin_name": "WBTC",
                "txn_fee": 0.007022820013343358,
            },
            {
                "from": 7.74,
                "to": 0.00012831,
                "USD Price": 60322.656067336924,
                "from_coin_name": "USDT",
                "to_coin_name": "WBTC",
                "txn_fee": 0.004945950006759465,
            },
            {
                "from": 0.00012602,
                "to": 7.594096,
                "USD Price": 60261.03793048723,
                "from_coin_name": "WBTC",
                "to_coin_name": "USDT",
                "txn_fee": 0.006832756486003916,
            },
            {
                "from": 0.00012647,
                "to": 7.903673,
                "USD Price": 62494.44927650827,
                "from_coin_name": "WBTC",
                "to_coin_name": "USDT",
                "txn_fee": 0.00786101582308452,
            },
            {
                "from": 7.664795,
                "to": 0.00012589,
                "USD Price": 60884.85979823655,
                "from_coin_name": "USDT",
                "to_coin_name": "WBTC",
                "txn_fee": 0.004802190080676792,
            },
        ]

        self.assertListEqual(result, expected_result)
