import unittest
from utils.time_utils import interval_to_milliseconds
from custom_exceptions.invalid_arguments import InvalidArgumentError

class TestIntervalToMilliseconds(unittest.TestCase):
    def test_valid_intervals(self):
        self.assertEqual(interval_to_milliseconds("1s"), 1000)
        self.assertEqual(interval_to_milliseconds("3s"), 3000)
        self.assertEqual(interval_to_milliseconds("1m"), 60000)
        self.assertEqual(interval_to_milliseconds("3m"), 180000)
        self.assertEqual(interval_to_milliseconds("1h"), 3600000)
        self.assertEqual(interval_to_milliseconds("3h"), 10800000)
        self.assertEqual(interval_to_milliseconds("1d"), 86400000)
        self.assertEqual(interval_to_milliseconds("3d"), 259200000)
        self.assertEqual(interval_to_milliseconds("1w"), 604800000)
        self.assertEqual(interval_to_milliseconds("3w"), 1814400000)

    def test_invalid_intervals(self):
        with self.assertRaises(InvalidArgumentError):
            interval_to_milliseconds("1x")
        with self.assertRaises(InvalidArgumentError):
            interval_to_milliseconds("xm")
        with self.assertRaises(InvalidArgumentError):
            interval_to_milliseconds("")

    def test_non_integer_prefix(self):
        with self.assertRaises(InvalidArgumentError):
            interval_to_milliseconds("1.5m")
        with self.assertRaises(InvalidArgumentError):
            interval_to_milliseconds("m")
