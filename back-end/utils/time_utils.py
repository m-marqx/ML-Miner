"""
Module for time-related utility functions.

This module provides functions for converting Binance interval strings
to milliseconds.

Functions:
- interval_to_milliseconds: Convert a Binance interval string to
milliseconds.
"""

from custom_exceptions.invalid_arguments import InvalidArgumentError


def interval_to_milliseconds(interval: str) -> int:
    """
    Convert a interval string to milliseconds

    Parameters
    ----------
    interval : str
        The interval string to convert. E.g. "1m", "1h", "1d", etc.

    Returns
    -------
    int
        The interval in milliseconds.

    Raises
    ------
    InvalidArgumentError
        If the interval string is invalid.

    """
    if not interval:
        raise InvalidArgumentError("Invalid interval string.")

    seconds_per_unit: dict = {
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }
    has_unit = interval[-1] in seconds_per_unit
    is_integer_interval = interval[:-1].isdigit()

    if has_unit and is_integer_interval:
        return int(interval[:-1]) * seconds_per_unit[interval[-1]] * 1000

    raise InvalidArgumentError("Invalid interval string.")
