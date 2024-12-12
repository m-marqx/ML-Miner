import logging

def create_logger(logger_name: str, verbose: bool = False):
    """
    Create a logger for the ModelFeatures class.

    Parameters
    ----------
    verbose : bool, optional
        Whether to set the logger to verbose mode.
        (default: False)
    """
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(levelname)s %(asctime)s: %(message)s", datefmt="%H:%M:%S"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)
    logger.propagate = False

    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    return logger
