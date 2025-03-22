class ApiError(Exception):
    """
    Exception raised for errors in the API.

    Attributes
    ----------
    message : str
        Explanation of the error.
    """
