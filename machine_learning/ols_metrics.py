import pandas as pd
import statsmodels.api as sm


def calculate_r2(data: pd.Series, precision: int = 6) -> float:
    """
    Calculate the R-squared value from the given data.

    Parameters
    ----------
    data : pandas.DataFrame
        The data used to calculate the R-squared value.

    Returns
    -------
    float
        The R-squared value.

    """
    x = data.reset_index().index
    x = sm.add_constant(x)
    fit_results = sm.OLS(data, x, missing="drop").fit()

    return round(fit_results.rsquared, precision)


def calculate_coef(data: pd.Series, precision: int = 7) -> float:
    """
    Calculate the coefficient value from the given data.

    Parameters
    ----------
    data : pandas.Series
        The data used to calculate the coefficient value.

    Returns
    -------
    float
        The coefficient value.
    """
    x = data.reset_index().index
    x = sm.add_constant(x)
    fit_results = sm.OLS(data, x, missing="drop").fit()

    return round(fit_results.params[1], precision)
