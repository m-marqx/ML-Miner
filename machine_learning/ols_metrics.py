from ast import literal_eval

import pandas as pd
import plotly.express as px
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


def calculate_coef(data: pd.Series) -> float:
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
    return literal_eval(
        px.scatter(data, trendline="ols")
        .data[1]["hovertemplate"]
        .split("<br>")[1]
        .split("*")[0]
        .split(" ")[-2]
    )
