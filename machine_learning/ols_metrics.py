from typing import Literal
import pandas as pd
import statsmodels.api as sm
from custom_exceptions.invalid_arguments import InvalidArgumentError

class OLSMetrics:
    def __init__(
        self,
        data: pd.Series,
        precision: int = 6,
        estimator:Literal["statsmodels"] = "statsmodels"
    ) -> None:
        """
        Initialize the OLSMetrics class with the given data.

        Parameters
        ----------

        data : pandas.Series
            The data used to calculate the coefficient value.
        precision : int
            The number of decimal places to round the result to.
        """
        self.data = data
        self.precision = precision
        match estimator:
            case "statsmodels":
                x = self.data.index
                x = sm.add_constant(x.astype("int64") / 10**9)
                self.estimator = sm.OLS(
                    data.astype("float64"), x, missing="drop"
                ).fit()
            case _:
                raise InvalidArgumentError(
                    f"Estimator {estimator} not supported"
                )

    def calculate_r2(self) -> float:
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
        return float(f"{self.estimator.rsquared:.{self.precision}g}")

    def calculate_coef(self) -> float:
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
        return float(f"{self.estimator.params[1]:.{self.precision}g}")
