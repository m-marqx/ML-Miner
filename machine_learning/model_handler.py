from typing import Literal
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn import metrics
from sklearn.model_selection import (
    learning_curve,
)


class ModelHandler:
    """
    A class for handling machine learning model evaluation.

    Parameters
    ----------
    estimator : object
        The machine learning model to be evaluated.
    X_test : array-like of shape (n_samples, n_features)
        Testing input samples.
    y_test : array-like of shape (n_samples,)
        True target values for testing.

    Attributes
    ----------
    x_test : array-like of shape (n_samples, n_features)
        Testing input samples.
    y_test : array-like of shape (n_samples,)
        True target values for testing.
    estimator : object
        The machine learning model.
    y_pred_probs : array-like of shape (n_samples,), optional
        Predicted class probabilities (if available).
    _has_predic_proba : bool
        Indicates whether the estimator has predict_proba method.

    Properties:
    -----------
    results_report : str
        A string containing a results report including a confusion matrix,
        a classification report, AUC, Gini index (if predict_proba is
        available), and support.
    """

    def __init__(self, estimator, X_test, y_test) -> None:
        """
        Initialize the ModelHandler object.

        Parameters
        ----------
        estimator : object
            An instance of a scikit-learn estimator for classification or
            regression.
        X_test : array-like of shape (n_samples, n_features)
            Test input samples.
        y_test : array-like of shape (n_samples,)
            True target values for testing.
        """
        self.x_test = X_test
        self.y_test = y_test
        self.estimator = estimator
        self.y_pred_probs = None
        self.y_pred = estimator.predict(X_test)
        self._has_predic_proba = (
            hasattr(estimator, 'predict_proba')
            and callable(getattr(estimator, 'predict_proba'))
        )

        if self._has_predic_proba:
            self.y_pred_probs = estimator.predict_proba(X_test)[:, 1]

    def model_returns(
        self,
        target_series: pd.Series,
        fee: float = 0.1,
        cutoff: float = 0.5,
        step: float = 0.0,
        long_only: bool = False,
        short_only: bool = False,
        drawdown_min_window: int = 365
    ) -> pd.DataFrame:
        """
        Calculate returns and performance metrics for a trading model.

        This method calculates returns and various performance metrics
        for a trading model using predicted probabilities and actual
        returns. It takes into account transaction fees for trading.

        Parameters
        ----------
        target_series : pd.Series
            A pandas Series containing the actual returns of the trading
            strategy.
        fee : float, optional
            The transaction fee as a percentage (e.g., 0.1% for 0.1)
            for each trade.
            (default: 0.1)

        Returns
        -------
        pd.DataFrame:
            A tuple containing:
            - pd.DataFrame: A DataFrame with various columns
            representing the trading results

        Raises:
        -------
        ValueError:
            If the estimator isn't suitable for classification
            (predict_proba isn't available).
        """
        if not self._has_predic_proba:
            raise ValueError(
                "The estimator isn't suitable for classification"
                " (predict_proba isn't available)."
            )

        if target_series.min() > 0:
            target_series = target_series - 1

        fee = fee / 100
        df_returns = (
            pd.DataFrame(
                {'y_pred_probs': self.y_pred_probs},
                self.x_test.index,
            )
        )

        target_return = target_series.reindex(df_returns.index)

        df_returns["target_Return"] = target_return

        if long_only:
            df_returns["Predict"] = np.where(
                (df_returns["y_pred_probs"] > cutoff + step), 1, 0
            )
        elif short_only:
            df_returns["Predict"] = np.where(
                (df_returns["y_pred_probs"] < cutoff - step), -1, 0
            )
        elif step > 0 and step is not None:
            df_returns["Predict"] = np.where(
                (df_returns["y_pred_probs"] > cutoff + step), 1, 0
            )

            df_returns["Predict"] = np.where(
                (df_returns["y_pred_probs"] < cutoff - step),
                -1, df_returns["Predict"]
            )
        else:
            df_returns["Predict"] = np.where(
                (df_returns["y_pred_probs"] > cutoff), 1, -1
            )

        df_returns["Position"] = df_returns["Predict"].shift().fillna(0)

        df_returns["Result"] = (
            df_returns["target_Return"]
            * df_returns["Predict"]
        )

        df_returns["Liquid_Result"] = np.where(
            (df_returns["Predict"] != 0)
            & (df_returns["Result"].abs() != 1),
            df_returns["Result"] - fee, 0
        )

        df_returns["Period_Return_cum"] = (
            df_returns["target_Return"]
        ).cumsum()

        df_returns["Total_Return"] = df_returns["Result"].cumsum() + 1
        df_returns["Liquid_Return"] = df_returns["Liquid_Result"].cumsum() + 1

        df_returns["max_Liquid_Return"] = (
            df_returns["Liquid_Return"].expanding(drawdown_min_window).max()
        )

        df_returns["max_Liquid_Return"] = np.where(
            df_returns["max_Liquid_Return"].diff(),
            np.nan, df_returns["max_Liquid_Return"],
        )

        df_returns["drawdown"] = (
            1 - df_returns["Liquid_Return"] / df_returns["max_Liquid_Return"]
        ).fillna(0)

        drawdown_positive = df_returns["drawdown"] > 0

        df_returns["drawdown_duration"] = drawdown_positive.groupby(
            (~drawdown_positive).cumsum()
        ).cumsum()
        return df_returns

    @property
    def results_report(self) -> str:
        """
        Generate a results report including a confusion matrix and a
        classification report.

        Returns
        -------
        str
            A string containing the results report.
        """
        if not self._has_predic_proba:
            raise ValueError(
                "The estimator isn't suitable for classification"
                " (predict_proba isn't available)."
            )

        names = pd.Series(self.y_test).sort_values().astype(str).unique()

        confusion_matrix = metrics.confusion_matrix(self.y_test, self.y_pred)
        column_names = "predicted_" + names
        index_names = "real_" + names

        confusion_matrix_df = pd.DataFrame(
            confusion_matrix,
            columns=column_names,
            index=index_names,
        )

        auc = metrics.roc_auc_score(self.y_test, self.y_pred_probs)
        gini = 2 * auc - 1
        support = self.y_test.shape[0]
        classification_report = metrics.classification_report(
            self.y_test, self.y_pred, digits=4
        )[:-1]

        auc_str = (
            f"\n         AUC                         {auc:.4f}"
            f"      {support}"
            f"\n        Gini                         {gini:.4f}"
            f"      {support}"
        )

        confusion_matrix_str = (
            f"Confusion matrix"
            f"\n--------------------------------------------------------------"
            f"\n{confusion_matrix_df}"
            f"\n"
            f"\n"
            f"\nClassification reports"
            f"\n--------------------------------------------------------------"
            f"\n"
            f"\n{classification_report}"
            f"{auc_str}"
        )
        return confusion_matrix_str
