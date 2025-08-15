import pandas as pd
import numpy as np

from machine_learning.model_builder import adjust_max_trades


def calculate_returns(
    target: pd.Series,
    y_pred_probs: pd.Series,
    y_true: pd.Series,
    fee: float = 0,
    off_days: int = 7,
    max_trades: int = 3,
    side: int = 1,
    cutoff_point: int = 5,
) -> pd.DataFrame:
    """
    Calculate trading returns based on model predictions and target
    values.

    This function computes various return metrics including cumulative
    returns, drawdowns, and drawdown durations based on predicted
    probabilities and actual target values. It applies trading fees and
    position management constraints.

    Parameters
    ----------
    target : pd.Series
        Series containing target return values for each observation.
    y_pred_probs : pd.Series
        Series containing predicted probabilities from the model.
    y_true : pd.Series
        Series containing true binary labels (0 or 1).
    fee : float, default 0
        Trading fee as a percentage (e.g., 0.1 for 0.1%).
    off_days : int, default 7
        Number of days to wait between trades.
    max_trades : int, default 3
        Maximum number of concurrent trades allowed.
    side : int, default 1
        Trading side indicator (1 for long, -1 for short).
    cutoff_point : int, default 5
        Percentile cutoff point for prediction threshold (0-100).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the following columns:
        - y_pred_probs : Original predicted probabilities
        - Predict : Binary predictions based on cutoff
        - y_true : Binary true labels
        - target_Return : Target returns aligned to DataFrame index
        - Position : Lagged predictions for position sizing
        - Result : Raw returns from predictions
        - Liquid_Result : Returns adjusted for trading fees
        - Period_Return_cum : Cumulative period returns
        - Total_Return : Cumulative total returns
        - Liquid_Return : Cumulative liquid returns
        - max_Liquid_Return : Rolling maximum liquid returns
        - drawdown : Drawdown percentage from peak
        - drawdown_duration : Duration of current drawdown period

    Raises
    ------
    ValueError
        If cutoff_point is >= 100 or <= 0.
    """
    target_series = target.copy()

    if target_series.min() > 0:
        target_series = target_series - 1

    predict = y_pred_probs.copy()

    cutoff = np.median(predict)

    if cutoff_point:
        if cutoff_point >= 100:
            raise ValueError("Cutoff point must be less than 100")
        if cutoff_point <= 0:
            raise ValueError("Cutoff point must be greater than 0")

        predict_mask = predict > cutoff

        cutoff = np.percentile(predict[predict_mask], cutoff_point)

    df_returns = pd.DataFrame(
        {
            "y_pred_probs": y_pred_probs,
        }
    )

    df_returns["Predict"] = np.where(y_pred_probs > cutoff, 1, 0)
    df_returns["y_true"] = np.where(y_true == 1, 1, 0)

    fee = fee / 100

    target_return = target_series.reindex(df_returns.index)

    df_returns["target_Return"] = target_return

    df_returns["Position"] = df_returns["Predict"].shift().fillna(0)

    df_returns["Result"] = df_returns["target_Return"] * df_returns["Predict"]

    df_returns["Liquid_Result"] = np.where(
        (df_returns["Predict"] != 0) & (df_returns["Result"].abs() != 1),
        df_returns["Result"] - fee,
        0,
    )

    df_returns["Period_Return_cum"] = (df_returns["target_Return"]).cumsum()

    df_returns["Total_Return"] = df_returns["Result"].cumsum() + 1
    df_returns["Liquid_Return"] = df_returns["Liquid_Result"].cumsum() + 1

    df_returns["max_Liquid_Return"] = (
        df_returns["Liquid_Return"].expanding(365).max()
    )

    df_returns["max_Liquid_Return"] = np.where(
        df_returns["max_Liquid_Return"].diff(),
        np.nan,
        df_returns["max_Liquid_Return"],
    )

    df_returns["drawdown"] = (
        1 - df_returns["Liquid_Return"] / df_returns["max_Liquid_Return"]
    ).fillna(0)

    drawdown_positive = df_returns["drawdown"] > 0

    df_returns["drawdown_duration"] = drawdown_positive.groupby(
        (~drawdown_positive).cumsum()
    ).cumsum()

    df_returns["Liquid_Result"] = np.where(
        df_returns["Predict"] != side,
        0,
        df_returns["Liquid_Result"],
    )

    return adjust_max_trades(
        data_set=df_returns,
        off_days=off_days,
        max_trades=max_trades,
        pct_adj=0.5,
        side=side,
    )
