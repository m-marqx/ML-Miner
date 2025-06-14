import pandas as pd
import numpy as np

def adjust_predict_one_side(
    predict: pd.Series,
    max_trades: int,
    target_days: int,
    side: int = 1,
) -> pd.Series:
    """
    Adjusts the maximum trades on one side of the data set.

    Parameters
    ----------
    predict : pd.Series
        The input series containing the predicted values.
    max_trades : int
        The maximum number of trades.
    target_days : int
        The number of days to consider for trade calculation.
    side : int, optional
        The side of the trade to adjust (1 for long and -1 for short).
        (default: 1).

    Returns
    -------
    pd.Series
        The adjusted series with maximum trades on one side.
    """
    predict_numpy = predict.to_numpy()
    target = np.where(predict_numpy == side, predict_numpy, 0)

    if side not in (-1, 1):
        raise ValueError("side must be 1 or -1")

    for idx in range(max_trades, len(predict_numpy)):
        if predict_numpy[idx] != 0:
            open_trades = np.sum(target[idx-(target_days):idx + 1])

            if side > 0 and open_trades > max_trades:
                target[idx] = 0
            elif side < 0 and open_trades < -max_trades:
                target[idx] = 0

    return pd.Series(target, index=predict.index, name=predict.name)

def adjust_predict_both_side(
    data_set: pd.DataFrame,
    off_days: int,
    max_trades: int,
):
    """
    Adjusts the maximum trades on both sides of the data set.

    Parameters
    ----------
    data_set : pd.Series
        The input data set.
    off_days : int
        The number of off days.
    max_trades : int
        The maximum number of trades.

    Returns
    -------
    pd.Series
        The adjusted data set.
    """
    for idx, row in data_set.iloc[max_trades:].iterrows():
        if row["Predict"] != 0:
            three_lag_days = data_set.loc[:idx].iloc[-(max_trades + 1) : -1]
            three_lag_days_trades = three_lag_days["Predict"].abs().sum()
            if three_lag_days_trades >= max_trades:
                data_set.loc[idx:, "Predict"].iloc[0:off_days] = 0
    return data_set
