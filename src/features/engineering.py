"""
Deterministic feature engineering shared by training AND serving.

Why this file exists: the balance-error features must be computed
identically when the model is trained and when it scores a live request.
Defining them once here, and importing them from both
src/features/build_features.py and src/serving/main.py, is what prevents
training-serving skew for these columns. (The fitted preprocessor guards the
learned statistics; this guards the hand-written arithmetic.)
Responsible for: adding errorBalanceOrig / errorBalanceDest.
Must not: learn anything from data (no fit), drop rows, or touch the target.
"""

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with the two balance-consistency features added.

    errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
        0 when the origin account was debited by exactly `amount`.
    errorBalanceDest = oldbalanceDest + amount - newbalanceDest
        0 when the destination account was credited by exactly `amount`.
    """
    df = df.copy()
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df
