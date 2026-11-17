"""Shared fixtures: a tiny, valid, PaySim-shaped frame."""

import numpy as np
import pandas as pd
import pytest


def make_paysim_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """
    A small valid PaySim-shaped frame with fraud in both the early and late
    time windows, so a chronological split has both classes on each side.
    Fraud rows drain the origin account and leave the destination unchanged.
    """
    rng = np.random.default_rng(seed)
    types = np.array(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"])[np.arange(n) % 4]
    step = np.linspace(1, 743, n).astype(int)
    amount = np.round(rng.uniform(10, 5000, n), 2)
    old_orig = np.round(amount + rng.uniform(0, 5000, n), 2)
    old_dest = np.round(rng.uniform(0, 10000, n), 2)
    new_orig = np.round(old_orig - amount, 2)
    new_dest = np.round(old_dest + amount, 2)

    # every 20th in-scope row is fraud (spread across all of time)
    in_scope = np.isin(types, ["TRANSFER", "CASH_OUT"])
    idx = np.flatnonzero(in_scope)
    fraud_idx = idx[::10]
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[fraud_idx] = 1
    new_orig[fraud_idx] = 0.0
    amount[fraud_idx] = old_orig[fraud_idx]
    new_dest[fraud_idx] = old_dest[fraud_idx]

    return pd.DataFrame(
        {
            "step": step,
            "type": types,
            "amount": amount,
            "nameOrig": [f"C{i}" for i in range(n)],
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "nameDest": [f"M{i}" for i in range(n)],
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": 0,
        }
    )


@pytest.fixture
def paysim_df() -> pd.DataFrame:
    return make_paysim_df()
