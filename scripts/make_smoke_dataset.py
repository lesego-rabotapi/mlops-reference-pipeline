"""
Generate a SYNTHETIC, PaySim-shaped smoke-test dataset.

This is NOT PaySim and not evidence of anything about real fraud. It exists
so CI and a fresh clone can exercise the whole pipeline (validate ->
features -> train -> serve) without downloading the ~470MB real file. It
encodes the same structural facts the real data has -- fraud only in
TRANSFER/CASH_OUT, fraud leaving the origin account drained and the
destination balance unchanged, 743 hourly steps -- so the engineered
balance-error features have something to learn, and any metrics produced
from it must be described as smoke-test metrics only.

Refuses to overwrite an existing file unless --force is passed, so it can
never clobber a real PaySim download sitting at the same path.

Usage:
    python scripts/make_smoke_dataset.py [--rows 60000] [--seed 42] [--force]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.paths import RAW_DATASET_PATH  # noqa: E402

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
TYPE_WEIGHTS = [0.22, 0.35, 0.01, 0.34, 0.08]


def make_smoke_dataset(n_rows: int = 60_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    step = np.sort(rng.integers(1, 744, size=n_rows))
    tx_type = rng.choice(TYPES, size=n_rows, p=TYPE_WEIGHTS)

    amount = np.round(rng.lognormal(mean=9.0, sigma=1.4, size=n_rows), 2)
    old_orig = np.round(amount * rng.uniform(1.0, 20.0, size=n_rows), 2)
    old_dest = np.round(rng.lognormal(mean=10.0, sigma=1.5, size=n_rows), 2)

    # Fraud: only TRANSFER / CASH_OUT, ~1.5% of those.
    scored = np.isin(tx_type, ["TRANSFER", "CASH_OUT"])
    is_fraud = (scored & (rng.random(n_rows) < 0.015)).astype(int)

    # Legitimate rows move balances consistently.
    new_orig = np.round(old_orig - amount, 2)
    new_dest = np.round(old_dest + amount, 2)
    cash_in = tx_type == "CASH_IN"
    new_orig = np.where(cash_in, np.round(old_orig + amount, 2), new_orig)
    payment_like = np.isin(tx_type, ["PAYMENT", "DEBIT"])
    new_dest = np.where(payment_like, old_dest, new_dest)  # merchants: no dest tracking

    # Fraud: drain the origin account entirely; destination balance not updated.
    fraud = is_fraud == 1
    amount = np.where(fraud, old_orig, amount)
    new_orig = np.where(fraud, 0.0, new_orig)
    new_dest = np.where(fraud, old_dest, new_dest)

    df = pd.DataFrame(
        {
            "step": step,
            "type": tx_type,
            "amount": amount,
            "nameOrig": [f"C{i:09d}" for i in range(n_rows)],
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": np.maximum(new_orig, 0.0),
            "nameDest": [f"C{n_rows + i:09d}" for i in range(n_rows)],
            "oldbalanceDest": old_dest,
            "newbalanceDest": np.maximum(new_dest, 0.0),
            "isFraud": is_fraud,
            "isFlaggedFraud": 0,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if RAW_DATASET_PATH.exists() and not args.force:
        raise SystemExit(
            f"{RAW_DATASET_PATH} already exists; refusing to overwrite it "
            "(it may be the real PaySim file). Pass --force to replace it."
        )

    RAW_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = make_smoke_dataset(args.rows, args.seed)
    df.to_csv(RAW_DATASET_PATH, index=False)
    print(
        f"Wrote SYNTHETIC smoke dataset: {RAW_DATASET_PATH} "
        f"({len(df)} rows, {int(df['isFraud'].sum())} fraud)"
    )


if __name__ == "__main__":
    main()
