from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_FILE


MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_main_model.joblib"

REPORT_DIR = PROJECT_ROOT / "reports" / "model_results"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


NUMERIC_FEATURES = [
    "amount",
    "log_amount",
    "distance_from_home",
    "transaction_hour",
    "weekend_transaction",
    "card_present",
    "customer_prior_transactions",
    "minutes_since_previous_transaction",
    "previous_customer_amount",
    "customer_avg_amount_prior",
    "customer_std_amount_prior",
    "amount_vs_customer_average",
    "amount_zscore_vs_customer_history",
    "calculated_txn_count_1h",
    "calculated_amount_1h",
    "calculated_txn_count_24h",
    "calculated_amount_24h",
    "new_merchant_for_customer_flag",
    "overnight_transaction_flag",
]

CATEGORICAL_FEATURES = [
    "merchant_category",
    "merchant_type",
    "merchant",
    "currency",
    "country",
    "city_size",
    "card_type",
    "device",
    "channel",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FINAL_THRESHOLD = 0.75


def main():

    start = time.time()

    print("Payment Fraud Analysis")
    print("Phase 4.3: Final Untouched Test")
    print(f"Threshold locked from validation: {FINAL_THRESHOLD}")

    print("\n[1/4] Loading model...")

    model = joblib.load(MODEL_PATH)

    con = duckdb.connect(str(DB_FILE))

    try:

        columns = ", ".join(FEATURES)

        print("\n[2/4] Loading final test period...")

        test = con.execute(
            f"""
            SELECT
                transaction_key,
                {columns},
                is_fraud
            FROM modeling.test
            """
        ).fetchdf()

    finally:
        con.close()

    print(f"Test transactions: {len(test):,}")

    print("\n[3/4] Preparing features...")

    for column in CATEGORICAL_FEATURES:

        test[column] = (
            test[column]
            .fillna("MISSING")
            .astype(str)
        )

        # Match training categories stored inside LightGBM model
        category_values = (
            model._Booster.pandas_categorical[
                CATEGORICAL_FEATURES.index(column)
            ]
        )

        test[column] = pd.Categorical(
            test[column],
            categories=category_values
        )

    X_test = test[FEATURES]
    y_test = test["is_fraud"].astype(int)

    print("\n[4/4] Running final evaluation...")

    probabilities = model.predict_proba(
        X_test,
        num_iteration=model.best_iteration_
    )[:, 1]

    predictions = (
        probabilities >= FINAL_THRESHOLD
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics = pd.DataFrame(
        [
            {
                "model": "lightgbm_final_test",
                "threshold": FINAL_THRESHOLD,
                "test_rows": len(test),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "review_rate": float(
                    predictions.mean()
                ),
            }
        ]
    )

    print("\nFINAL TEST RESULTS")
    print(metrics.T.to_string(header=False))

    metrics.to_csv(
        REPORT_DIR / "final_test_metrics.csv",
        index=False
    )

    output = pd.DataFrame(
        {
            "transaction_key":
                test["transaction_key"],
            "actual_fraud":
                y_test,
            "fraud_probability":
                probabilities,
            "predicted_fraud":
                predictions,
        }
    )

    output.to_parquet(
        REPORT_DIR /
        "final_test_predictions.parquet",
        index=False
    )

    minutes = (
        time.time() - start
    ) / 60

    print(
        f"\nPhase 4.3 complete in "
        f"{minutes:.1f} minutes."
    )

    print("\nCreated:")
    print(
        "  reports/model_results/"
        "final_test_metrics.csv"
    )
    print(
        "  reports/model_results/"
        "final_test_predictions.parquet"
    )

    print(
        "\nNext: business-cost threshold "
        "and manual-review capacity analysis."
    )


if __name__ == "__main__":
    main()