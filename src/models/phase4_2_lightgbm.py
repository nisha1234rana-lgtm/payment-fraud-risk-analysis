from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
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


REPORT_DIR = PROJECT_ROOT / "reports" / "model_results"
MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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


def load_data(con):

    columns = ", ".join(FEATURES)

    print("\n[1/6] Loading development training sample...")

    train = con.execute(
        f"""
        SELECT
            transaction_key,
            {columns},
            is_fraud
        FROM modeling.train_dev_sample
        """
    ).fetchdf()

    print(f"Training rows: {len(train):,}")

    print("\n[2/6] Loading validation sample...")

    validation = con.execute(
        f"""
        SELECT
            transaction_key,
            {columns},
            is_fraud
        FROM modeling.validation
        WHERE ABS(HASH(transaction_key)) % 5 = 0
        """
    ).fetchdf()

    print(f"Validation rows: {len(validation):,}")

    return train, validation


def prepare_categories(train, validation):

    for column in CATEGORICAL_FEATURES:

        train[column] = (
            train[column]
            .fillna("MISSING")
            .astype(str)
        )

        validation[column] = (
            validation[column]
            .fillna("MISSING")
            .astype(str)
        )

        categories = pd.Index(
            train[column].unique()
        )

        train[column] = pd.Categorical(
            train[column],
            categories=categories
        )

        validation[column] = pd.Categorical(
            validation[column],
            categories=categories
        )

    return train, validation


def threshold_table(y_true, probabilities):

    results = []

    for threshold in np.arange(
        0.10,
        0.91,
        0.05
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).ravel()

        results.append(
            {
                "threshold": round(
                    float(threshold),
                    2
                ),

                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0
                ),

                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0
                ),

                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0
                ),

                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),

                "review_rate":
                    float(predictions.mean()),
            }
        )

    return pd.DataFrame(results)


def main():

    start = time.time()

    print("Payment Fraud Analysis")
    print("Phase 4.2: LightGBM")
    print(f"Database: {DB_FILE}")

    con = duckdb.connect(str(DB_FILE))

    try:
        train, validation = load_data(con)

    finally:
        con.close()

    print("\n[3/6] Preparing categorical features...")

    train, validation = prepare_categories(
        train,
        validation
    )

    X_train = train[FEATURES]
    y_train = train["is_fraud"].astype(int)

    X_val = validation[FEATURES]
    y_val = validation["is_fraud"].astype(int)

    print("\n[4/6] Training LightGBM...")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,

        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=0.1,
        reg_lambda=0.1,

        class_weight="balanced",

        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,

        categorical_feature=
            CATEGORICAL_FEATURES,

        eval_set=[
            (X_val, y_val)
        ],

        eval_metric="average_precision",

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=75,
                verbose=True
            )
        ],
    )

    print(
        f"\nBest iteration: "
        f"{model.best_iteration_}"
    )

    print("\n[5/6] Evaluating validation performance...")

    probabilities = model.predict_proba(
        X_val,
        num_iteration=model.best_iteration_
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    roc_auc = roc_auc_score(
        y_val,
        probabilities
    )

    pr_auc = average_precision_score(
        y_val,
        probabilities
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics = pd.DataFrame(
        [
            {
                "model":
                    "lightgbm_main_model",

                "training_rows":
                    len(train),

                "validation_rows":
                    len(validation),

                "best_iteration":
                    model.best_iteration_,

                "roc_auc":
                    roc_auc,

                "pr_auc":
                    pr_auc,

                "precision_0_5":
                    precision,

                "recall_0_5":
                    recall,

                "f1_0_5":
                    f1,

                "true_positives":
                    int(tp),

                "false_positives":
                    int(fp),

                "false_negatives":
                    int(fn),

                "true_negatives":
                    int(tn),
            }
        ]
    )

    print("\nValidation metrics:")
    print(
        metrics.T.to_string(
            header=False
        )
    )

    metrics.to_csv(
        REPORT_DIR /
        "lightgbm_metrics.csv",
        index=False
    )

    print("\n[6/6] Evaluating thresholds...")

    thresholds = threshold_table(
        y_val,
        probabilities
    )

    thresholds.to_csv(
        REPORT_DIR /
        "lightgbm_threshold_results.csv",
        index=False
    )

    best_row = thresholds.loc[
        thresholds["f1"].idxmax()
    ]

    print("\nBest F1 threshold:")
    print(best_row.to_string())

    # ------------------------------------------
    # Feature importance
    # ------------------------------------------

    importance = pd.DataFrame(
        {
            "feature":
                FEATURES,

            "importance_gain":
                model.booster_
                .feature_importance(
                    importance_type="gain"
                ),

            "importance_split":
                model.booster_
                .feature_importance(
                    importance_type="split"
                ),
        }
    )

    importance = importance.sort_values(
        "importance_gain",
        ascending=False
    )

    importance.to_csv(
        REPORT_DIR /
        "lightgbm_feature_importance.csv",
        index=False
    )

    print("\nTop 15 features:")
    print(
        importance.head(15)
        .to_string(index=False)
    )

    predictions_output = pd.DataFrame(
        {
            "transaction_key":
                validation[
                    "transaction_key"
                ],

            "actual_fraud":
                y_val,

            "fraud_probability":
                probabilities,
        }
    )

    predictions_output.to_parquet(
        REPORT_DIR /
        "lightgbm_validation_predictions.parquet",
        index=False
    )

    joblib.dump(
        model,
        MODEL_DIR /
        "lightgbm_main_model.joblib"
    )

    metadata = {

        "numeric_features":
            NUMERIC_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "training_rows":
            len(train),

        "validation_rows":
            len(validation),

        "best_iteration":
            int(model.best_iteration_),

        "roc_auc":
            float(roc_auc),

        "pr_auc":
            float(pr_auc),

        "feature_policy":
            "Main-model features only. "
            "Near-deterministic synthetic "
            "device-network signals excluded."
    }

    with open(
        REPORT_DIR /
        "lightgbm_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    minutes = (
        time.time() - start
    ) / 60

    print(
        f"\nPhase 4.2 complete "
        f"in {minutes:.1f} minutes."
    )

    print("\nCreated:")

    print(
        "  models/lightgbm_main_model.joblib"
    )

    print(
        "  reports/model_results/"
        "lightgbm_metrics.csv"
    )

    print(
        "  reports/model_results/"
        "lightgbm_threshold_results.csv"
    )

    print(
        "  reports/model_results/"
        "lightgbm_feature_importance.csv"
    )

    print(
        "\nNext: compare Logistic Regression "
        "vs LightGBM and choose the production candidate."
    )


if __name__ == "__main__":
    main()