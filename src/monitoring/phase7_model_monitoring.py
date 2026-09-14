from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_FILE


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lightgbm_main_model.joblib"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "monitoring"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

DECISION_THRESHOLD = 0.75

NUMERIC_FEATURES = [
    "amount",
    "distance_from_home",
    "transaction_hour",
    "customer_prior_transactions",
    "minutes_since_previous_transaction",
    "customer_avg_amount_prior",
    "amount_vs_customer_average",
    "calculated_txn_count_1h",
    "calculated_amount_24h",
]

CATEGORICAL_FEATURES = [
    "merchant_category",
    "currency",
    "country",
    "card_type",
    "device",
    "channel",
]

MODEL_FEATURES = [
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

MODEL_CATEGORICAL_FEATURES = [
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


# ============================================================
# HELPERS
# ============================================================

def prepare_categories(
    dataframe: pd.DataFrame,
    model,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    for index, column in enumerate(
        MODEL_CATEGORICAL_FEATURES
    ):

        dataframe[column] = (
            dataframe[column]
            .fillna("MISSING")
            .astype(str)
        )

        categories = (
            model._Booster
            .pandas_categorical[index]
        )

        dataframe[column] = pd.Categorical(
            dataframe[column],
            categories=categories,
        )

    return dataframe


def calculate_numeric_psi(
    reference,
    current,
    bins=10,
):

    reference = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    current = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )

    edges = np.unique(
        reference.quantile(
            quantiles
        ).values
    )

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts = np.histogram(
        reference,
        bins=edges,
    )[0]

    current_counts = np.histogram(
        current,
        bins=edges,
    )[0]

    reference_pct = (
        reference_counts
        / max(
            reference_counts.sum(),
            1,
        )
    )

    current_pct = (
        current_counts
        / max(
            current_counts.sum(),
            1,
        )
    )

    epsilon = 0.000001

    reference_pct = np.where(
        reference_pct == 0,
        epsilon,
        reference_pct,
    )

    current_pct = np.where(
        current_pct == 0,
        epsilon,
        current_pct,
    )

    psi = np.sum(
        (
            current_pct
            - reference_pct
        )
        * np.log(
            current_pct
            / reference_pct
        )
    )

    return float(psi)


def calculate_categorical_psi(
    reference,
    current,
):

    reference = (
        reference
        .fillna("MISSING")
        .astype(str)
    )

    current = (
        current
        .fillna("MISSING")
        .astype(str)
    )

    categories = sorted(
        set(reference.unique())
        | set(current.unique())
    )

    reference_dist = (
        reference
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0,
        )
    )

    current_dist = (
        current
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0,
        )
    )

    epsilon = 0.000001

    reference_dist = (
        reference_dist
        .replace(
            0,
            epsilon,
        )
    )

    current_dist = (
        current_dist
        .replace(
            0,
            epsilon,
        )
    )

    psi = (
        (
            current_dist
            - reference_dist
        )
        * np.log(
            current_dist
            / reference_dist
        )
    ).sum()

    return float(psi)


def drift_status(
    psi,
):

    if pd.isna(psi):
        return "Unavailable"

    if psi < 0.10:
        return "Stable"

    if psi < 0.25:
        return "Moderate Drift"

    return "High Drift"


def score_dataframe(
    dataframe,
    model,
):

    X = prepare_categories(
        dataframe[
            MODEL_FEATURES
        ],
        model,
    )

    return model.predict_proba(
        X,
        num_iteration=model.best_iteration_,
    )[:, 1]


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print("Payment Fraud Analysis")
    print("Phase 7: Model Monitoring + Drift")

    print("\n[1/7] Loading production candidate...")

    model = joblib.load(
        MODEL_PATH
    )

    con = duckdb.connect(
        str(DB_FILE)
    )

    try:

        # ====================================================
        # DAILY OPERATIONAL MONITORING
        # ====================================================

        print(
            "\n[2/7] Building daily model monitoring..."
        )

        daily_metrics = con.execute(
            f"""
            SELECT

                transaction_date,

                COUNT(*)
                    AS transactions,

                SUM(is_fraud)
                    AS fraud_transactions,

                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                )
                    AS actual_fraud_rate_pct,

                ROUND(
                    AVG(fraud_probability),
                    6
                )
                    AS avg_fraud_probability,

                ROUND(
                    100.0 * AVG(
                        CASE
                            WHEN fraud_probability >=
                                {DECISION_THRESHOLD}
                            THEN 1
                            ELSE 0
                        END
                    ),
                    4
                )
                    AS alert_rate_pct,

                SUM(
                    CASE
                        WHEN fraud_probability >=
                                {DECISION_THRESHOLD}
                         AND is_fraud = 1
                        THEN 1
                        ELSE 0
                    END
                )
                    AS true_positives,

                SUM(
                    CASE
                        WHEN fraud_probability >=
                                {DECISION_THRESHOLD}
                         AND is_fraud = 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS false_positives,

                SUM(
                    CASE
                        WHEN fraud_probability <
                                {DECISION_THRESHOLD}
                         AND is_fraud = 1
                        THEN 1
                        ELSE 0
                    END
                )
                    AS false_negatives,

                ROUND(
                    AVG(amount),
                    2
                )
                    AS avg_transaction_amount

            FROM modeling.test_scored

            GROUP BY transaction_date

            ORDER BY transaction_date
            """
        ).fetchdf()

        daily_metrics["precision"] = (
            daily_metrics[
                "true_positives"
            ]
            / (
                daily_metrics[
                    "true_positives"
                ]
                + daily_metrics[
                    "false_positives"
                ]
            )
        )

        daily_metrics["recall"] = (
            daily_metrics[
                "true_positives"
            ]
            / (
                daily_metrics[
                    "true_positives"
                ]
                + daily_metrics[
                    "false_negatives"
                ]
            )
        )

        daily_metrics.to_csv(
            REPORT_DIR
            / "daily_model_monitoring.csv",
            index=False,
        )

        print(
            daily_metrics.to_string(
                index=False
            )
        )


        # ====================================================
        # REFERENCE SAMPLE
        # ====================================================

        print(
            "\n[3/7] Loading validation reference population..."
        )

        selected_columns = list(
            dict.fromkeys(
                MODEL_FEATURES
                + ["transaction_key", "is_fraud"]
            )
        )

        reference_columns = ", ".join(
            selected_columns
        )

        reference = con.execute(
            f"""
            SELECT
                {reference_columns}

            FROM modeling.validation

            WHERE
                ABS(HASH(transaction_key)) % 5 = 0
            """
        ).fetchdf()

        print(
            f"Reference transactions: "
            f"{len(reference):,}"
        )

        reference_scores = score_dataframe(
            reference,
            model,
        )

        reference_predictions = (
            reference_scores
            >= DECISION_THRESHOLD
        ).astype(int)

        reference_precision = precision_score(
            reference[
                "is_fraud"
            ].astype(int),
            reference_predictions,
            zero_division=0,
        )

        reference_recall = recall_score(
            reference[
                "is_fraud"
            ].astype(int),
            reference_predictions,
            zero_division=0,
        )

        print(
            f"Reference precision: "
            f"{reference_precision:.4f}"
        )

        print(
            f"Reference recall: "
            f"{reference_recall:.4f}"
        )


        # ====================================================
        # DAILY FEATURE DRIFT
        # ====================================================

        print(
            "\n[4/7] Measuring daily feature drift..."
        )

        test_dates = con.execute(
            """
            SELECT DISTINCT
                transaction_date

            FROM modeling.test

            ORDER BY transaction_date
            """
        ).fetchdf()[
            "transaction_date"
        ].tolist()

        numeric_results = []
        categorical_results = []
        score_results = []

        for transaction_date in test_dates:

            print(
                f"  Checking {transaction_date}..."
            )

            current = con.execute(
                f"""
                SELECT
                    {reference_columns}

                FROM modeling.test

                WHERE
                    transaction_date = ?
                    AND ABS(HASH(transaction_key)) % 5 = 0
                """,
                [transaction_date],
            ).fetchdf()

            current_scores = score_dataframe(
                current,
                model,
            )

            # -----------------------------------------------
            # SCORE DRIFT
            # -----------------------------------------------

            score_psi = (
                calculate_numeric_psi(
                    pd.Series(
                        reference_scores
                    ),
                    pd.Series(
                        current_scores
                    ),
                )
            )

            score_results.append(
                {
                    "transaction_date":
                        transaction_date,

                    "reference_avg_score":
                        float(
                            np.mean(
                                reference_scores
                            )
                        ),

                    "current_avg_score":
                        float(
                            np.mean(
                                current_scores
                            )
                        ),

                    "score_psi":
                        score_psi,

                    "status":
                        drift_status(
                            score_psi
                        ),
                }
            )


            # -----------------------------------------------
            # NUMERIC FEATURES
            # -----------------------------------------------

            for feature in NUMERIC_FEATURES:

                psi = calculate_numeric_psi(
                    reference[
                        feature
                    ],
                    current[
                        feature
                    ],
                )

                numeric_results.append(
                    {
                        "transaction_date":
                            transaction_date,

                        "feature":
                            feature,

                        "psi":
                            psi,

                        "status":
                            drift_status(
                                psi
                            ),

                        "reference_mean":
                            pd.to_numeric(
                                reference[
                                    feature
                                ],
                                errors="coerce",
                            ).mean(),

                        "current_mean":
                            pd.to_numeric(
                                current[
                                    feature
                                ],
                                errors="coerce",
                            ).mean(),
                    }
                )


            # -----------------------------------------------
            # CATEGORICAL FEATURES
            # -----------------------------------------------

            for feature in CATEGORICAL_FEATURES:

                psi = calculate_categorical_psi(
                    reference[
                        feature
                    ],
                    current[
                        feature
                    ],
                )

                categorical_results.append(
                    {
                        "transaction_date":
                            transaction_date,

                        "feature":
                            feature,

                        "psi":
                            psi,

                        "status":
                            drift_status(
                                psi
                            ),
                    }
                )


        numeric_drift = pd.DataFrame(
            numeric_results
        )

        categorical_drift = pd.DataFrame(
            categorical_results
        )

        score_drift = pd.DataFrame(
            score_results
        )

        numeric_drift.to_csv(
            REPORT_DIR
            / "numeric_feature_drift.csv",
            index=False,
        )

        categorical_drift.to_csv(
            REPORT_DIR
            / "categorical_feature_drift.csv",
            index=False,
        )

        score_drift.to_csv(
            REPORT_DIR
            / "score_drift.csv",
            index=False,
        )


        # ====================================================
        # PERFORMANCE DRIFT
        # ====================================================

        print(
            "\n[5/7] Checking performance drift..."
        )

        performance = (
            daily_metrics[
                [
                    "transaction_date",
                    "precision",
                    "recall",
                    "actual_fraud_rate_pct",
                    "alert_rate_pct",
                ]
            ]
            .copy()
        )

        performance[
            "reference_precision"
        ] = reference_precision

        performance[
            "reference_recall"
        ] = reference_recall

        performance[
            "precision_change_pp"
        ] = (
            (
                performance[
                    "precision"
                ]
                - reference_precision
            )
            * 100
        )

        performance[
            "recall_change_pp"
        ] = (
            (
                performance[
                    "recall"
                ]
                - reference_recall
            )
            * 100
        )

        performance[
            "performance_status"
        ] = np.where(
            (
                performance[
                    "precision_change_pp"
                ] < -5
            )
            |
            (
                performance[
                    "recall_change_pp"
                ] < -5
            ),
            "Review Required",
            "Stable",
        )

        performance.to_csv(
            REPORT_DIR
            / "performance_drift.csv",
            index=False,
        )

        print(
            performance.to_string(
                index=False
            )
        )


        # ====================================================
        # ALERT GENERATION
        # ====================================================

        print(
            "\n[6/7] Creating monitoring alerts..."
        )

        alerts = []


        for _, row in score_drift.iterrows():

            if row["psi"] if "psi" in row else False:
                pass

        for _, row in score_drift.iterrows():

            if row[
                "score_psi"
            ] >= 0.10:

                alerts.append(
                    {
                        "transaction_date":
                            row[
                                "transaction_date"
                            ],

                        "alert_type":
                            "Score Drift",

                        "item":
                            "fraud_probability",

                        "severity":
                            (
                                "High"
                                if row[
                                    "score_psi"
                                ] >= 0.25
                                else "Moderate"
                            ),

                        "value":
                            row[
                                "score_psi"
                            ],

                        "message":
                            "Model score distribution shifted from reference population.",
                    }
                )


        for _, row in numeric_drift.iterrows():

            if row["psi"] >= 0.10:

                alerts.append(
                    {
                        "transaction_date":
                            row[
                                "transaction_date"
                            ],

                        "alert_type":
                            "Numeric Feature Drift",

                        "item":
                            row[
                                "feature"
                            ],

                        "severity":
                            (
                                "High"
                                if row[
                                    "psi"
                                ] >= 0.25
                                else "Moderate"
                            ),

                        "value":
                            row[
                                "psi"
                            ],

                        "message":
                            "Numeric feature distribution shifted from reference population.",
                    }
                )


        for _, row in categorical_drift.iterrows():

            if row["psi"] >= 0.10:

                alerts.append(
                    {
                        "transaction_date":
                            row[
                                "transaction_date"
                            ],

                        "alert_type":
                            "Categorical Feature Drift",

                        "item":
                            row[
                                "feature"
                            ],

                        "severity":
                            (
                                "High"
                                if row[
                                    "psi"
                                ] >= 0.25
                                else "Moderate"
                            ),

                        "value":
                            row[
                                "psi"
                            ],

                        "message":
                            "Categorical feature distribution shifted from reference population.",
                    }
                )


        for _, row in performance.iterrows():

            if (
                row[
                    "performance_status"
                ]
                == "Review Required"
            ):

                alerts.append(
                    {
                        "transaction_date":
                            row[
                                "transaction_date"
                            ],

                        "alert_type":
                            "Model Performance",

                        "item":
                            "precision_or_recall",

                        "severity":
                            "High",

                        "value":
                            min(
                                row[
                                    "precision_change_pp"
                                ],
                                row[
                                    "recall_change_pp"
                                ],
                            ),

                        "message":
                            "Precision or recall declined by more than 5 percentage points.",
                    }
                )


        alerts_df = pd.DataFrame(
            alerts,
            columns=[
                "transaction_date",
                "alert_type",
                "item",
                "severity",
                "value",
                "message",
            ],
        )

        alerts_df.to_csv(
            REPORT_DIR
            / "monitoring_alerts.csv",
            index=False,
        )

        if alerts_df.empty:

            print(
                "No monitoring alerts triggered."
            )

        else:

            print(
                alerts_df.to_string(
                    index=False
                )
            )


        # ====================================================
        # EXECUTIVE MONITORING SUMMARY
        # ====================================================

        print(
            "\n[7/7] Creating monitoring summary..."
        )

        high_numeric = (
            numeric_drift[
                numeric_drift[
                    "status"
                ]
                == "High Drift"
            ]
        )

        high_categorical = (
            categorical_drift[
                categorical_drift[
                    "status"
                ]
                == "High Drift"
            ]
        )

        high_score = (
            score_drift[
                score_drift[
                    "status"
                ]
                == "High Drift"
            ]
        )

        summary = pd.DataFrame(
            [
                {
                    "monitoring_days":
                        len(
                            test_dates
                        ),

                    "reference_period":
                        "2024-10-21 to 2024-10-25",

                    "monitoring_period":
                        "2024-10-26 to 2024-10-30",

                    "decision_threshold":
                        DECISION_THRESHOLD,

                    "reference_precision":
                        reference_precision,

                    "reference_recall":
                        reference_recall,

                    "high_numeric_drift_events":
                        len(
                            high_numeric
                        ),

                    "high_categorical_drift_events":
                        len(
                            high_categorical
                        ),

                    "high_score_drift_events":
                        len(
                            high_score
                        ),

                    "total_monitoring_alerts":
                        len(
                            alerts_df
                        ),
                }
            ]
        )

        summary.to_csv(
            REPORT_DIR
            / "monitoring_summary.csv",
            index=False,
        )

        print(
            summary.T.to_string(
                header=False
            )
        )


    finally:

        con.close()


    minutes = (
        time.time()
        - start
    ) / 60


    print(
        f"\nPhase 7 complete "
        f"in {minutes:.1f} minutes."
    )


    print("\nCreated:")

    print(
        "  reports/monitoring/"
        "daily_model_monitoring.csv"
    )

    print(
        "  reports/monitoring/"
        "numeric_feature_drift.csv"
    )

    print(
        "  reports/monitoring/"
        "categorical_feature_drift.csv"
    )

    print(
        "  reports/monitoring/"
        "score_drift.csv"
    )

    print(
        "  reports/monitoring/"
        "performance_drift.csv"
    )

    print(
        "  reports/monitoring/"
        "monitoring_alerts.csv"
    )

    print(
        "  reports/monitoring/"
        "monitoring_summary.csv"
    )


    print(
        "\nNext: final project packaging, README, "
        "GitHub cleanup and resume bullets."
    )


if __name__ == "__main__":
    main()