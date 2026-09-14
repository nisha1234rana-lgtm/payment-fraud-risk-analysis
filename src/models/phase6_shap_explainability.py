from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd


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
    / "explainability"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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

FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

THRESHOLD = 0.75


FRIENDLY_NAMES = {
    "amount": "Transaction amount",
    "log_amount": "Transaction amount scale",
    "distance_from_home": "Distance from home",
    "transaction_hour": "Transaction hour",
    "weekend_transaction": "Weekend transaction",
    "card_present": "Card present",
    "customer_prior_transactions": "Prior customer transactions",
    "minutes_since_previous_transaction": "Time since previous transaction",
    "previous_customer_amount": "Previous transaction amount",
    "customer_avg_amount_prior": "Historical average amount",
    "customer_std_amount_prior": "Historical amount variation",
    "amount_vs_customer_average": "Amount vs customer average",
    "amount_zscore_vs_customer_history": "Amount anomaly score",
    "calculated_txn_count_1h": "Transactions in prior hour",
    "calculated_amount_1h": "Spend in prior hour",
    "calculated_txn_count_24h": "Transactions in prior 24 hours",
    "calculated_amount_24h": "Spend in prior 24 hours",
    "new_merchant_for_customer_flag": "New merchant for customer",
    "overnight_transaction_flag": "Overnight transaction",
    "merchant_category": "Merchant category",
    "merchant_type": "Merchant type",
    "merchant": "Merchant",
    "currency": "Currency",
    "country": "Country",
    "city_size": "City size",
    "card_type": "Card type",
    "device": "Device type",
    "channel": "Payment channel",
}


def prepare_categories(
    dataframe,
    model,
):

    for index, column in enumerate(
        CATEGORICAL_FEATURES
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
            categories=categories
        )

    return dataframe


def main():

    start = time.time()

    print("Payment Fraud Analysis")
    print("Phase 6: SHAP Explainability")

    print("\n[1/6] Loading LightGBM model...")

    model = joblib.load(
        MODEL_PATH
    )

    con = duckdb.connect(
        str(DB_FILE)
    )

    try:

        feature_sql = ", ".join(
            f"m.{feature}"
            for feature in FEATURES
        )

        # ======================================================
        # GLOBAL SAMPLE
        # ======================================================

        print(
            "\n[2/6] Loading global explanation sample..."
        )

        sample = con.execute(
            f"""
            SELECT

                m.transaction_key,

                {feature_sql},

                m.is_fraud,

                s.fraud_probability

            FROM modeling.test AS m

            INNER JOIN modeling.test_scored AS s
                ON m.transaction_key =
                   s.transaction_key

            WHERE
                ABS(HASH(m.transaction_key)) % 50 = 0
            """
        ).fetchdf()

        print(
            f"Global explanation sample: "
            f"{len(sample):,} transactions"
        )

        # ======================================================
        # CASE STUDIES
        # ======================================================

        print(
            "\n[3/6] Selecting investigation cases..."
        )

        case_query = f"""
        WITH scored AS (

            SELECT

                m.transaction_key,

                {feature_sql},

                m.is_fraud,

                s.fraud_probability

            FROM modeling.test AS m

            INNER JOIN modeling.test_scored AS s
                ON m.transaction_key =
                   s.transaction_key
        ),

        high_risk_fraud AS (

            SELECT
                *,
                'high_risk_true_positive'
                    AS case_type

            FROM scored

            WHERE
                is_fraud = 1

            ORDER BY fraud_probability DESC

            LIMIT 5
        ),

        false_positives AS (

            SELECT
                *,
                'false_positive'
                    AS case_type

            FROM scored

            WHERE
                is_fraud = 0
                AND fraud_probability >= {THRESHOLD}

            ORDER BY fraud_probability DESC

            LIMIT 5
        ),

        false_negatives AS (

            SELECT
                *,
                'false_negative'
                    AS case_type

            FROM scored

            WHERE
                is_fraud = 1
                AND fraud_probability < {THRESHOLD}

            ORDER BY fraud_probability DESC

            LIMIT 5
        ),

        low_risk_legitimate AS (

            SELECT
                *,
                'low_risk_true_negative'
                    AS case_type

            FROM scored

            WHERE
                is_fraud = 0

            ORDER BY fraud_probability ASC

            LIMIT 5
        )

        SELECT *
        FROM high_risk_fraud

        UNION ALL

        SELECT *
        FROM false_positives

        UNION ALL

        SELECT *
        FROM false_negatives

        UNION ALL

        SELECT *
        FROM low_risk_legitimate
        """

        cases = con.execute(
            case_query
        ).fetchdf()

        print(
            cases[
                [
                    "case_type",
                    "transaction_key",
                    "is_fraud",
                    "fraud_probability",
                ]
            ].to_string(
                index=False
            )
        )

    finally:

        con.close()

    # ==========================================================
    # CATEGORY PREPARATION
    # ==========================================================

    sample = prepare_categories(
        sample,
        model
    )

    cases = prepare_categories(
        cases,
        model
    )

    X_sample = sample[
        FEATURES
    ]

    X_cases = cases[
        FEATURES
    ]

    # ==========================================================
    # LIGHTGBM SHAP CONTRIBUTIONS
    # ==========================================================

    print(
        "\n[4/6] Calculating global SHAP contributions..."
    )

    sample_contributions = (
        model.booster_.predict(
            X_sample,
            pred_contrib=True,
            num_iteration=model.best_iteration_,
        )
    )

    # Last column is expected value / base score.
    sample_shap = (
        sample_contributions[
            :,
            :-1
        ]
    )

    global_importance = pd.DataFrame(
        {
            "feature":
                FEATURES,

            "friendly_name":
                [
                    FRIENDLY_NAMES.get(
                        feature,
                        feature
                    )
                    for feature in FEATURES
                ],

            "mean_absolute_shap":
                np.abs(
                    sample_shap
                ).mean(
                    axis=0
                ),
        }
    )

    global_importance = (
        global_importance
        .sort_values(
            "mean_absolute_shap",
            ascending=False
        )
    )

    global_importance.to_csv(
        REPORT_DIR /
        "global_shap_importance.csv",
        index=False
    )

    print("\nTop 15 global risk drivers:")

    print(
        global_importance
        .head(15)
        .to_string(
            index=False
        )
    )

    # ==========================================================
    # TRANSACTION-LEVEL EXPLANATIONS
    # ==========================================================

    print(
        "\n[5/6] Building transaction-level explanations..."
    )

    case_contributions = (
        model.booster_.predict(
            X_cases,
            pred_contrib=True,
            num_iteration=model.best_iteration_,
        )
    )

    case_shap = (
        case_contributions[
            :,
            :-1
        ]
    )

    base_values = (
        case_contributions[
            :,
            -1
        ]
    )

    explanation_rows = []

    summary_rows = []

    for row_index in range(
        len(cases)
    ):

        shap_values = (
            case_shap[
                row_index
            ]
        )

        top_indexes = np.argsort(
            np.abs(
                shap_values
            )
        )[::-1][:5]

        top_drivers = []

        for rank, feature_index in enumerate(
            top_indexes,
            start=1
        ):

            feature = (
                FEATURES[
                    feature_index
                ]
            )

            value = (
                X_cases.iloc[
                    row_index
                ][feature]
            )

            contribution = float(
                shap_values[
                    feature_index
                ]
            )

            direction = (
                "Higher fraud risk"
                if contribution > 0
                else "Lower fraud risk"
            )

            friendly_name = (
                FRIENDLY_NAMES.get(
                    feature,
                    feature
                )
            )

            explanation_rows.append(
                {
                    "transaction_key":
                        cases.iloc[
                            row_index
                        ][
                            "transaction_key"
                        ],

                    "case_type":
                        cases.iloc[
                            row_index
                        ][
                            "case_type"
                        ],

                    "actual_fraud":
                        int(
                            cases.iloc[
                                row_index
                            ][
                                "is_fraud"
                            ]
                        ),

                    "fraud_probability":
                        float(
                            cases.iloc[
                                row_index
                            ][
                                "fraud_probability"
                            ]
                        ),

                    "driver_rank":
                        rank,

                    "feature":
                        feature,

                    "friendly_name":
                        friendly_name,

                    "feature_value":
                        str(value),

                    "shap_contribution_log_odds":
                        contribution,

                    "direction":
                        direction,
                }
            )

            top_drivers.append(
                f"{friendly_name}: "
                f"{direction}"
            )

        probability = float(
            cases.iloc[
                row_index
            ][
                "fraud_probability"
            ]
        )

        risk_score = round(
            probability * 100
        )

        if probability < 0.30:
            risk_band = "Low"

        elif probability < 0.60:
            risk_band = "Medium"

        elif probability < 0.80:
            risk_band = "High"

        else:
            risk_band = "Critical"

        summary_rows.append(
            {
                "transaction_key":
                    cases.iloc[
                        row_index
                    ][
                        "transaction_key"
                    ],

                "case_type":
                    cases.iloc[
                        row_index
                    ][
                        "case_type"
                    ],

                "actual_fraud":
                    int(
                        cases.iloc[
                            row_index
                        ][
                            "is_fraud"
                        ]
                    ),

                "fraud_probability":
                    probability,

                "risk_score":
                    risk_score,

                "risk_band":
                    risk_band,

                "base_value_log_odds":
                    float(
                        base_values[
                            row_index
                        ]
                    ),

                "top_driver_1":
                    top_drivers[0],

                "top_driver_2":
                    top_drivers[1],

                "top_driver_3":
                    top_drivers[2],

                "top_driver_4":
                    top_drivers[3],

                "top_driver_5":
                    top_drivers[4],
            }
        )

    explanations = pd.DataFrame(
        explanation_rows
    )

    explanations.to_csv(
        REPORT_DIR /
        "transaction_shap_explanations.csv",
        index=False
    )

    summaries = pd.DataFrame(
        summary_rows
    )

    summaries.to_csv(
        REPORT_DIR /
        "transaction_explanation_summary.csv",
        index=False
    )

    print(
        summaries[
            [
                "case_type",
                "risk_score",
                "risk_band",
                "top_driver_1",
                "top_driver_2",
                "top_driver_3",
            ]
        ].to_string(
            index=False
        )
    )

    # ==========================================================
    # MODEL EXPLANATION NOTES
    # ==========================================================

    print(
        "\n[6/6] Saving explainability documentation..."
    )

    notes = """
SHAP Explainability Notes

The LightGBM model's SHAP contributions are reported in raw model
score/log-odds space.

Positive contributions push a transaction toward a higher fraud score.
Negative contributions push a transaction toward a lower fraud score.

The model intentionally excludes near-deterministic synthetic network
rules from its main feature set.

Network relationships are retained separately for fraud investigation
and rule-system benchmarking.

Global feature importance is calculated using mean absolute SHAP
contribution across a deterministic sample of the final test period.

Transaction-level explanations show the five strongest model drivers
for selected true positives, false positives, false negatives and
low-risk legitimate transactions.
"""

    with open(
        REPORT_DIR /
        "explainability_notes.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            notes.strip()
        )

    minutes = (
        time.time() - start
    ) / 60

    print(
        f"\nPhase 6 complete "
        f"in {minutes:.1f} minutes."
    )

    print("\nCreated:")

    print(
        "  reports/explainability/"
        "global_shap_importance.csv"
    )

    print(
        "  reports/explainability/"
        "transaction_shap_explanations.csv"
    )

    print(
        "  reports/explainability/"
        "transaction_explanation_summary.csv"
    )

    print(
        "  reports/explainability/"
        "explainability_notes.txt"
    )

    print(
        "\nNext: build the Streamlit fraud investigation app."
    )


if __name__ == "__main__":
    main()