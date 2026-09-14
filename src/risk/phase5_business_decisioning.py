from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_FILE


PREDICTION_FILE = (
    PROJECT_ROOT
    / "reports"
    / "model_results"
    / "final_test_predictions.parquet"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "business_results"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# BUSINESS ASSUMPTIONS
# ---------------------------------------------------------
#
# These values DO NOT come from the dataset.
# They are explicit project assumptions used for scenario analysis.
#
# We will later perform sensitivity analysis rather than pretending
# there is one universally correct cost structure.
# ---------------------------------------------------------

MANUAL_REVIEW_COST = 5.00

FALSE_POSITIVE_FRICTION_COST = 15.00

FRAUD_RECOVERY_RATE = 0.90


def main():

    start = time.time()

    print("Payment Fraud Analysis")
    print("Phase 5: Business Decisioning")
    print(f"Database: {DB_FILE}")

    con = duckdb.connect(str(DB_FILE))

    try:

        # ======================================================
        # 1. JOIN MODEL SCORES WITH TRANSACTION VALUE
        # ======================================================

        print(
            "\n[1/6] Joining final-test predictions "
            "with transaction values..."
        )

        con.execute(
            f"""
            CREATE OR REPLACE TABLE
                modeling.test_scored AS

            SELECT
                t.transaction_key,
                t.transaction_ts,
                t.transaction_date,

                t.amount,

                t.merchant_category,
                t.merchant,
                t.channel,
                t.country,

                t.is_fraud,

                p.fraud_probability

            FROM modeling.test AS t

            INNER JOIN read_parquet(
                '{PREDICTION_FILE.as_posix()}'
            ) AS p
                ON t.transaction_key =
                   p.transaction_key
            """
        )

        scored_rows = con.execute(
            """
            SELECT COUNT(*)
            FROM modeling.test_scored
            """
        ).fetchone()[0]

        print(
            f"Scored transactions joined: "
            f"{scored_rows:,}"
        )

        # ======================================================
        # 2. TEST-SET ECONOMIC BASELINE
        # ======================================================

        print(
            "\n[2/6] Measuring fraud-value baseline..."
        )

        baseline = con.execute(
            """
            SELECT

                COUNT(*) AS transactions,

                SUM(is_fraud)
                    AS fraud_transactions,

                ROUND(
                    SUM(amount),
                    2
                ) AS transaction_value,

                ROUND(
                    SUM(
                        CASE
                            WHEN is_fraud = 1
                            THEN amount
                            ELSE 0
                        END
                    ),
                    2
                ) AS fraud_value,

                ROUND(
                    AVG(
                        CASE
                            WHEN is_fraud = 1
                            THEN amount
                        END
                    ),
                    2
                ) AS avg_fraud_amount

            FROM modeling.test_scored
            """
        ).fetchdf()

        print(
            baseline.T.to_string(
                header=False
            )
        )

        baseline.to_csv(
            REPORT_DIR /
            "test_fraud_value_baseline.csv",
            index=False
        )

        # ======================================================
        # 3. THRESHOLD ECONOMICS
        # ======================================================

        print(
            "\n[3/6] Evaluating threshold economics..."
        )

        thresholds = np.arange(
            0.10,
            0.96,
            0.05
        )

        threshold_results = []

        for threshold in thresholds:

            result = con.execute(
                """
                SELECT

                    COUNT(*) AS transactions,

                    SUM(
                        CASE
                            WHEN fraud_probability >= ?
                            THEN 1
                            ELSE 0
                        END
                    ) AS flagged_transactions,

                    SUM(
                        CASE
                            WHEN fraud_probability >= ?
                             AND is_fraud = 1
                            THEN 1
                            ELSE 0
                        END
                    ) AS fraud_caught,

                    SUM(
                        CASE
                            WHEN fraud_probability >= ?
                             AND is_fraud = 0
                            THEN 1
                            ELSE 0
                        END
                    ) AS false_alerts,

                    SUM(
                        CASE
                            WHEN fraud_probability < ?
                             AND is_fraud = 1
                            THEN 1
                            ELSE 0
                        END
                    ) AS fraud_missed,

                    SUM(
                        CASE
                            WHEN is_fraud = 1
                            THEN amount
                            ELSE 0
                        END
                    ) AS total_fraud_value,

                    SUM(
                        CASE
                            WHEN fraud_probability >= ?
                             AND is_fraud = 1
                            THEN amount
                            ELSE 0
                        END
                    ) AS fraud_value_caught,

                    SUM(
                        CASE
                            WHEN fraud_probability < ?
                             AND is_fraud = 1
                            THEN amount
                            ELSE 0
                        END
                    ) AS fraud_value_missed

                FROM modeling.test_scored
                """,
                [
                    threshold,
                    threshold,
                    threshold,
                    threshold,
                    threshold,
                    threshold,
                ]
            ).fetchone()

            (
                transactions,
                flagged,
                fraud_caught,
                false_alerts,
                fraud_missed,
                total_fraud_value,
                fraud_value_caught,
                fraud_value_missed,
            ) = result

            review_cost = (
                flagged *
                MANUAL_REVIEW_COST
            )

            friction_cost = (
                false_alerts *
                FALSE_POSITIVE_FRICTION_COST
            )

            estimated_prevented_loss = (
                fraud_value_caught *
                FRAUD_RECOVERY_RATE
            )

            remaining_fraud_loss = (
                total_fraud_value -
                estimated_prevented_loss
            )

            estimated_total_cost = (
                remaining_fraud_loss +
                review_cost +
                friction_cost
            )

            threshold_results.append(
                {
                    "threshold":
                        round(
                            float(threshold),
                            2
                        ),

                    "flagged_transactions":
                        flagged,

                    "review_rate_pct":
                        100.0 *
                        flagged /
                        transactions,

                    "fraud_caught":
                        fraud_caught,

                    "fraud_missed":
                        fraud_missed,

                    "false_alerts":
                        false_alerts,

                    "transaction_recall_pct":
                        100.0 *
                        fraud_caught /
                        (
                            fraud_caught +
                            fraud_missed
                        ),

                    "total_fraud_value":
                        total_fraud_value,

                    "fraud_value_caught":
                        fraud_value_caught,

                    "fraud_value_capture_pct":
                        100.0 *
                        fraud_value_caught /
                        total_fraud_value,

                    "fraud_value_missed":
                        fraud_value_missed,

                    "estimated_prevented_loss":
                        estimated_prevented_loss,

                    "manual_review_cost":
                        review_cost,

                    "false_positive_friction_cost":
                        friction_cost,

                    "estimated_total_cost":
                        estimated_total_cost,
                }
            )

        threshold_df = pd.DataFrame(
            threshold_results
        )

        threshold_df.to_csv(
            REPORT_DIR /
            "threshold_business_analysis.csv",
            index=False
        )

        best_cost_row = threshold_df.loc[
            threshold_df[
                "estimated_total_cost"
            ].idxmin()
        ]

        print(
            "\nLowest estimated-cost threshold:"
        )

        print(
            best_cost_row.to_string()
        )

        # ======================================================
        # 4. REVIEW CAPACITY ANALYSIS
        # ======================================================

        print(
            "\n[4/6] Simulating analyst review capacity..."
        )

        daily_capacities = [
            500,
            1000,
            2000,
            5000,
            10000,
        ]

        capacity_results = []

        dates = con.execute(
            """
            SELECT DISTINCT transaction_date

            FROM modeling.test_scored

            ORDER BY transaction_date
            """
        ).fetchdf()[
            "transaction_date"
        ].tolist()

        for capacity in daily_capacities:

            daily_rows = []

            for transaction_date in dates:

                daily_result = con.execute(
                    """
                    WITH ranked AS (

                        SELECT
                            *,

                            ROW_NUMBER() OVER (
                                ORDER BY
                                    fraud_probability DESC,
                                    transaction_key
                            ) AS risk_rank

                        FROM modeling.test_scored

                        WHERE transaction_date = ?
                    )

                    SELECT

                        COUNT(*) AS reviewed,

                        SUM(is_fraud)
                            AS fraud_caught,

                        SUM(
                            CASE
                                WHEN is_fraud = 0
                                THEN 1
                                ELSE 0
                            END
                        ) AS false_alerts,

                        SUM(
                            CASE
                                WHEN is_fraud = 1
                                THEN amount
                                ELSE 0
                            END
                        ) AS fraud_value_caught

                    FROM ranked

                    WHERE risk_rank <= ?
                    """,
                    [
                        transaction_date,
                        capacity
                    ]
                ).fetchone()

                reviewed = (
                    daily_result[0] or 0
                )

                fraud_caught = (
                    daily_result[1] or 0
                )

                false_alerts = (
                    daily_result[2] or 0
                )

                fraud_value_caught = (
                    daily_result[3] or 0
                )

                daily_rows.append(
                    {
                        "reviewed":
                            reviewed,

                        "fraud_caught":
                            fraud_caught,

                        "false_alerts":
                            false_alerts,

                        "fraud_value_caught":
                            fraud_value_caught,
                    }
                )

            totals = pd.DataFrame(
                daily_rows
            ).sum()

            total_fraud = baseline.iloc[
                0
            ]["fraud_transactions"]

            total_fraud_value = baseline.iloc[
                0
            ]["fraud_value"]

            capacity_results.append(
                {
                    "daily_review_capacity":
                        capacity,

                    "days":
                        len(dates),

                    "total_reviews":
                        int(
                            totals[
                                "reviewed"
                            ]
                        ),

                    "fraud_transactions_caught":
                        int(
                            totals[
                                "fraud_caught"
                            ]
                        ),

                    "false_alerts":
                        int(
                            totals[
                                "false_alerts"
                            ]
                        ),

                    "fraud_recall_pct":
                        100.0 *
                        totals[
                            "fraud_caught"
                        ] /
                        total_fraud,

                    "fraud_value_caught":
                        totals[
                            "fraud_value_caught"
                        ],

                    "fraud_value_capture_pct":
                        100.0 *
                        totals[
                            "fraud_value_caught"
                        ] /
                        total_fraud_value,

                    "fraud_per_100_reviews":
                        100.0 *
                        totals[
                            "fraud_caught"
                        ] /
                        totals[
                            "reviewed"
                        ],
                }
            )

        capacity_df = pd.DataFrame(
            capacity_results
        )

        print(
            capacity_df.to_string(
                index=False
            )
        )

        capacity_df.to_csv(
            REPORT_DIR /
            "manual_review_capacity.csv",
            index=False
        )

        # ======================================================
        # 5. RISK BANDS
        # ======================================================

        print(
            "\n[5/6] Building operational risk bands..."
        )

        risk_bands = con.execute(
            """
            SELECT

                CASE

                    WHEN fraud_probability < 0.30
                    THEN 'Low'

                    WHEN fraud_probability < 0.60
                    THEN 'Medium'

                    WHEN fraud_probability < 0.80
                    THEN 'High'

                    ELSE 'Critical'

                END AS risk_band,

                COUNT(*) AS transactions,

                SUM(is_fraud)
                    AS fraud_transactions,

                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                ) AS actual_fraud_rate_pct,

                ROUND(
                    AVG(
                        fraud_probability
                    ),
                    4
                ) AS avg_model_probability,

                ROUND(
                    SUM(amount),
                    2
                ) AS transaction_value,

                ROUND(
                    SUM(
                        CASE
                            WHEN is_fraud = 1
                            THEN amount
                            ELSE 0
                        END
                    ),
                    2
                ) AS fraud_value

            FROM modeling.test_scored

            GROUP BY risk_band

            ORDER BY

                CASE risk_band
                    WHEN 'Low' THEN 1
                    WHEN 'Medium' THEN 2
                    WHEN 'High' THEN 3
                    WHEN 'Critical' THEN 4
                END
            """
        ).fetchdf()

        print(
            risk_bands.to_string(
                index=False
            )
        )

        risk_bands.to_csv(
            REPORT_DIR /
            "risk_band_analysis.csv",
            index=False
        )

        # ======================================================
        # 6. WRITE BUSINESS ASSUMPTIONS
        # ======================================================

        print(
            "\n[6/6] Saving business assumptions..."
        )

        assumptions = pd.DataFrame(
            [
                {
                    "assumption":
                        "manual_review_cost",

                    "value":
                        MANUAL_REVIEW_COST,

                    "unit":
                        "USD per flagged transaction",

                    "source":
                        "Project scenario assumption",
                },

                {
                    "assumption":
                        "false_positive_friction_cost",

                    "value":
                        FALSE_POSITIVE_FRICTION_COST,

                    "unit":
                        "USD per legitimate flagged transaction",

                    "source":
                        "Project scenario assumption",
                },

                {
                    "assumption":
                        "fraud_recovery_rate",

                    "value":
                        FRAUD_RECOVERY_RATE,

                    "unit":
                        "share of flagged fraud value prevented",

                    "source":
                        "Project scenario assumption",
                },
            ]
        )

        assumptions.to_csv(
            REPORT_DIR /
            "business_cost_assumptions.csv",
            index=False
        )

    finally:

        con.close()

    minutes = (
        time.time() - start
    ) / 60

    print(
        f"\nPhase 5 complete "
        f"in {minutes:.1f} minutes."
    )

    print("\nCreated:")

    print(
        "  modeling.test_scored"
    )

    print(
        "  reports/business_results/"
        "test_fraud_value_baseline.csv"
    )

    print(
        "  reports/business_results/"
        "threshold_business_analysis.csv"
    )

    print(
        "  reports/business_results/"
        "manual_review_capacity.csv"
    )

    print(
        "  reports/business_results/"
        "risk_band_analysis.csv"
    )

    print(
        "  reports/business_results/"
        "business_cost_assumptions.csv"
    )

    print(
        "\nNext: rule-based system vs ML vs hybrid comparison."
    )


if __name__ == "__main__":
    main()