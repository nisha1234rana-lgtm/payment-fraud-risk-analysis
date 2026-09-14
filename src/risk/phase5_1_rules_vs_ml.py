from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_FILE


REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "business_results"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ML_THRESHOLD = 0.75


def main():

    start = time.time()

    print("Payment Fraud Analysis")
    print("Phase 5.1: Rules vs ML vs Hybrid")
    print(f"Database: {DB_FILE}")

    con = duckdb.connect(str(DB_FILE))

    try:

        con.execute(
            "SET preserve_insertion_order = false"
        )

        # ======================================================
        # 1. TRAIN-ONLY THRESHOLDS
        # ======================================================

        print(
            "\n[1/5] Deriving behavioral rule thresholds "
            "from training data only..."
        )

        thresholds = con.execute(
            """
            SELECT

                QUANTILE_CONT(
                    distance_from_home,
                    0.95
                ) AS distance_p95,

                QUANTILE_CONT(
                    calculated_txn_count_1h,
                    0.95
                ) AS velocity_1h_p95,

                QUANTILE_CONT(
                    calculated_amount_24h,
                    0.95
                ) AS amount_24h_p95

            FROM modeling.train

            WHERE
                distance_from_home IS NOT NULL
            """
        ).fetchdf()

        distance_p95 = float(
            thresholds.iloc[0]["distance_p95"]
        )

        velocity_p95 = float(
            thresholds.iloc[0]["velocity_1h_p95"]
        )

        amount_24h_p95 = float(
            thresholds.iloc[0]["amount_24h_p95"]
        )

        print(
            f"Distance from home P95: "
            f"{distance_p95:,.2f}"
        )

        print(
            f"1-hour velocity P95: "
            f"{velocity_p95:,.2f}"
        )

        print(
            f"24-hour spend P95: "
            f"{amount_24h_p95:,.2f}"
        )

        # ======================================================
        # 2. BUILD TEST DECISION TABLE
        # ======================================================

        print(
            "\n[2/5] Building rule, ML and hybrid decisions..."
        )

        con.execute(
            f"""
            CREATE OR REPLACE TABLE
                modeling.test_decision_comparison AS

            SELECT

                n.transaction_key,
                n.transaction_ts,

                n.amount,
                n.is_fraud,

                s.fraud_probability,

                -- ==========================================
                -- TRANSPARENT BEHAVIORAL RULE SCORE
                -- ==========================================

                (
                    CASE
                        WHEN n.distance_from_home
                            >= {distance_p95}
                        THEN 1
                        ELSE 0
                    END

                    +

                    CASE
                        WHEN n.card_present = FALSE
                        THEN 1
                        ELSE 0
                    END

                    +

                    CASE
                        WHEN n.transaction_hour
                            BETWEEN 0 AND 5
                        THEN 1
                        ELSE 0
                    END

                    +

                    CASE
                        WHEN n.amount_vs_customer_average
                            >= 3
                        THEN 1
                        ELSE 0
                    END

                    +

                    CASE
                        WHEN n.calculated_txn_count_1h
                            >= {velocity_p95}
                        THEN 1
                        ELSE 0
                    END

                    +

                    CASE
                        WHEN n.calculated_amount_24h
                            >= {amount_24h_p95}
                        THEN 1
                        ELSE 0
                    END

                    +

                    CASE
                        WHEN n.new_merchant_for_customer_flag = 1
                        THEN 1
                        ELSE 0
                    END

                ) AS behavioral_rule_score,

                -- ==========================================
                -- SYNTHETIC NETWORK RULE
                -- Kept separate for transparency
                -- ==========================================

                CASE

                    WHEN
                        n.new_customer_on_existing_device_flag = 1

                        OR
                        n.new_card_on_existing_device_flag = 1

                        OR
                        n.device_shared_by_multiple_customers_flag = 1

                    THEN 1

                    ELSE 0

                END AS network_rule_flag,

                -- ==========================================
                -- ML DECISION
                -- ==========================================

                CASE
                    WHEN s.fraud_probability
                        >= {ML_THRESHOLD}
                    THEN 1
                    ELSE 0
                END AS ml_flag

            FROM analytics.network_features AS n

            INNER JOIN modeling.test_scored AS s
                ON n.transaction_key =
                   s.transaction_key

            WHERE
                n.transaction_ts >=
                    TIMESTAMP '2024-10-26 00:00:00'
            """
        )

        rows = con.execute(
            """
            SELECT COUNT(*)
            FROM modeling.test_decision_comparison
            """
        ).fetchone()[0]

        print(
            f"Decision table rows: {rows:,}"
        )

        # ======================================================
        # 3. CREATE SYSTEM FLAGS
        # ======================================================

        print(
            "\n[3/5] Creating system decisions..."
        )

        con.execute(
            """
            CREATE OR REPLACE TABLE
                modeling.test_system_decisions AS

            SELECT
                *,

                CASE
                    WHEN behavioral_rule_score >= 3
                    THEN 1
                    ELSE 0
                END AS behavioral_rules_flag,

                CASE
                    WHEN ml_flag = 1
                      OR behavioral_rule_score >= 3
                    THEN 1
                    ELSE 0
                END AS ml_behavioral_hybrid_flag,

                CASE
                    WHEN ml_flag = 1
                      OR network_rule_flag = 1
                    THEN 1
                    ELSE 0
                END AS ml_network_hybrid_flag

            FROM modeling.test_decision_comparison
            """
        )

        # ======================================================
        # 4. COMPARE SYSTEM PERFORMANCE
        # ======================================================

        print(
            "\n[4/5] Comparing Rules vs ML vs Hybrid..."
        )

        systems = [
            (
                "Behavioral rules",
                "behavioral_rules_flag"
            ),
            (
                "ML only",
                "ml_flag"
            ),
            (
                "ML + behavioral rules",
                "ml_behavioral_hybrid_flag"
            ),
            (
                "Synthetic network rules",
                "network_rule_flag"
            ),
            (
                "ML + network rules",
                "ml_network_hybrid_flag"
            ),
        ]

        results = []

        for system_name, column in systems:

            values = con.execute(
                f"""
                SELECT

                    COUNT(*) AS transactions,

                    SUM(
                        CASE
                            WHEN {column} = 1
                            THEN 1
                            ELSE 0
                        END
                    ) AS flagged,

                    SUM(
                        CASE
                            WHEN {column} = 1
                             AND is_fraud = 1
                            THEN 1
                            ELSE 0
                        END
                    ) AS tp,

                    SUM(
                        CASE
                            WHEN {column} = 1
                             AND is_fraud = 0
                            THEN 1
                            ELSE 0
                        END
                    ) AS fp,

                    SUM(
                        CASE
                            WHEN {column} = 0
                             AND is_fraud = 1
                            THEN 1
                            ELSE 0
                        END
                    ) AS fn,

                    SUM(
                        CASE
                            WHEN {column} = 0
                             AND is_fraud = 0
                            THEN 1
                            ELSE 0
                        END
                    ) AS tn,

                    SUM(
                        CASE
                            WHEN is_fraud = 1
                            THEN amount
                            ELSE 0
                        END
                    ) AS total_fraud_value,

                    SUM(
                        CASE
                            WHEN {column} = 1
                             AND is_fraud = 1
                            THEN amount
                            ELSE 0
                        END
                    ) AS fraud_value_caught

                FROM modeling.test_system_decisions
                """
            ).fetchone()

            (
                transactions,
                flagged,
                tp,
                fp,
                fn,
                tn,
                total_fraud_value,
                fraud_value_caught,
            ) = values

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0
            )

            f1 = (
                2 * precision * recall
                / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            results.append(
                {
                    "system":
                        system_name,

                    "flagged_transactions":
                        flagged,

                    "review_rate_pct":
                        100.0
                        * flagged
                        / transactions,

                    "true_positives":
                        tp,

                    "false_positives":
                        fp,

                    "false_negatives":
                        fn,

                    "true_negatives":
                        tn,

                    "precision":
                        precision,

                    "recall":
                        recall,

                    "f1":
                        f1,

                    "fraud_value_caught":
                        fraud_value_caught,

                    "fraud_value_capture_pct":
                        100.0
                        * fraud_value_caught
                        / total_fraud_value,
                }
            )

        comparison = pd.DataFrame(
            results
        )

        print(
            comparison.to_string(
                index=False
            )
        )

        comparison.to_csv(
            REPORT_DIR /
            "rules_ml_hybrid_comparison.csv",
            index=False
        )

        # ======================================================
        # 5. RULE SCORE PROFILE
        # ======================================================

        print(
            "\n[5/5] Profiling behavioral rule scores..."
        )

        score_profile = con.execute(
            """
            SELECT

                behavioral_rule_score,

                COUNT(*) AS transactions,

                SUM(is_fraud)
                    AS fraud_transactions,

                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                ) AS fraud_rate_pct,

                ROUND(
                    AVG(fraud_probability),
                    4
                ) AS avg_ml_probability

            FROM modeling.test_system_decisions

            GROUP BY behavioral_rule_score

            ORDER BY behavioral_rule_score
            """
        ).fetchdf()

        print(
            score_profile.to_string(
                index=False
            )
        )

        score_profile.to_csv(
            REPORT_DIR /
            "behavioral_rule_score_profile.csv",
            index=False
        )

    finally:

        con.close()

    minutes = (
        time.time() - start
    ) / 60

    print(
        f"\nPhase 5.1 complete "
        f"in {minutes:.1f} minutes."
    )

    print("\nCreated:")

    print(
        "  modeling.test_decision_comparison"
    )

    print(
        "  modeling.test_system_decisions"
    )

    print(
        "  reports/business_results/"
        "rules_ml_hybrid_comparison.csv"
    )

    print(
        "  reports/business_results/"
        "behavioral_rule_score_profile.csv"
    )

    print(
        "\nNext: SHAP explainability and "
        "transaction-level fraud explanations."
    )


if __name__ == "__main__":
    main()