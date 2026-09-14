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

REPORT_DIR = PROJECT_ROOT / "reports" / "model_results"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    start = time.time()

    print("Payment Fraud Analysis")
    print("Phase 4: Modeling Dataset + Chronological Split")
    print(f"Database: {DB_FILE}")

    con = duckdb.connect(str(DB_FILE))

    try:

        con.execute("SET preserve_insertion_order = false")
        con.execute("SET threads = 4")

        print("\n[1/5] Creating modeling schema...")

        con.execute(
            """
            CREATE SCHEMA IF NOT EXISTS modeling
            """
        )

        # ======================================================
        # MODEL BASE
        # ======================================================

        print("\n[2/5] Building modeling.model_base...")

        con.execute(
            """
            CREATE OR REPLACE TABLE modeling.model_base AS

            SELECT

                transaction_key,
                transaction_ts,
                transaction_date,

                CASE

                    WHEN transaction_ts <
                        TIMESTAMP '2024-10-21 00:00:00'
                    THEN 'train'

                    WHEN transaction_ts <
                        TIMESTAMP '2024-10-26 00:00:00'
                    THEN 'validation'

                    ELSE 'test'

                END AS data_split,

                -- =============================================
                -- TARGET
                -- =============================================

                is_fraud,

                -- =============================================
                -- TRANSACTION FEATURES
                -- =============================================

                amount,

                LN(1 + amount)
                    AS log_amount,

                distance_from_home,

                transaction_hour,

                CAST(
                    weekend_transaction
                    AS INTEGER
                ) AS weekend_transaction,

                CAST(
                    card_present
                    AS INTEGER
                ) AS card_present,

                -- =============================================
                -- CATEGORICAL BUSINESS FEATURES
                -- =============================================

                merchant_category,
                merchant_type,
                merchant,

                currency,
                country,
                city_size,

                card_type,
                device,
                channel,

                -- =============================================
                -- HISTORICAL CUSTOMER BEHAVIOR
                -- =============================================

                customer_prior_transactions,

                minutes_since_previous_transaction,

                previous_customer_amount,

                customer_avg_amount_prior,

                customer_std_amount_prior,

                amount_vs_customer_average,

                amount_zscore_vs_customer_history,

                calculated_txn_count_1h,
                calculated_amount_1h,

                calculated_txn_count_24h,
                calculated_amount_24h,

                new_merchant_for_customer_flag,

                overnight_transaction_flag,

                -- =============================================
                -- NETWORK FEATURES
                -- Benchmark / investigation only
                -- =============================================

                prior_device_transactions,
                prior_card_transactions,

                prior_customers_on_device,
                prior_cards_on_device,
                prior_devices_on_card,
                prior_merchants_on_device,

                new_device_for_customer_flag,
                device_changed_from_previous_flag,

                new_customer_on_existing_device_flag,
                new_card_on_existing_device_flag,

                device_shared_by_multiple_customers_flag,
                device_shared_by_multiple_cards_flag,

                card_used_across_multiple_devices_flag

            FROM analytics.network_features

            WHERE
                transaction_ts IS NOT NULL
                AND is_fraud IN (0, 1)
            """
        )

        rows = con.execute(
            """
            SELECT COUNT(*)
            FROM modeling.model_base
            """
        ).fetchone()[0]

        print(
            f"Created modeling base with {rows:,} transactions."
        )

        # ======================================================
        # SPLIT TABLES
        # ======================================================

        print("\n[3/5] Creating chronological splits...")

        con.execute(
            """
            CREATE OR REPLACE TABLE modeling.train AS

            SELECT *
            FROM modeling.model_base

            WHERE data_split = 'train'
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TABLE modeling.validation AS

            SELECT *
            FROM modeling.model_base

            WHERE data_split = 'validation'
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TABLE modeling.test AS

            SELECT *
            FROM modeling.model_base

            WHERE data_split = 'test'
            """
        )

        split_summary = con.execute(
            """
            SELECT

                data_split,

                COUNT(*) AS transactions,

                SUM(is_fraud)
                    AS fraud_transactions,

                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                ) AS fraud_rate_pct,

                MIN(transaction_ts)
                    AS first_transaction,

                MAX(transaction_ts)
                    AS last_transaction

            FROM modeling.model_base

            GROUP BY data_split

            ORDER BY

                CASE data_split
                    WHEN 'train' THEN 1
                    WHEN 'validation' THEN 2
                    WHEN 'test' THEN 3
                END
            """
        ).fetchdf()

        print(split_summary.to_string(index=False))

        split_summary.to_csv(
            REPORT_DIR / "phase4_split_summary.csv",
            index=False
        )

        # ======================================================
        # DEVELOPMENT SAMPLE
        # ======================================================

        print(
            "\n[4/5] Creating deterministic development sample..."
        )

        con.execute(
            """
            CREATE OR REPLACE TABLE modeling.train_dev_sample AS

            SELECT *

            FROM modeling.train

            WHERE
                ABS(HASH(transaction_key)) % 10 = 0
            """
        )

        dev_summary = con.execute(
            """
            SELECT

                COUNT(*) AS transactions,

                SUM(is_fraud)
                    AS fraud_transactions,

                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                ) AS fraud_rate_pct

            FROM modeling.train_dev_sample
            """
        ).fetchdf()

        print(dev_summary.T.to_string(header=False))

        # ======================================================
        # FEATURE POLICY
        # ======================================================

        print("\n[5/5] Writing feature policy...")

        feature_policy = [

            # MAIN MODEL
            ("amount", "transaction", "main_model",
             "Core transaction value"),

            ("log_amount", "transaction", "main_model",
             "Reduces extreme amount skew"),

            ("distance_from_home", "behavior", "main_model",
             "Transaction context"),

            ("transaction_hour", "time", "main_model",
             "Time-of-day behavior"),

            ("weekend_transaction", "time", "main_model",
             "Weekend behavior"),

            ("card_present", "transaction", "main_model",
             "Card-present context"),

            ("merchant_category", "categorical", "main_model",
             "Merchant category"),

            ("merchant_type", "categorical", "main_model",
             "Merchant type"),

            ("merchant", "categorical", "main_model",
             "Merchant-level pattern"),

            ("currency", "categorical", "main_model",
             "Currency context"),

            ("country", "categorical", "main_model",
             "Country context"),

            ("city_size", "categorical", "main_model",
             "Location context"),

            ("card_type", "categorical", "main_model",
             "Card type"),

            ("device", "categorical", "main_model",
             "General device category"),

            ("channel", "categorical", "main_model",
             "Payment channel"),

            ("customer_prior_transactions", "behavior", "main_model",
             "Prior customer activity"),

            ("minutes_since_previous_transaction", "velocity", "main_model",
             "Transaction timing"),

            ("previous_customer_amount", "behavior", "main_model",
             "Previous spend"),

            ("customer_avg_amount_prior", "behavior", "main_model",
             "Leakage-safe historical average"),

            ("customer_std_amount_prior", "behavior", "main_model",
             "Historical spend variation"),

            ("amount_vs_customer_average", "behavior", "main_model",
             "Deviation from historical average"),

            ("amount_zscore_vs_customer_history", "behavior", "main_model",
             "Historical amount anomaly"),

            ("calculated_txn_count_1h", "velocity", "main_model",
             "Customer transaction velocity"),

            ("calculated_amount_1h", "velocity", "main_model",
             "Customer spend velocity"),

            ("calculated_txn_count_24h", "velocity", "main_model",
             "24-hour customer activity"),

            ("calculated_amount_24h", "velocity", "main_model",
             "24-hour customer spend"),

            ("new_merchant_for_customer_flag", "behavior", "main_model",
             "Merchant novelty"),

            ("overnight_transaction_flag", "time", "main_model",
             "Overnight activity"),

            # RULE / BENCHMARK FEATURES
            ("new_device_for_customer_flag", "network",
             "rule_or_benchmark",
             "99%+ fraud relationship in synthetic data"),

            ("device_changed_from_previous_flag", "network",
             "rule_or_benchmark",
             "Very strong synthetic device signal"),

            ("new_customer_on_existing_device_flag", "network",
             "rule_or_benchmark",
             "100% fraud in observed dataset"),

            ("new_card_on_existing_device_flag", "network",
             "rule_or_benchmark",
             "Nearly 100% fraud in observed dataset"),

            ("device_shared_by_multiple_customers_flag", "network",
             "rule_or_benchmark",
             "100% fraud in observed dataset"),

            ("device_shared_by_multiple_cards_flag", "network",
             "rule_or_benchmark",
             "Strong synthetic fraud relationship"),

            ("card_used_across_multiple_devices_flag", "network",
             "rule_or_benchmark",
             "Network investigation feature"),

            ("prior_customers_on_device", "network",
             "rule_or_benchmark",
             "Historical network connectivity"),

            ("prior_cards_on_device", "network",
             "rule_or_benchmark",
             "Historical network connectivity"),

            ("prior_devices_on_card", "network",
             "rule_or_benchmark",
             "Historical network connectivity"),

            ("prior_merchants_on_device", "network",
             "rule_or_benchmark",
             "Historical network connectivity"),

            # EXCLUDED
            ("customer_id", "identifier", "excluded",
             "Would encourage customer memorization"),

            ("card_number", "identifier", "excluded",
             "Raw identifier"),

            ("device_fingerprint", "identifier", "excluded",
             "Raw high-cardinality identifier"),

            ("ip_address", "identifier", "excluded",
             "Nearly unique per transaction"),

            ("velocity_last_hour", "source_feature", "excluded",
             "Meaning could not be validated against calculated velocity"),
        ]

        policy_df = pd.DataFrame(
            feature_policy,
            columns=[
                "feature",
                "feature_group",
                "policy",
                "reason",
            ]
        )

        policy_df.to_csv(
            REPORT_DIR / "feature_policy.csv",
            index=False
        )

        print(
            policy_df.groupby("policy")
            .size()
            .to_string()
        )

    finally:

        con.close()

    minutes = (time.time() - start) / 60

    print(
        f"\nPhase 4 complete in {minutes:.1f} minutes."
    )

    print("\nCreated:")

    print("  modeling.model_base")
    print("  modeling.train")
    print("  modeling.validation")
    print("  modeling.test")
    print("  modeling.train_dev_sample")

    print(
        "  reports/model_results/"
        "phase4_split_summary.csv"
    )

    print(
        "  reports/model_results/"
        "feature_policy.csv"
    )

    print(
        "\nNext: Logistic Regression baseline "
        "using only the main-model feature set."
    )


if __name__ == "__main__":
    main()