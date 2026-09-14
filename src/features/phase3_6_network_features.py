from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_FILE

REPORT_DIR = PROJECT_ROOT / "reports" / "analysis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    start = time.time()

    print("Payment Fraud Analysis")
    print("Phase 3.6: Historical Network Features")
    print(f"Database: {DB_FILE}")

    con = duckdb.connect(str(DB_FILE))

    try:

        con.execute("SET preserve_insertion_order = false")
        con.execute("SET threads = 4")

        # ==========================================================
        # 1. BUILD RELATIONSHIP SEQUENCES
        # ==========================================================

        print("\n[1/4] Building card-device-customer relationship history...")

        con.execute(
            """
            CREATE OR REPLACE TABLE analytics.network_features AS

            WITH edge_sequences AS (

                SELECT
                    *,

                    ROW_NUMBER() OVER (
                        PARTITION BY
                            device_fingerprint,
                            customer_id
                        ORDER BY
                            transaction_ts,
                            transaction_key
                    ) AS device_customer_sequence,

                    ROW_NUMBER() OVER (
                        PARTITION BY
                            device_fingerprint,
                            card_number
                        ORDER BY
                            transaction_ts,
                            transaction_key
                    ) AS device_card_sequence,

                    ROW_NUMBER() OVER (
                        PARTITION BY
                            device_fingerprint,
                            merchant
                        ORDER BY
                            transaction_ts,
                            transaction_key
                    ) AS device_merchant_sequence,

                    ROW_NUMBER() OVER (
                        PARTITION BY device_fingerprint
                        ORDER BY
                            transaction_ts,
                            transaction_key
                    ) - 1 AS prior_device_transactions,

                    ROW_NUMBER() OVER (
                        PARTITION BY card_number
                        ORDER BY
                            transaction_ts,
                            transaction_key
                    ) - 1 AS prior_card_transactions

                FROM analytics.behavioral_features
            ),

            relationship_history AS (

                SELECT
                    *,

                    COALESCE(

                        SUM(
                            CASE
                                WHEN device_customer_sequence = 1
                                THEN 1
                                ELSE 0
                            END
                        ) OVER (
                            PARTITION BY device_fingerprint

                            ORDER BY
                                transaction_ts,
                                transaction_key

                            ROWS BETWEEN
                                UNBOUNDED PRECEDING
                                AND 1 PRECEDING
                        ),

                        0

                    ) AS prior_customers_on_device,

                    COALESCE(

                        SUM(
                            CASE
                                WHEN device_card_sequence = 1
                                THEN 1
                                ELSE 0
                            END
                        ) OVER (
                            PARTITION BY device_fingerprint

                            ORDER BY
                                transaction_ts,
                                transaction_key

                            ROWS BETWEEN
                                UNBOUNDED PRECEDING
                                AND 1 PRECEDING
                        ),

                        0

                    ) AS prior_cards_on_device,

                    COALESCE(

                        SUM(
                            CASE
                                WHEN device_card_sequence = 1
                                THEN 1
                                ELSE 0
                            END
                        ) OVER (
                            PARTITION BY card_number

                            ORDER BY
                                transaction_ts,
                                transaction_key

                            ROWS BETWEEN
                                UNBOUNDED PRECEDING
                                AND 1 PRECEDING
                        ),

                        0

                    ) AS prior_devices_on_card,

                    COALESCE(

                        SUM(
                            CASE
                                WHEN device_merchant_sequence = 1
                                THEN 1
                                ELSE 0
                            END
                        ) OVER (
                            PARTITION BY device_fingerprint

                            ORDER BY
                                transaction_ts,
                                transaction_key

                            ROWS BETWEEN
                                UNBOUNDED PRECEDING
                                AND 1 PRECEDING
                        ),

                        0

                    ) AS prior_merchants_on_device

                FROM edge_sequences
            )

            SELECT
                *,

                CASE
                    WHEN device_customer_sequence = 1
                     AND prior_device_transactions > 0
                    THEN 1
                    ELSE 0
                END AS new_customer_on_existing_device_flag,

                CASE
                    WHEN device_card_sequence = 1
                     AND prior_device_transactions > 0
                    THEN 1
                    ELSE 0
                END AS new_card_on_existing_device_flag,

                CASE
                    WHEN prior_customers_on_device >= 2
                    THEN 1
                    ELSE 0
                END AS device_shared_by_multiple_customers_flag,

                CASE
                    WHEN prior_cards_on_device >= 2
                    THEN 1
                    ELSE 0
                END AS device_shared_by_multiple_cards_flag,

                CASE
                    WHEN prior_devices_on_card >= 2
                    THEN 1
                    ELSE 0
                END AS card_used_across_multiple_devices_flag

            FROM relationship_history
            """
        )

        row_count = con.execute(
            """
            SELECT COUNT(*)
            FROM analytics.network_features
            """
        ).fetchone()[0]

        print(
            f"Created historical network features for "
            f"{row_count:,} transactions."
        )

        # ==========================================================
        # 2. VALIDATION
        # ==========================================================

        print("\n[2/4] Validating network features...")

        validation = con.execute(
            """
            SELECT

                COUNT(*) AS transactions,

                COUNT(DISTINCT transaction_key)
                    AS unique_transaction_keys,

                MAX(prior_customers_on_device)
                    AS max_prior_customers_on_device,

                MAX(prior_cards_on_device)
                    AS max_prior_cards_on_device,

                MAX(prior_devices_on_card)
                    AS max_prior_devices_on_card,

                MAX(prior_merchants_on_device)
                    AS max_prior_merchants_on_device,

                ROUND(
                    100.0 *
                    AVG(new_customer_on_existing_device_flag),
                    4
                ) AS new_customer_existing_device_pct,

                ROUND(
                    100.0 *
                    AVG(new_card_on_existing_device_flag),
                    4
                ) AS new_card_existing_device_pct,

                ROUND(
                    100.0 *
                    AVG(device_shared_by_multiple_customers_flag),
                    4
                ) AS multi_customer_device_pct,

                ROUND(
                    100.0 *
                    AVG(device_shared_by_multiple_cards_flag),
                    4
                ) AS multi_card_device_pct

            FROM analytics.network_features
            """
        ).fetchdf()

        print(validation.T.to_string(header=False))

        validation.to_csv(
            REPORT_DIR / "network_feature_summary.csv",
            index=False
        )

        # ==========================================================
        # 3. FRAUD RATE BY NETWORK SIGNAL
        # ==========================================================

        print("\n[3/4] Measuring fraud rates for network signals...")

        signals = con.execute(
            """
            SELECT
                'new_customer_existing_device'
                    AS signal,

                new_customer_on_existing_device_flag
                    AS signal_value,

                COUNT(*) AS transactions,

                SUM(is_fraud) AS fraud_transactions,

                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                ) AS fraud_rate_pct

            FROM analytics.network_features

            GROUP BY
                new_customer_on_existing_device_flag


            UNION ALL


            SELECT
                'new_card_existing_device',

                new_card_on_existing_device_flag,

                COUNT(*),

                SUM(is_fraud),

                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                )

            FROM analytics.network_features

            GROUP BY
                new_card_on_existing_device_flag


            UNION ALL


            SELECT
                'device_multiple_customers',

                device_shared_by_multiple_customers_flag,

                COUNT(*),

                SUM(is_fraud),

                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                )

            FROM analytics.network_features

            GROUP BY
                device_shared_by_multiple_customers_flag


            UNION ALL


            SELECT
                'device_multiple_cards',

                device_shared_by_multiple_cards_flag,

                COUNT(*),

                SUM(is_fraud),

                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                )

            FROM analytics.network_features

            GROUP BY
                device_shared_by_multiple_cards_flag


            UNION ALL


            SELECT
                'card_multiple_devices',

                card_used_across_multiple_devices_flag,

                COUNT(*),

                SUM(is_fraud),

                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                )

            FROM analytics.network_features

            GROUP BY
                card_used_across_multiple_devices_flag


            ORDER BY
                signal,
                signal_value
            """
        ).fetchdf()

        print(signals.to_string(index=False))

        signals.to_csv(
            REPORT_DIR / "network_signal_fraud_rates.csv",
            index=False
        )

        # ==========================================================
        # 4. INVESTIGATION EDGE TABLES
        # ==========================================================

        print("\n[4/4] Building fraud-investigation network tables...")

        # These full-period aggregates are for analyst investigation
        # and visualization, NOT predictive model features.

        con.execute(
            """
            CREATE OR REPLACE TABLE
                analytics.device_card_edges AS

            SELECT
                device_fingerprint,
                card_number,

                COUNT(*) AS transactions,

                COUNT(DISTINCT customer_id)
                    AS customers,

                MIN(transaction_ts)
                    AS first_seen,

                MAX(transaction_ts)
                    AS last_seen,

                SUM(is_fraud)
                    AS fraud_transactions,

                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                ) AS fraud_rate_pct

            FROM analytics.network_features

            GROUP BY
                device_fingerprint,
                card_number
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TABLE
                analytics.device_customer_edges AS

            SELECT
                device_fingerprint,
                customer_id,

                COUNT(*) AS transactions,

                COUNT(DISTINCT card_number)
                    AS cards,

                MIN(transaction_ts)
                    AS first_seen,

                MAX(transaction_ts)
                    AS last_seen,

                SUM(is_fraud)
                    AS fraud_transactions,

                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                ) AS fraud_rate_pct

            FROM analytics.network_features

            GROUP BY
                device_fingerprint,
                customer_id
            """
        )

        suspicious_devices = con.execute(
            """
            SELECT
                device_fingerprint,

                COUNT(*) AS transactions,

                COUNT(DISTINCT customer_id)
                    AS customers,

                COUNT(DISTINCT card_number)
                    AS cards,

                COUNT(DISTINCT merchant)
                    AS merchants,

                SUM(is_fraud)
                    AS fraud_transactions,

                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                ) AS fraud_rate_pct

            FROM analytics.network_features

            GROUP BY device_fingerprint

            HAVING COUNT(*) >= 2

            ORDER BY
                customers DESC,
                cards DESC,
                fraud_transactions DESC

            LIMIT 500
            """
        ).fetchdf()

        suspicious_devices.to_csv(
            REPORT_DIR / "suspicious_device_networks.csv",
            index=False
        )

        print(
            f"Saved {len(suspicious_devices):,} "
            "high-connectivity device records."
        )

    finally:

        con.close()

    minutes = (time.time() - start) / 60

    print(
        f"\nPhase 3.6 complete in {minutes:.1f} minutes."
    )

    print("\nCreated:")
    print("  analytics.network_features")
    print("  analytics.device_card_edges")
    print("  analytics.device_customer_edges")
    print(
        "  reports/analysis/network_feature_summary.csv"
    )
    print(
        "  reports/analysis/network_signal_fraud_rates.csv"
    )
    print(
        "  reports/analysis/suspicious_device_networks.csv"
    )

    print(
        "\nNext: prepare the modeling dataset and "
        "chronological train/validation/test split."
    )


if __name__ == "__main__":
    main()