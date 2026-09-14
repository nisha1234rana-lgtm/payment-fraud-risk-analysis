from __future__ import annotations

import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_FILE


def main():

    print("Payment Fraud Analysis")
    print("Phase 2.7: Create Unique Transaction Keys")
    print(f"Database: {DB_FILE}")

    con = duckdb.connect(str(DB_FILE))

    try:

        print("\n[1/3] Creating analysis-ready transaction table...")

        con.execute(
            """
            CREATE OR REPLACE TABLE analytics.transactions_final AS

            WITH numbered AS (

                SELECT
                    *,

                    COUNT(*) OVER (
                        PARTITION BY transaction_id
                    ) AS source_id_occurrences,

                    ROW_NUMBER() OVER (
                        PARTITION BY transaction_id

                        ORDER BY
                            transaction_ts,
                            customer_id,
                            card_number,
                            amount,
                            merchant,
                            device_fingerprint,
                            ip_address
                    ) AS source_id_sequence

                FROM analytics.transactions_clean
            )

            SELECT
                transaction_id AS source_transaction_id,

                CASE

                    WHEN source_id_occurrences = 1
                        THEN transaction_id

                    ELSE
                        transaction_id
                        || '_'
                        || CAST(source_id_sequence AS VARCHAR)

                END AS transaction_key,

                CASE
                    WHEN source_id_occurrences > 1
                    THEN 1
                    ELSE 0
                END AS reused_source_id_flag,

                source_id_occurrences,

                customer_id,
                card_number,
                transaction_ts,
                transaction_date,
                derived_hour,
                day_of_week,

                merchant_category,
                merchant_type,
                merchant,

                amount,
                currency,

                country,
                city,
                city_size,

                card_type,
                card_present,

                device,
                channel,
                device_fingerprint,
                ip_address,

                distance_from_home,
                high_risk_merchant,
                transaction_hour,
                weekend_transaction,

                velocity_last_hour_raw,

                is_fraud,

                * EXCLUDE (
                    transaction_id,
                    customer_id,
                    card_number,
                    transaction_ts,
                    transaction_date,
                    derived_hour,
                    day_of_week,
                    merchant_category,
                    merchant_type,
                    merchant,
                    amount,
                    currency,
                    country,
                    city,
                    city_size,
                    card_type,
                    card_present,
                    device,
                    channel,
                    device_fingerprint,
                    ip_address,
                    distance_from_home,
                    high_risk_merchant,
                    transaction_hour,
                    weekend_transaction,
                    velocity_last_hour_raw,
                    is_fraud,
                    source_id_occurrences,
                    source_id_sequence
                )

            FROM numbered
            """
        )

        print("\n[2/3] Validating transaction keys...")

        validation = con.execute(
            """
            SELECT
                COUNT(*) AS total_rows,

                COUNT(DISTINCT transaction_key)
                    AS unique_transaction_keys,

                COUNT(DISTINCT source_transaction_id)
                    AS unique_source_ids,

                SUM(reused_source_id_flag)
                    AS rows_with_reused_source_ids,

                COUNT(*) -
                COUNT(DISTINCT transaction_key)
                    AS duplicate_transaction_keys

            FROM analytics.transactions_final
            """
        ).fetchdf()

        print(validation.T.to_string(header=False))

        print("\n[3/3] Checking final fraud population...")

        fraud_summary = con.execute(
            """
            SELECT
                COUNT(*) AS transactions,

                SUM(is_fraud) AS fraud_transactions,

                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                ) AS fraud_rate_pct,

                MIN(transaction_ts) AS start_date,

                MAX(transaction_ts) AS end_date

            FROM analytics.transactions_final
            """
        ).fetchdf()

        print(fraud_summary.T.to_string(header=False))

    finally:

        con.close()

    print("\nPhase 2.7 complete.")
    print("analytics.transactions_final is now our official analysis table.")


if __name__ == "__main__":
    main()