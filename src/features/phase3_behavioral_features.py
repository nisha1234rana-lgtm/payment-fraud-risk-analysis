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
    print("Phase 3: Behavioral Transaction Features")
    print(f"Database: {DB_FILE}")

    con = duckdb.connect(str(DB_FILE))

    try:

        con.execute("SET preserve_insertion_order = false")
        con.execute("SET threads = 4")

        print("\n[1/4] Building historical customer behavior...")

        con.execute(
            """
            CREATE OR REPLACE TABLE analytics.behavioral_features AS

            WITH historical AS (

                SELECT
                    *,

                    ROW_NUMBER() OVER (
                        PARTITION BY customer_id
                        ORDER BY transaction_ts, transaction_key
                    ) - 1
                        AS customer_prior_transactions,

                    LAG(transaction_ts) OVER (
                        PARTITION BY customer_id
                        ORDER BY transaction_ts, transaction_key
                    )
                        AS previous_customer_transaction_ts,

                    LAG(amount) OVER (
                        PARTITION BY customer_id
                        ORDER BY transaction_ts, transaction_key
                    )
                        AS previous_customer_amount,

                    LAG(merchant) OVER (
                        PARTITION BY customer_id
                        ORDER BY transaction_ts, transaction_key
                    )
                        AS previous_customer_merchant,

                    LAG(device_fingerprint) OVER (
                        PARTITION BY customer_id
                        ORDER BY transaction_ts, transaction_key
                    )
                        AS previous_device_fingerprint,

                    AVG(amount) OVER (
                        PARTITION BY customer_id
                        ORDER BY transaction_ts, transaction_key
                        ROWS BETWEEN UNBOUNDED PRECEDING
                        AND 1 PRECEDING
                    )
                        AS customer_avg_amount_prior,

                    STDDEV_SAMP(amount) OVER (
                        PARTITION BY customer_id
                        ORDER BY transaction_ts, transaction_key
                        ROWS BETWEEN UNBOUNDED PRECEDING
                        AND 1 PRECEDING
                    )
                        AS customer_std_amount_prior,

                    ROW_NUMBER() OVER (
                        PARTITION BY
                            customer_id,
                            device_fingerprint

                        ORDER BY
                            transaction_ts,
                            transaction_key
                    )
                        AS customer_device_sequence,

                    ROW_NUMBER() OVER (
                        PARTITION BY
                            customer_id,
                            ip_address

                        ORDER BY
                            transaction_ts,
                            transaction_key
                    )
                        AS customer_ip_sequence,

                    ROW_NUMBER() OVER (
                        PARTITION BY
                            customer_id,
                            merchant

                        ORDER BY
                            transaction_ts,
                            transaction_key
                    )
                        AS customer_merchant_sequence

                FROM analytics.transactions_final
            ),

            velocity AS (

                SELECT
                    *,

                    COUNT(*) OVER (
                        PARTITION BY customer_id

                        ORDER BY transaction_ts

                        RANGE BETWEEN
                            INTERVAL '1 hour' PRECEDING
                            AND CURRENT ROW
                    ) - 1
                        AS calculated_txn_count_1h,

                    SUM(amount) OVER (
                        PARTITION BY customer_id

                        ORDER BY transaction_ts

                        RANGE BETWEEN
                            INTERVAL '1 hour' PRECEDING
                            AND CURRENT ROW
                    ) - amount
                        AS calculated_amount_1h,

                    COUNT(*) OVER (
                        PARTITION BY customer_id

                        ORDER BY transaction_ts

                        RANGE BETWEEN
                            INTERVAL '24 hours' PRECEDING
                            AND CURRENT ROW
                    ) - 1
                        AS calculated_txn_count_24h,

                    SUM(amount) OVER (
                        PARTITION BY customer_id

                        ORDER BY transaction_ts

                        RANGE BETWEEN
                            INTERVAL '24 hours' PRECEDING
                            AND CURRENT ROW
                    ) - amount
                        AS calculated_amount_24h

                FROM historical
            )

            SELECT

                transaction_key,
                source_transaction_id,
                reused_source_id_flag,

                customer_id,
                card_number,

                transaction_ts,
                transaction_date,

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

                is_fraud,

                velocity_num_transactions,
                velocity_total_amount,
                velocity_unique_merchants,
                velocity_unique_countries,
                velocity_max_single_amount,

                customer_prior_transactions,

                previous_customer_transaction_ts,
                previous_customer_amount,
                previous_customer_merchant,
                previous_device_fingerprint,

                DATE_DIFF(
                    'second',
                    previous_customer_transaction_ts,
                    transaction_ts
                ) / 60.0
                    AS minutes_since_previous_transaction,

                customer_avg_amount_prior,
                customer_std_amount_prior,

                amount /
                    NULLIF(customer_avg_amount_prior, 0)
                    AS amount_vs_customer_average,

                (
                    amount - customer_avg_amount_prior
                ) /
                    NULLIF(customer_std_amount_prior, 0)
                    AS amount_zscore_vs_customer_history,

                calculated_txn_count_1h,
                calculated_amount_1h,

                calculated_txn_count_24h,
                calculated_amount_24h,

                CASE
                    WHEN customer_device_sequence = 1
                    THEN 1
                    ELSE 0
                END
                    AS new_device_for_customer_flag,

                CASE
                    WHEN customer_ip_sequence = 1
                    THEN 1
                    ELSE 0
                END
                    AS new_ip_for_customer_flag,

                CASE
                    WHEN customer_merchant_sequence = 1
                    THEN 1
                    ELSE 0
                END
                    AS new_merchant_for_customer_flag,

                CASE
                    WHEN previous_device_fingerprint IS NOT NULL
                     AND previous_device_fingerprint
                         <> device_fingerprint
                    THEN 1
                    ELSE 0
                END
                    AS device_changed_from_previous_flag,

                CASE
                    WHEN transaction_hour BETWEEN 0 AND 5
                    THEN 1
                    ELSE 0
                END
                    AS overnight_transaction_flag

            FROM velocity
            """
        )

        rows = con.execute(
            """
            SELECT COUNT(*)
            FROM analytics.behavioral_features
            """
        ).fetchone()[0]

        print(
            f"Created behavioral features for "
            f"{rows:,} transactions."
        )

        print("\n[2/4] Validating behavioral features...")

        validation = con.execute(
            """
            SELECT

                COUNT(*) AS transactions,

                COUNT(DISTINCT transaction_key)
                    AS unique_transaction_keys,

                SUM(
                    CASE
                        WHEN minutes_since_previous_transaction < 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS negative_time_gaps,

                ROUND(
                    AVG(calculated_txn_count_1h),
                    2
                )
                    AS avg_prior_transactions_1h,

                MAX(calculated_txn_count_1h)
                    AS max_prior_transactions_1h,

                ROUND(
                    AVG(calculated_txn_count_24h),
                    2
                )
                    AS avg_prior_transactions_24h,

                MAX(calculated_txn_count_24h)
                    AS max_prior_transactions_24h,

                ROUND(
                    100.0 *
                    AVG(new_device_for_customer_flag),
                    2
                )
                    AS new_device_pct,

                ROUND(
                    100.0 *
                    AVG(new_merchant_for_customer_flag),
                    2
                )
                    AS new_merchant_pct

            FROM analytics.behavioral_features
            """
        ).fetchdf()

        print(validation.T.to_string(header=False))

        validation.to_csv(
            REPORT_DIR /
            "phase3_behavioral_feature_summary.csv",
            index=False
        )

        print("\n[3/4] Comparing our velocity with source velocity...")

        velocity_check = con.execute(
            """
            SELECT

                COUNT(*) AS transactions,

                ROUND(
                    AVG(
                        ABS(
                            velocity_num_transactions -
                            calculated_txn_count_1h
                        )
                    ),
                    4
                )
                    AS avg_count_difference,

                ROUND(
                    AVG(
                        ABS(
                            velocity_total_amount -
                            calculated_amount_1h
                        )
                    ),
                    2
                )
                    AS avg_amount_difference,

                CORR(
                    velocity_num_transactions,
                    calculated_txn_count_1h
                )
                    AS count_correlation,

                CORR(
                    velocity_total_amount,
                    calculated_amount_1h
                )
                    AS amount_correlation

            FROM analytics.behavioral_features

            WHERE velocity_num_transactions IS NOT NULL
            """
        ).fetchdf()

        print(velocity_check.T.to_string(header=False))

        velocity_check.to_csv(
            REPORT_DIR /
            "source_vs_calculated_velocity.csv",
            index=False
        )

        print("\n[4/4] Measuring fraud rates for new behavioral signals...")

        fraud_signals = con.execute(
            """
            SELECT
                'new_device' AS signal,
                new_device_for_customer_flag AS signal_value,
                COUNT(*) AS transactions,
                SUM(is_fraud) AS fraud_transactions,
                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                ) AS fraud_rate_pct

            FROM analytics.behavioral_features

            GROUP BY new_device_for_customer_flag

            UNION ALL

            SELECT
                'new_ip',
                new_ip_for_customer_flag,
                COUNT(*),
                SUM(is_fraud),
                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                )

            FROM analytics.behavioral_features

            GROUP BY new_ip_for_customer_flag

            UNION ALL

            SELECT
                'new_merchant',
                new_merchant_for_customer_flag,
                COUNT(*),
                SUM(is_fraud),
                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                )

            FROM analytics.behavioral_features

            GROUP BY new_merchant_for_customer_flag

            UNION ALL

            SELECT
                'device_changed',
                device_changed_from_previous_flag,
                COUNT(*),
                SUM(is_fraud),
                ROUND(
                    100.0 * AVG(is_fraud),
                    4
                )

            FROM analytics.behavioral_features

            GROUP BY device_changed_from_previous_flag

            ORDER BY
                signal,
                signal_value
            """
        ).fetchdf()

        print(fraud_signals.to_string(index=False))

        fraud_signals.to_csv(
            REPORT_DIR /
            "behavioral_signal_fraud_rates.csv",
            index=False
        )

    finally:

        con.close()

    minutes = (time.time() - start) / 60

    print(
        f"\nPhase 3 complete in {minutes:.1f} minutes."
    )

    print("\nCreated:")
    print("  analytics.behavioral_features")
    print(
        "  reports/analysis/"
        "phase3_behavioral_feature_summary.csv"
    )
    print(
        "  reports/analysis/"
        "source_vs_calculated_velocity.csv"
    )
    print(
        "  reports/analysis/"
        "behavioral_signal_fraud_rates.csv"
    )

    print(
        "\nNext: card-device-IP relationships "
        "and network fraud features."
    )


if __name__ == "__main__":
    main()