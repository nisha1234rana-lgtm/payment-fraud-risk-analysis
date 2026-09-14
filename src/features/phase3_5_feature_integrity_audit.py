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
    print("Phase 3.5: Feature Integrity Audit")
    print(f"Database: {DB_FILE}")

    con = duckdb.connect(str(DB_FILE))

    try:

        con.execute("SET preserve_insertion_order = false")
        con.execute("SET threads = 4")

        # --------------------------------------------------
        # 1. Strong feature / target relationships
        # --------------------------------------------------

        print("\n[1/5] Auditing unusually strong fraud signals...")

        signals = con.execute(
            """
            SELECT
                'new_device' AS feature,
                CAST(new_device_for_customer_flag AS VARCHAR) AS feature_value,
                COUNT(*) AS transactions,
                SUM(is_fraud) AS fraud_transactions,
                ROUND(100.0 * AVG(is_fraud), 4) AS fraud_rate_pct
            FROM analytics.behavioral_features
            GROUP BY new_device_for_customer_flag

            UNION ALL

            SELECT
                'device_changed',
                CAST(device_changed_from_previous_flag AS VARCHAR),
                COUNT(*),
                SUM(is_fraud),
                ROUND(100.0 * AVG(is_fraud), 4)
            FROM analytics.behavioral_features
            GROUP BY device_changed_from_previous_flag

            UNION ALL

            SELECT
                'new_merchant',
                CAST(new_merchant_for_customer_flag AS VARCHAR),
                COUNT(*),
                SUM(is_fraud),
                ROUND(100.0 * AVG(is_fraud), 4)
            FROM analytics.behavioral_features
            GROUP BY new_merchant_for_customer_flag

            UNION ALL

            SELECT
                'high_risk_merchant',
                CAST(high_risk_merchant AS VARCHAR),
                COUNT(*),
                SUM(is_fraud),
                ROUND(100.0 * AVG(is_fraud), 4)
            FROM analytics.behavioral_features
            GROUP BY high_risk_merchant

            UNION ALL

            SELECT
                'amount_under_10',
                CAST(
                    CASE WHEN amount < 10 THEN 1 ELSE 0 END
                    AS VARCHAR
                ),
                COUNT(*),
                SUM(is_fraud),
                ROUND(100.0 * AVG(is_fraud), 4)
            FROM analytics.behavioral_features
            GROUP BY
                CASE WHEN amount < 10 THEN 1 ELSE 0 END

            UNION ALL

            SELECT
                'amount_under_25',
                CAST(
                    CASE WHEN amount < 25 THEN 1 ELSE 0 END
                    AS VARCHAR
                ),
                COUNT(*),
                SUM(is_fraud),
                ROUND(100.0 * AVG(is_fraud), 4)
            FROM analytics.behavioral_features
            GROUP BY
                CASE WHEN amount < 25 THEN 1 ELSE 0 END

            ORDER BY feature, feature_value
            """
        ).fetchdf()

        print(signals.to_string(index=False))

        signals.to_csv(
            REPORT_DIR / "feature_integrity_signal_audit.csv",
            index=False
        )

        # --------------------------------------------------
        # 2. Device fingerprint usefulness
        # --------------------------------------------------

        print("\n[2/5] Measuring device-network usefulness...")

        device_summary = con.execute(
            """
            WITH device_stats AS (

                SELECT
                    device_fingerprint,

                    COUNT(*) AS transactions,

                    COUNT(DISTINCT customer_id)
                        AS customers,

                    COUNT(DISTINCT card_number)
                        AS cards,

                    SUM(is_fraud)
                        AS fraud_transactions,

                    MIN(is_fraud)
                        AS min_fraud,

                    MAX(is_fraud)
                        AS max_fraud

                FROM analytics.behavioral_features

                GROUP BY device_fingerprint
            )

            SELECT

                COUNT(*) AS devices,

                SUM(
                    CASE
                        WHEN transactions > 1 THEN 1
                        ELSE 0
                    END
                ) AS reused_devices,

                SUM(
                    CASE
                        WHEN customers > 1 THEN 1
                        ELSE 0
                    END
                ) AS devices_shared_by_customers,

                SUM(
                    CASE
                        WHEN cards > 1 THEN 1
                        ELSE 0
                    END
                ) AS devices_shared_by_cards,

                SUM(
                    CASE
                        WHEN min_fraud = 1
                         AND max_fraud = 1
                        THEN 1
                        ELSE 0
                    END
                ) AS fraud_only_devices,

                SUM(
                    CASE
                        WHEN min_fraud = 0
                         AND max_fraud = 0
                        THEN 1
                        ELSE 0
                    END
                ) AS legitimate_only_devices,

                SUM(
                    CASE
                        WHEN min_fraud <> max_fraud
                        THEN 1
                        ELSE 0
                    END
                ) AS mixed_fraud_devices

            FROM device_stats
            """
        ).fetchdf()

        print(device_summary.T.to_string(header=False))

        device_summary.to_csv(
            REPORT_DIR / "device_network_summary.csv",
            index=False
        )

        # --------------------------------------------------
        # 3. IP usefulness
        # --------------------------------------------------

        print("\n[3/5] Measuring IP-network usefulness...")

        ip_summary = con.execute(
            """
            WITH ip_stats AS (

                SELECT
                    ip_address,

                    COUNT(*) AS transactions,

                    COUNT(DISTINCT customer_id)
                        AS customers,

                    COUNT(DISTINCT card_number)
                        AS cards

                FROM analytics.behavioral_features

                GROUP BY ip_address
            )

            SELECT

                COUNT(*) AS ip_addresses,

                SUM(
                    CASE
                        WHEN transactions > 1
                        THEN 1
                        ELSE 0
                    END
                ) AS reused_ips,

                SUM(
                    CASE
                        WHEN customers > 1
                        THEN 1
                        ELSE 0
                    END
                ) AS ips_shared_by_customers,

                SUM(
                    CASE
                        WHEN cards > 1
                        THEN 1
                        ELSE 0
                    END
                ) AS ips_shared_by_cards,

                MAX(transactions)
                    AS max_transactions_per_ip

            FROM ip_stats
            """
        ).fetchdf()

        print(ip_summary.T.to_string(header=False))

        ip_summary.to_csv(
            REPORT_DIR / "ip_network_summary.csv",
            index=False
        )

        # --------------------------------------------------
        # 4. Top shared devices
        # --------------------------------------------------

        print("\n[4/5] Finding highly connected devices...")

        top_devices = con.execute(
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

            FROM analytics.behavioral_features

            GROUP BY device_fingerprint

            HAVING COUNT(*) > 1

            ORDER BY
                customers DESC,
                cards DESC,
                transactions DESC

            LIMIT 200
            """
        ).fetchdf()

        print(top_devices.head(20).to_string(index=False))

        top_devices.to_csv(
            REPORT_DIR / "top_shared_devices.csv",
            index=False
        )

        # --------------------------------------------------
        # 5. Source velocity audit
        # --------------------------------------------------

        print("\n[5/5] Auditing supplied vs calculated velocity...")

        velocity = con.execute(
            """
            SELECT

                ROUND(
                    AVG(velocity_num_transactions),
                    2
                ) AS source_avg_transaction_count,

                ROUND(
                    AVG(calculated_txn_count_1h),
                    2
                ) AS calculated_customer_avg_1h,

                ROUND(
                    CORR(
                        velocity_num_transactions,
                        calculated_txn_count_1h
                    ),
                    4
                ) AS count_correlation,

                ROUND(
                    CORR(
                        velocity_total_amount,
                        calculated_amount_1h
                    ),
                    4
                ) AS amount_correlation,

                ROUND(
                    AVG(
                        ABS(
                            velocity_num_transactions -
                            calculated_txn_count_1h
                        )
                    ),
                    2
                ) AS avg_absolute_count_difference

            FROM analytics.behavioral_features

            WHERE velocity_num_transactions IS NOT NULL
            """
        ).fetchdf()

        print(velocity.T.to_string(header=False))

        velocity.to_csv(
            REPORT_DIR / "velocity_integrity_audit.csv",
            index=False
        )

    finally:

        con.close()

    minutes = (time.time() - start) / 60

    print(
        f"\nPhase 3.5 complete in {minutes:.1f} minutes."
    )

    print("\nCreated:")
    print(
        "  reports/analysis/feature_integrity_signal_audit.csv"
    )
    print(
        "  reports/analysis/device_network_summary.csv"
    )
    print(
        "  reports/analysis/ip_network_summary.csv"
    )
    print(
        "  reports/analysis/top_shared_devices.csv"
    )
    print(
        "  reports/analysis/velocity_integrity_audit.csv"
    )

    print(
        "\nNext: build the device/card network using only "
        "relationships that actually contain useful reuse."
    )


if __name__ == "__main__":
    main()