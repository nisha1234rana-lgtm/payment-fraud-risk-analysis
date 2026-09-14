from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_FILE

REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("Payment Fraud Analysis")
    print("Phase 2.6: Duplicate Conflict Analysis")
    print(f"Database: {DB_FILE}")

    con = duckdb.connect(str(DB_FILE))

    try:

        # --------------------------------------------------
        # 1. Duplicate group sizes
        # --------------------------------------------------

        print("\n[1/5] Checking duplicate group sizes...")

        group_sizes = con.execute(
            """
            WITH duplicate_groups AS (

                SELECT
                    transaction_id,
                    COUNT(*) AS rows_per_id

                FROM analytics.transactions_clean

                GROUP BY transaction_id

                HAVING COUNT(*) > 1
            )

            SELECT
                rows_per_id,
                COUNT(*) AS transaction_ids

            FROM duplicate_groups

            GROUP BY rows_per_id

            ORDER BY rows_per_id
            """
        ).fetchdf()

        print(group_sizes.to_string(index=False))

        group_sizes.to_csv(
            REPORT_DIR / "duplicate_group_sizes.csv",
            index=False
        )

        # --------------------------------------------------
        # 2. Check important business-field conflicts
        # --------------------------------------------------

        print("\n[2/5] Checking which important fields conflict...")

        conflict_summary = con.execute(
            """
            WITH duplicates AS (

                SELECT *

                FROM analytics.transactions_clean

                WHERE transaction_id IN (

                    SELECT transaction_id

                    FROM analytics.transactions_clean

                    GROUP BY transaction_id

                    HAVING COUNT(*) > 1
                )
            ),

            grouped AS (

                SELECT
                    transaction_id,

                    COUNT(DISTINCT customer_id) AS customer_versions,
                    COUNT(DISTINCT card_number) AS card_versions,
                    COUNT(DISTINCT transaction_ts) AS timestamp_versions,
                    COUNT(DISTINCT merchant) AS merchant_versions,
                    COUNT(DISTINCT amount) AS amount_versions,
                    COUNT(DISTINCT currency) AS currency_versions,
                    COUNT(DISTINCT country) AS country_versions,
                    COUNT(DISTINCT city) AS city_versions,
                    COUNT(DISTINCT card_type) AS card_type_versions,
                    COUNT(DISTINCT device) AS device_versions,
                    COUNT(DISTINCT channel) AS channel_versions,
                    COUNT(DISTINCT device_fingerprint) AS device_fingerprint_versions,
                    COUNT(DISTINCT ip_address) AS ip_versions,
                    COUNT(DISTINCT high_risk_merchant) AS high_risk_versions,
                    COUNT(DISTINCT is_fraud) AS fraud_label_versions

                FROM duplicates

                GROUP BY transaction_id
            )

            SELECT

                COUNT(*) AS duplicate_ids,

                SUM(customer_versions > 1) AS customer_conflicts,
                SUM(card_versions > 1) AS card_conflicts,
                SUM(timestamp_versions > 1) AS timestamp_conflicts,
                SUM(merchant_versions > 1) AS merchant_conflicts,
                SUM(amount_versions > 1) AS amount_conflicts,
                SUM(currency_versions > 1) AS currency_conflicts,
                SUM(country_versions > 1) AS country_conflicts,
                SUM(city_versions > 1) AS city_conflicts,
                SUM(card_type_versions > 1) AS card_type_conflicts,
                SUM(device_versions > 1) AS device_conflicts,
                SUM(channel_versions > 1) AS channel_conflicts,
                SUM(device_fingerprint_versions > 1)
                    AS device_fingerprint_conflicts,
                SUM(ip_versions > 1) AS ip_conflicts,
                SUM(high_risk_versions > 1) AS high_risk_conflicts,
                SUM(fraud_label_versions > 1) AS fraud_label_conflicts

            FROM grouped
            """
        ).fetchdf()

        print(conflict_summary.T.to_string(header=False))

        conflict_summary.to_csv(
            REPORT_DIR / "duplicate_field_conflicts.csv",
            index=False
        )

        # --------------------------------------------------
        # 3. Fraud label conflict specifically
        # --------------------------------------------------

        print("\n[3/5] Investigating fraud-label conflicts...")

        fraud_conflicts = con.execute(
            """
            SELECT
                transaction_id,

                COUNT(*) AS rows_per_id,

                COUNT(DISTINCT is_fraud) AS fraud_versions,

                MIN(is_fraud) AS min_fraud_label,

                MAX(is_fraud) AS max_fraud_label

            FROM analytics.transactions_clean

            GROUP BY transaction_id

            HAVING
                COUNT(*) > 1
                AND COUNT(DISTINCT is_fraud) > 1

            ORDER BY transaction_id

            LIMIT 100
            """
        ).fetchdf()

        print(
            f"Fraud-label conflict sample rows: "
            f"{len(fraud_conflicts):,}"
        )

        fraud_conflicts.to_csv(
            REPORT_DIR / "duplicate_fraud_label_conflicts.csv",
            index=False
        )

        # --------------------------------------------------
        # 4. Show detailed duplicate examples
        # --------------------------------------------------

        print("\n[4/5] Saving detailed conflict examples...")

        examples = con.execute(
            """
            WITH ranked_duplicates AS (

                SELECT
                    transaction_id,

                    COUNT(DISTINCT customer_id) +
                    COUNT(DISTINCT card_number) +
                    COUNT(DISTINCT transaction_ts) +
                    COUNT(DISTINCT merchant) +
                    COUNT(DISTINCT amount) +
                    COUNT(DISTINCT device_fingerprint) +
                    COUNT(DISTINCT ip_address) +
                    COUNT(DISTINCT is_fraud)
                        AS conflict_score

                FROM analytics.transactions_clean

                GROUP BY transaction_id

                HAVING COUNT(*) > 1

                ORDER BY conflict_score DESC

                LIMIT 50
            )

            SELECT
                t.*

            FROM analytics.transactions_clean t

            INNER JOIN ranked_duplicates d
                ON t.transaction_id = d.transaction_id

            ORDER BY
                d.conflict_score DESC,
                t.transaction_id,
                t.transaction_ts
            """
        ).fetchdf()

        examples.to_csv(
            REPORT_DIR / "duplicate_conflict_examples.csv",
            index=False
        )

        print(
            f"Saved {len(examples):,} detailed duplicate rows."
        )

        # --------------------------------------------------
        # 5. Test whether rows look like separate transactions
        # --------------------------------------------------

        print("\n[5/5] Testing duplicate ID behavior...")

        behavior = con.execute(
            """
            WITH d AS (

                SELECT
                    transaction_id,

                    COUNT(*) AS rows_per_id,

                    COUNT(DISTINCT customer_id) AS customers,

                    COUNT(DISTINCT card_number) AS cards,

                    COUNT(DISTINCT transaction_ts) AS timestamps,

                    COUNT(DISTINCT amount) AS amounts,

                    COUNT(DISTINCT merchant) AS merchants,

                    COUNT(DISTINCT is_fraud) AS fraud_labels

                FROM analytics.transactions_clean

                GROUP BY transaction_id

                HAVING COUNT(*) > 1
            )

            SELECT

                COUNT(*) AS duplicate_ids,

                SUM(timestamps > 1) AS different_timestamp_ids,

                SUM(customers > 1) AS different_customer_ids,

                SUM(cards > 1) AS different_card_ids,

                SUM(amounts > 1) AS different_amount_ids,

                SUM(merchants > 1) AS different_merchant_ids,

                SUM(fraud_labels > 1) AS different_fraud_label_ids,

                SUM(
                    CASE
                        WHEN timestamps > 1
                         AND customers > 1
                         AND cards > 1
                        THEN 1
                        ELSE 0
                    END
                ) AS strongly_separate_transaction_like_ids

            FROM d
            """
        ).fetchdf()

        print(behavior.T.to_string(header=False))

        behavior.to_csv(
            REPORT_DIR / "duplicate_behavior_summary.csv",
            index=False
        )

    finally:
        con.close()

    print("\nPhase 2.6 complete.")
    print(
        "Do not delete duplicates yet. "
        "The results will determine whether these are corrupted duplicates "
        "or separate transactions sharing an ID."
    )


if __name__ == "__main__":
    main()