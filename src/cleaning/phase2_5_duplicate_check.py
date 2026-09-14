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
    print("Phase 2.5: Duplicate Transaction Investigation")
    print(f"Database: {DB_FILE}")

    con = duckdb.connect(str(DB_FILE))

    try:

        print("\n[1/4] Measuring duplicate transaction IDs...")

        summary = con.execute(
            """
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT transaction_id) AS unique_transaction_ids,
                COUNT(*) - COUNT(DISTINCT transaction_id) AS extra_duplicate_rows
            FROM analytics.transactions_clean
            """
        ).fetchdf()

        duplicate_groups = con.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT transaction_id
                FROM analytics.transactions_clean
                GROUP BY transaction_id
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]

        summary["duplicate_id_groups"] = duplicate_groups

        print(summary.T.to_string(header=False))

        summary.to_csv(
            REPORT_DIR / "duplicate_summary.csv",
            index=False
        )

        print("\n[2/4] Checking whether duplicate IDs contain conflicting records...")

        conflicts = con.execute(
            """
            WITH duplicate_versions AS (

                SELECT
                    transaction_id,
                    COUNT(*) AS row_count,

                    COUNT(
                        DISTINCT HASH(
                            customer_id,
                            card_number,
                            transaction_ts,
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
                            is_fraud
                        )
                    ) AS distinct_versions

                FROM analytics.transactions_clean

                GROUP BY transaction_id

                HAVING COUNT(*) > 1
            )

            SELECT
                COUNT(*) AS duplicate_transaction_ids,

                SUM(
                    CASE
                        WHEN distinct_versions = 1 THEN 1
                        ELSE 0
                    END
                ) AS exact_duplicate_ids,

                SUM(
                    CASE
                        WHEN distinct_versions > 1 THEN 1
                        ELSE 0
                    END
                ) AS conflicting_duplicate_ids

            FROM duplicate_versions
            """
        ).fetchdf()

        print(conflicts.T.to_string(header=False))

        conflicts.to_csv(
            REPORT_DIR / "duplicate_conflict_summary.csv",
            index=False
        )

        print("\n[3/4] Saving duplicate samples...")

        samples = con.execute(
            """
            WITH duplicate_ids AS (

                SELECT transaction_id

                FROM analytics.transactions_clean

                GROUP BY transaction_id

                HAVING COUNT(*) > 1

                LIMIT 100
            )

            SELECT t.*

            FROM analytics.transactions_clean t

            INNER JOIN duplicate_ids d
                ON t.transaction_id = d.transaction_id

            ORDER BY
                t.transaction_id,
                t.transaction_ts
            """
        ).fetchdf()

        samples.to_csv(
            REPORT_DIR / "duplicate_samples.csv",
            index=False
        )

        print(
            f"Saved {len(samples):,} duplicate sample rows."
        )

        conflict_count = int(
            conflicts.iloc[0]["conflicting_duplicate_ids"]
        )

        print("\n[4/4] Creating analysis-ready transaction table...")

        if conflict_count == 0:

            con.execute(
                """
                CREATE OR REPLACE TABLE analytics.transactions_final AS

                SELECT *

                FROM analytics.transactions_clean

                QUALIFY
                    ROW_NUMBER() OVER (
                        PARTITION BY transaction_id
                        ORDER BY transaction_ts
                    ) = 1
                """
            )

            final_rows = con.execute(
                """
                SELECT COUNT(*)
                FROM analytics.transactions_final
                """
            ).fetchone()[0]

            print(
                "All duplicate transaction IDs were exact duplicates."
            )

            print(
                f"Created analytics.transactions_final "
                f"with {final_rows:,} unique transactions."
            )

        else:

            print(
                f"Found {conflict_count:,} transaction IDs "
                "with conflicting records."
            )

            print(
                "No automatic deduplication was performed."
            )

            print(
                "We need to inspect those records before choosing "
                "which version to keep."
            )

    finally:
        con.close()

    print("\nPhase 2.5 complete.")


if __name__ == "__main__":
    main()