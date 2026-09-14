from __future__ import annotations

import ast
import json
import re
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_FILE

REPORT_DIR = PROJECT_ROOT / "reports"
QUALITY_DIR = REPORT_DIR / "data_quality"
ANALYSIS_DIR = REPORT_DIR / "analysis"

QUALITY_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_column_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def parse_velocity_value(value):
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    # Try proper JSON first
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    # Try Python dictionary format
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    return None


def discover_velocity_keys(con):
    print("\n[1/7] Inspecting velocity_last_hour structure...")

    samples = con.execute(
        """
        SELECT velocity_last_hour_raw
        FROM staging.transactions_clean
        WHERE velocity_last_hour_raw IS NOT NULL
          AND TRIM(velocity_last_hour_raw) <> ''
        USING SAMPLE 10000 ROWS
        """
    ).fetchall()

    discovered = {}

    for row in samples:
        parsed = parse_velocity_value(row[0])

        if not parsed:
            continue

        for key, value in parsed.items():
            if key not in discovered:
                discovered[key] = type(value).__name__

    result = pd.DataFrame(
        [
            {
                "source_key": key,
                "column_name": f"velocity_{sanitize_column_name(key)}",
                "sample_type": value_type,
            }
            for key, value_type in discovered.items()
        ]
    )

    result.to_csv(
        QUALITY_DIR / "velocity_keys.csv",
        index=False,
    )

    print(f"Found {len(discovered)} velocity fields.")

    if discovered:
        print(result.to_string(index=False))
    else:
        print("No parseable velocity dictionary keys were discovered.")

    return list(discovered.keys())


def build_clean_table(con, velocity_keys):
    print("\n[2/7] Building analytics.transactions_clean...")

    velocity_expressions = []

    for key in velocity_keys:
        clean_name = sanitize_column_name(key)

        escaped_key = key.replace('"', '\\"')

        expression = f"""
        TRY_CAST(
            json_extract_string(
                TRY_CAST(
                    REPLACE(velocity_last_hour_raw, '''', '"')
                    AS JSON
                ),
                '$."{escaped_key}"'
            )
            AS DOUBLE
        ) AS velocity_{clean_name}
        """

        velocity_expressions.append(expression.strip())

    extra_velocity_sql = ""

    if velocity_expressions:
        extra_velocity_sql = ",\n            " + ",\n            ".join(
            velocity_expressions
        )

    query = f"""
    CREATE OR REPLACE TABLE analytics.transactions_clean AS

    SELECT
        transaction_id,
        customer_id,
        card_number,
        transaction_ts,

        CAST(transaction_ts AS DATE) AS transaction_date,
        EXTRACT(HOUR FROM transaction_ts) AS derived_hour,
        EXTRACT(DOW FROM transaction_ts) AS day_of_week,

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

        TRY_CAST(
            regexp_extract(
                distance_from_home_raw,
                '-?[0-9]+(?:\\.[0-9]+)?',
                0
            )
            AS DOUBLE
        ) AS distance_from_home,

        high_risk_merchant,
        transaction_hour,
        weekend_transaction,

        velocity_last_hour_raw,

        is_fraud

        {extra_velocity_sql}

    FROM staging.transactions_clean
    """

    con.execute(query)

    row_count = con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.transactions_clean
        """
    ).fetchone()[0]

    print(f"Created analytics.transactions_clean with {row_count:,} rows.")


def basic_quality_checks(con):
    print("\n[3/7] Running cleaned-data quality checks...")

    query = """
    SELECT
        COUNT(*) AS rows,

        COUNT(DISTINCT transaction_id)
            AS unique_transaction_ids,

        COUNT(DISTINCT customer_id)
            AS unique_customers,

        COUNT(DISTINCT card_number)
            AS unique_cards,

        COUNT(DISTINCT merchant)
            AS unique_merchants,

        COUNT(DISTINCT device_fingerprint)
            AS unique_device_fingerprints,

        COUNT(DISTINCT ip_address)
            AS unique_ip_addresses,

        MIN(transaction_ts)
            AS min_timestamp,

        MAX(transaction_ts)
            AS max_timestamp,

        SUM(
            CASE
                WHEN is_fraud = 1 THEN 1
                ELSE 0
            END
        ) AS fraud_transactions,

        ROUND(
            100.0 * AVG(is_fraud),
            4
        ) AS fraud_rate_pct,

        SUM(
            CASE
                WHEN amount IS NULL THEN 1
                ELSE 0
            END
        ) AS missing_amount,

        SUM(
            CASE
                WHEN amount < 0 THEN 1
                ELSE 0
            END
        ) AS negative_amount,

        SUM(
            CASE
                WHEN transaction_ts IS NULL THEN 1
                ELSE 0
            END
        ) AS missing_timestamp,

        SUM(
            CASE
                WHEN customer_id IS NULL THEN 1
                ELSE 0
            END
        ) AS missing_customer,

        SUM(
            CASE
                WHEN card_number IS NULL THEN 1
                ELSE 0
            END
        ) AS missing_card,

        SUM(
            CASE
                WHEN merchant IS NULL THEN 1
                ELSE 0
            END
        ) AS missing_merchant,

        SUM(
            CASE
                WHEN device_fingerprint IS NULL THEN 1
                ELSE 0
            END
        ) AS missing_device_fingerprint,

        SUM(
            CASE
                WHEN ip_address IS NULL THEN 1
                ELSE 0
            END
        ) AS missing_ip_address

    FROM analytics.transactions_clean
    """

    df = con.execute(query).fetchdf()

    duplicate_count = con.execute(
        """
        SELECT COUNT(*)

        FROM (
            SELECT transaction_id

            FROM analytics.transactions_clean

            WHERE transaction_id IS NOT NULL

            GROUP BY transaction_id

            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    df["duplicate_transaction_ids"] = duplicate_count

    df.to_csv(
        QUALITY_DIR / "phase2_quality_summary.csv",
        index=False,
    )

    print(df.T.to_string(header=False))


def profile_dimensions(con):
    print("\n[4/7] Profiling fraud across business dimensions...")

    dimensions = [
        "merchant_category",
        "merchant_type",
        "currency",
        "country",
        "city_size",
        "card_type",
        "card_present",
        "device",
        "channel",
        "high_risk_merchant",
        "transaction_hour",
        "weekend_transaction",
    ]

    results = []

    for dimension in dimensions:

        query = f"""
        SELECT
            '{dimension}' AS dimension,

            CAST({dimension} AS VARCHAR)
                AS category,

            COUNT(*)
                AS transactions,

            SUM(is_fraud)
                AS fraud_transactions,

            ROUND(
                100.0 * AVG(is_fraud),
                4
            ) AS fraud_rate_pct,

            ROUND(
                AVG(amount),
                2
            ) AS avg_transaction_amount,

            ROUND(
                SUM(amount),
                2
            ) AS total_transaction_value,

            ROUND(
                SUM(
                    CASE
                        WHEN is_fraud = 1
                        THEN amount
                        ELSE 0
                    END
                ),
                2
            ) AS fraud_transaction_value

        FROM analytics.transactions_clean

        GROUP BY {dimension}

        ORDER BY fraud_rate_pct DESC
        """

        df = con.execute(query).fetchdf()

        results.append(df)

    final = pd.concat(
        results,
        ignore_index=True,
    )

    final.to_csv(
        ANALYSIS_DIR / "fraud_profile_by_dimension.csv",
        index=False,
    )

    print(
        f"Saved {len(final):,} dimensional fraud-profile rows."
    )


def profile_amounts(con):
    print("\n[5/7] Profiling transaction amounts...")

    query = """
    WITH amount_bands AS (

        SELECT
            *,

            CASE

                WHEN amount < 10
                    THEN '01_under_10'

                WHEN amount < 25
                    THEN '02_10_25'

                WHEN amount < 50
                    THEN '03_25_50'

                WHEN amount < 100
                    THEN '04_50_100'

                WHEN amount < 250
                    THEN '05_100_250'

                WHEN amount < 500
                    THEN '06_250_500'

                WHEN amount < 1000
                    THEN '07_500_1000'

                WHEN amount < 2500
                    THEN '08_1000_2500'

                ELSE '09_2500_plus'

            END AS amount_band

        FROM analytics.transactions_clean

        WHERE amount IS NOT NULL
    )

    SELECT
        amount_band,

        COUNT(*)
            AS transactions,

        SUM(is_fraud)
            AS fraud_transactions,

        ROUND(
            100.0 * AVG(is_fraud),
            4
        ) AS fraud_rate_pct,

        ROUND(
            AVG(amount),
            2
        ) AS avg_amount,

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

    FROM amount_bands

    GROUP BY amount_band

    ORDER BY amount_band
    """

    df = con.execute(query).fetchdf()

    df.to_csv(
        ANALYSIS_DIR / "fraud_by_amount_band.csv",
        index=False,
    )

    print(df.to_string(index=False))


def profile_time(con):
    print("\n[6/7] Building daily fraud trend...")

    query = """
    SELECT
        transaction_date,

        COUNT(*)
            AS transactions,

        SUM(is_fraud)
            AS fraud_transactions,

        ROUND(
            100.0 * AVG(is_fraud),
            4
        ) AS fraud_rate_pct,

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

    FROM analytics.transactions_clean

    GROUP BY transaction_date

    ORDER BY transaction_date
    """

    df = con.execute(query).fetchdf()

    df.to_csv(
        ANALYSIS_DIR / "daily_fraud_trend.csv",
        index=False,
    )

    print(df.head(10).to_string(index=False))


def create_first_risk_flags(con):
    print("\n[7/7] Creating first analytical risk flags...")

    con.execute(
        """
        CREATE OR REPLACE VIEW analytics.transaction_risk_flags AS

        SELECT
            *,

            CASE
                WHEN amount >
                    AVG(amount) OVER (
                        PARTITION BY customer_id
                    ) * 3
                THEN 1
                ELSE 0
            END AS unusually_high_amount_flag,

            CASE
                WHEN high_risk_merchant = TRUE
                THEN 1
                ELSE 0
            END AS high_risk_merchant_flag,

            CASE
                WHEN transaction_hour BETWEEN 0 AND 5
                THEN 1
                ELSE 0
            END AS overnight_transaction_flag,

            CASE
                WHEN card_present = FALSE
                THEN 1
                ELSE 0
            END AS card_not_present_flag,

            ROW_NUMBER() OVER (
                PARTITION BY customer_id
                ORDER BY transaction_ts
            ) AS customer_transaction_number,

            LAG(transaction_ts) OVER (
                PARTITION BY customer_id
                ORDER BY transaction_ts
            ) AS previous_customer_transaction_ts,

            LAG(amount) OVER (
                PARTITION BY customer_id
                ORDER BY transaction_ts
            ) AS previous_customer_amount

        FROM analytics.transactions_clean
        """
    )

    print("Created analytics.transaction_risk_flags.")


def main():

    start = time.time()

    print("Payment Fraud Analysis")
    print("Phase 2: Cleaning + Fraud Profiling")
    print(f"Database: {DB_FILE}")

    con = duckdb.connect(str(DB_FILE))

    try:

        con.execute(
            "SET preserve_insertion_order = false"
        )

        velocity_keys = discover_velocity_keys(con)

        build_clean_table(
            con,
            velocity_keys,
        )

        basic_quality_checks(con)

        profile_dimensions(con)

        profile_amounts(con)

        profile_time(con)

        create_first_risk_flags(con)

    finally:
        con.close()

    minutes = (time.time() - start) / 60

    print(
        f"\nPhase 2 complete in {minutes:.1f} minutes."
    )

    print("\nCreated:")

    print(
        "  analytics.transactions_clean"
    )

    print(
        "  analytics.transaction_risk_flags"
    )

    print(
        "  reports/data_quality/phase2_quality_summary.csv"
    )

    print(
        "  reports/data_quality/velocity_keys.csv"
    )

    print(
        "  reports/analysis/fraud_profile_by_dimension.csv"
    )

    print(
        "  reports/analysis/fraud_by_amount_band.csv"
    )

    print(
        "  reports/analysis/daily_fraud_trend.csv"
    )

    print(
        "\nNext: behavioral SQL features and transaction velocity."
    )


if __name__ == "__main__":
    main()