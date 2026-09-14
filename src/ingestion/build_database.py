from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_FILE, PROCESSED_DIR, RAW_DIR, REPORT_DIR, SOURCE_FILE


def check_source() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{SOURCE_FILE}\n\n"
            "Download it first with:\n"
            "powershell -ExecutionPolicy Bypass -File .\\scripts\\download_data.ps1"
        )


def create_database(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute("CREATE SCHEMA IF NOT EXISTS staging")
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics")


def ingest_raw(con: duckdb.DuckDBPyConnection) -> None:
    print("\n[1/4] Loading source CSV into raw.transactions ...")
    print("Raw values are preserved as text so cleaning decisions remain explicit.")

    con.execute(
        """
        CREATE OR REPLACE TABLE raw.transactions AS
        SELECT *
        FROM read_csv(
            ?,
            header = true,
            all_varchar = true,
            auto_detect = true,
            sample_size = 200000,
            parallel = true
        )
        """,
        [str(SOURCE_FILE)],
    )

    rows = con.execute("SELECT COUNT(*) FROM raw.transactions").fetchone()[0]
    print(f"Loaded {rows:,} raw rows.")


def build_staging(con: duckdb.DuckDBPyConnection) -> None:
    print("\n[2/4] Building typed staging.transactions_clean ...")

    con.execute(
        """
        CREATE OR REPLACE TABLE staging.transactions_clean AS
        SELECT
            NULLIF(TRIM(transaction_id), '') AS transaction_id,
            NULLIF(TRIM(customer_id), '') AS customer_id,
            NULLIF(TRIM(card_number), '') AS card_number,

            TRY_CAST(timestamp AS TIMESTAMP) AS transaction_ts,

            NULLIF(TRIM(merchant_category), '') AS merchant_category,
            NULLIF(TRIM(merchant_type), '') AS merchant_type,
            NULLIF(TRIM(merchant), '') AS merchant,

            TRY_CAST(amount AS DOUBLE) AS amount,
            UPPER(NULLIF(TRIM(currency), '')) AS currency,

            NULLIF(TRIM(country), '') AS country,
            NULLIF(TRIM(city), '') AS city,
            LOWER(NULLIF(TRIM(city_size), '')) AS city_size,

            NULLIF(TRIM(card_type), '') AS card_type,

            CASE
                WHEN LOWER(TRIM(card_present)) IN ('true','1','yes','y') THEN TRUE
                WHEN LOWER(TRIM(card_present)) IN ('false','0','no','n') THEN FALSE
                ELSE NULL
            END AS card_present,

            NULLIF(TRIM(device), '') AS device,
            LOWER(NULLIF(TRIM(channel), '')) AS channel,
            NULLIF(TRIM(device_fingerprint), '') AS device_fingerprint,
            NULLIF(TRIM(ip_address), '') AS ip_address,

            NULLIF(TRIM(distance_from_home), '') AS distance_from_home_raw,

            CASE
                WHEN LOWER(TRIM(high_risk_merchant)) IN ('true','1','yes','y') THEN TRUE
                WHEN LOWER(TRIM(high_risk_merchant)) IN ('false','0','no','n') THEN FALSE
                ELSE NULL
            END AS high_risk_merchant,

            TRY_CAST(transaction_hour AS INTEGER) AS transaction_hour,

            CASE
                WHEN LOWER(TRIM(weekend_transaction)) IN ('true','1','yes','y') THEN TRUE
                WHEN LOWER(TRIM(weekend_transaction)) IN ('false','0','no','n') THEN FALSE
                ELSE NULL
            END AS weekend_transaction,

            NULLIF(TRIM(velocity_last_hour), '') AS velocity_last_hour_raw,

            CASE
                WHEN LOWER(TRIM(is_fraud)) IN ('true','1','yes','y') THEN 1
                WHEN LOWER(TRIM(is_fraud)) IN ('false','0','no','n') THEN 0
                ELSE NULL
            END AS is_fraud

        FROM raw.transactions
        """
    )

    rows = con.execute("SELECT COUNT(*) FROM staging.transactions_clean").fetchone()[0]
    print(f"Created {rows:,} staged rows.")


def build_profile(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    print("\n[3/4] Profiling dataset ...")

    profile_query = """
    SELECT
        COUNT(*) AS row_count,
        COUNT(DISTINCT transaction_id) AS unique_transaction_ids,
        COUNT(DISTINCT customer_id) AS unique_customers,
        COUNT(DISTINCT card_number) AS unique_cards,
        COUNT(DISTINCT merchant) AS unique_merchants,
        COUNT(DISTINCT device_fingerprint) AS unique_device_fingerprints,
        COUNT(DISTINCT ip_address) AS unique_ip_addresses,
        COUNT(DISTINCT country) AS countries,
        COUNT(DISTINCT currency) AS currencies,

        MIN(transaction_ts) AS min_timestamp,
        MAX(transaction_ts) AS max_timestamp,

        SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
        AVG(CAST(is_fraud AS DOUBLE)) AS fraud_rate,

        SUM(CASE WHEN transaction_id IS NULL THEN 1 ELSE 0 END) AS missing_transaction_id,
        SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS missing_customer_id,
        SUM(CASE WHEN transaction_ts IS NULL THEN 1 ELSE 0 END) AS invalid_or_missing_timestamp,
        SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS invalid_or_missing_amount,
        SUM(CASE WHEN is_fraud IS NULL THEN 1 ELSE 0 END) AS invalid_or_missing_fraud_label,

        SUM(CASE WHEN amount < 0 THEN 1 ELSE 0 END) AS negative_amounts
    FROM staging.transactions_clean
    """

    df = con.execute(profile_query).fetchdf()

    duplicate_ids = con.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT transaction_id
            FROM staging.transactions_clean
            WHERE transaction_id IS NOT NULL
            GROUP BY transaction_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    df["duplicate_transaction_ids"] = duplicate_ids

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "phase1_profile.csv"
    df.to_csv(output, index=False)

    print("\nDataset profile:")
    for col in df.columns:
        print(f"  {col}: {df.iloc[0][col]}")

    print(f"\nSaved report: {output.relative_to(PROJECT_ROOT)}")
    return df


def build_missingness_report(con: duckdb.DuckDBPyConnection) -> None:
    print("\n[4/4] Building raw-column missingness report ...")

    columns = [
        r[0]
        for r in con.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'raw'
              AND table_name = 'transactions'
            ORDER BY ordinal_position
            """
        ).fetchall()
    ]

    pieces = []
    for col in columns:
        safe = col.replace('"', '""')
        pieces.append(
            f"""
            SELECT
                '{safe}' AS column_name,
                COUNT(*) AS total_rows,
                SUM(
                    CASE
                        WHEN "{safe}" IS NULL
                          OR TRIM("{safe}") = ''
                          OR LOWER(TRIM("{safe}")) IN ('null','none','nan','na','n/a')
                        THEN 1 ELSE 0
                    END
                ) AS missing_rows
            FROM raw.transactions
            """
        )

    query = "\nUNION ALL\n".join(pieces)
    missing = con.execute(query).fetchdf()
    missing["missing_pct"] = (missing["missing_rows"] / missing["total_rows"] * 100).round(4)
    missing = missing.sort_values(["missing_pct", "column_name"], ascending=[False, True])

    output = REPORT_DIR / "missingness_report.csv"
    missing.to_csv(output, index=False)
    print(f"Saved report: {output.relative_to(PROJECT_ROOT)}")


def main() -> None:
    start = time.time()

    check_source()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source:   {SOURCE_FILE}")
    print(f"Database: {DB_FILE}")

    con = duckdb.connect(str(DB_FILE))
    try:
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET threads = 4")

        create_database(con)
        ingest_raw(con)
        build_staging(con)
        build_profile(con)
        build_missingness_report(con)

    finally:
        con.close()

    elapsed = (time.time() - start) / 60
    print(f"\nPhase 1 complete in {elapsed:.1f} minutes.")
    print("Next phase: inspect anomalies, parse velocity_last_hour, and build SQL fraud profiling.")


if __name__ == "__main__":
    main()
