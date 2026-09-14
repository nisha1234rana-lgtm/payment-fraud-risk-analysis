from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"

SOURCE_FILE = RAW_DIR / "synthetic_fraud_data.csv"
DB_FILE = PROCESSED_DIR / "payment_fraud.duckdb"
