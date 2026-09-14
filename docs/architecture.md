# Architecture

```text
IEEE-CIS CSV files
        |
        v
Python ingestion
        |
        v
DuckDB
  raw.train_transaction
  raw.train_identity
        |
        v
analytics.transaction_base
        |
        v
SQL feature engineering
        |
        v
Fraud analytics / ML / risk decisions
        |
        +--> Power BI
        +--> Streamlit
        +--> Monitoring
```

## Data handling rule

Raw IEEE-CIS competition files are never committed to Git.
The repository stores code, SQL, documentation and derived project logic only.
