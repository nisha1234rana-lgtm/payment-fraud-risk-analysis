-- Phase 1 raw-data validation

SELECT
    COUNT(*) AS transaction_rows,
    COUNT(DISTINCT TransactionID) AS unique_transaction_ids,
    SUM(CASE WHEN TransactionID IS NULL THEN 1 ELSE 0 END) AS null_transaction_ids,
    SUM(CASE WHEN TransactionAmt IS NULL OR TransactionAmt < 0 THEN 1 ELSE 0 END)
        AS invalid_or_null_transaction_amounts,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    AVG(CAST(isFraud AS DOUBLE)) AS fraud_rate
FROM raw.train_transaction;

SELECT
    COUNT(*) AS identity_rows,
    COUNT(DISTINCT TransactionID) AS unique_identity_transaction_ids
FROM raw.train_identity;

SELECT
    COUNT(*) AS joined_rows,
    AVG(CASE WHEN i.TransactionID IS NOT NULL THEN 1.0 ELSE 0.0 END)
        AS identity_match_rate
FROM raw.train_transaction AS t
LEFT JOIN raw.train_identity AS i
    ON t.TransactionID = i.TransactionID;
