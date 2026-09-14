-- First fraud profile after Phase 1

SELECT
    COUNT(*) AS transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(100.0 * AVG(is_fraud), 4) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_value
FROM staging.transactions_clean;

SELECT
    merchant_category,
    COUNT(*) AS transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(100.0 * AVG(is_fraud), 3) AS fraud_rate_pct,
    ROUND(AVG(amount), 2) AS avg_amount
FROM staging.transactions_clean
GROUP BY merchant_category
ORDER BY fraud_rate_pct DESC;

SELECT
    channel,
    COUNT(*) AS transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(100.0 * AVG(is_fraud), 3) AS fraud_rate_pct
FROM staging.transactions_clean
GROUP BY channel
ORDER BY fraud_rate_pct DESC;

SELECT
    transaction_hour,
    COUNT(*) AS transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(100.0 * AVG(is_fraud), 3) AS fraud_rate_pct
FROM staging.transactions_clean
GROUP BY transaction_hour
ORDER BY transaction_hour;
