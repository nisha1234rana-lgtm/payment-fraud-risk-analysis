-- Dataset inventory: columns and inferred DuckDB types

SELECT
    table_schema,
    table_name,
    column_name,
    data_type,
    ordinal_position
FROM information_schema.columns
WHERE table_schema IN ('raw', 'analytics')
ORDER BY table_schema, table_name, ordinal_position;
