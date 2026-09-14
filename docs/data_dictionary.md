# Source Data Dictionary

The selected source file contains 24 fields:

1. transaction_id
2. customer_id
3. card_number
4. timestamp
5. merchant_category
6. merchant_type
7. merchant
8. amount
9. currency
10. country
11. city
12. city_size
13. card_type
14. card_present
15. device
16. channel
17. device_fingerprint
18. ip_address
19. distance_from_home
20. high_risk_merchant
21. transaction_hour
22. weekend_transaction
23. velocity_last_hour
24. is_fraud

`velocity_last_hour` is intentionally preserved raw in Phase 1. We will inspect its exact
serialized structure before parsing it into multiple behavioral features.
