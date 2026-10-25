# Curated Data Quality

_Generated 2026-09-26 17:42 by `src/jobs/quality_main.py`._

6 checks on `hdfs://localhost:9000/user/binhn/fre/curated`: **0 failed**, 0 warned.

## Summary

| check                 | status   | result                                                          |
|:----------------------|:---------|:----------------------------------------------------------------|
| partition content     | PASS     | 18,132 file(s) checked; every partition column matches its path |
| keys                  | PASS     | no duplicate keys                                               |
| coverage              | PASS     | 1999-01 .. 2026-03, 327 month(s), contiguous                    |
| referential integrity | PASS     | every performance row has an origination row                    |
| value domains         | PASS     | nulls and domains as documented                                 |
| schema consistency    | PASS     | one schema per grain                                            |

## partition content -- PASS

18,132 file(s) checked; every partition column matches its path

## keys -- PASS

no duplicate keys

## coverage -- PASS

1999-01 .. 2026-03, 327 month(s), contiguous

## referential integrity -- PASS

every performance row has an origination row

## value domains -- PASS

nulls and domains as documented

| metric                   | value                      |
|:-------------------------|:---------------------------|
| performance rows scanned | 2,881,397,251              |
| zero balance codes seen  | 01, 02, 03, 09, 15, 16, 96 |

## schema consistency -- PASS

one schema per grain
