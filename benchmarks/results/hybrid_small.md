# Hybrid search timing - `indexes/small` (640 patents, 27999 chunks, ann=flat)

20 queries x best of 3 runs; retrieval only (query embedding excluded: 15.9 ms/query, identical for all variants). Times in ms.

| filter | patents passing | naive post-filter median / p95 | filter-first median / p95 | speed-up (median) | plan |
|---|---|---|---|---|---|
| **none (no filters)** | all | - | 1.86 / 1.99 | baseline | exact |
| cls B60B | 298 (46.56%) | 1.98 / 2.19 | 1.92 / 2.03 | 1.0x | exact-masked |
| kw sensor* | 32 (5.00%) | 2.50 / 2.77 | 1.14 / 1.28 | 2.2x | prefilter-exact |
| cls B60C + kw noise + date | 2 (0.31%) | 1.79 / 2.02 | 0.28 / 0.32 | 6.3x | prefilter-exact |
| title phrase | 29 (4.53%) | 4.26 / 4.73 | 1.16 / 1.26 | 3.7x | prefilter-exact |
