# Hybrid search timing - `indexes/large` (457385 patents, 1668005 chunks, ann=hnsw)

20 queries x best of 3 runs; retrieval only (query embedding excluded: 10.6 ms/query, identical for all variants). Times in ms.

| filter | patents passing | naive post-filter median / p95 | filter-first median / p95 | speed-up (median) | plan |
|---|---|---|---|---|---|
| **none (no filters)** | all | - | 4.73 / 6.31 | baseline | hnsw |
| cls H01L | 15481 (3.38%) | 107.47 / 116.73 | 13.38 / 14.39 | 8.0x | prefilter-exact |
| cls G06F or H04L | 36694 (8.02%) | 96.05 / 108.49 | 28.71 / 30.30 | 3.3x | prefilter-exact |
| kw network | 25325 (5.54%) | 98.37 / 113.99 | 19.93 / 21.17 | 4.9x | prefilter-exact |
| cls H01M + kw electrode* + 2021-22 | 337 (0.07%) | 252.25 / 366.01 | 5.44 / 5.70 | 46.4x | prefilter-exact |
| title phrase | 9915 (2.17%) | 234.59 / 338.61 | 11.78 / 12.97 | 19.9x | prefilter-exact |
