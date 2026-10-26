# Vector search benchmark - `indexes/large` (457385 patents, 1668005 vectors x 384d)

50 queries (20 natural-language + 30 patent abstracts used as queries), batch size 1, best of 2; recall@10 at patent level vs exact search; dense retrieval only (ms).

## Unfiltered

| method | index memory | build | median | p95 | recall@10 |
|---|---|---|---|---|---|
| exact brute force (numpy fp32) | 2.56 GB | 0.7s | 51.6 | 54.4 | 1.000 |
| HNSW M=32, efSearch=max(32, k) | 3.02 GB | 97.99s | 1.9 | 2.8 | 0.994 |
| HNSW M=32, efSearch=max(64, k) | 3.02 GB | 97.99s | 1.6 | 2.2 | 0.994 |
| HNSW M=32, efSearch=max(128, k) | 3.02 GB | 97.99s | 1.5 | 2.2 | 0.994 |
| HNSW M=32, efSearch=max(256, k) | 3.02 GB | 97.99s | 1.5 | 2.2 | 0.994 |
| IVF5166-SQ8 (int8), nprobe=8 | ~0.66 GB | 35s | 0.6 | 0.7 | 0.846 |
| IVF5166-SQ8 (int8), nprobe=32 | ~0.66 GB | 35s | 1.2 | 1.6 | 0.926 |
| IVF5166-SQ8 (int8), nprobe=128 | ~0.66 GB | 35s | 3.5 | 4.2 | 0.980 |

## Filtered: which plan wins at which selectivity (HNSW, efSearch=max(128, k))

| filter | patents passing | chunks passing | plan | median | p95 | recall@10 vs exact filtered |
|---|---|---|---|---|---|---|
| cls H01L | 15481 (3.38%) | 51783 | prefilter-exact **(auto)** | 9.8 | 10.4 | 1.000 |
| cls H01L | 15481 (3.38%) | 51783 | filtered-ann | 10.7 | 20.7 | 0.414 |
| cls H01L | 15481 (3.38%) | 51783 | postfilter-ann | 315.0 | 387.7 | 0.836 |
| cls G06F or H04L | 36694 (8.02%) | 142657 | prefilter-exact **(auto)** | 26.3 | 28.9 | 1.000 |
| cls G06F or H04L | 36694 (8.02%) | 142657 | filtered-ann | 5.0 | 7.2 | 0.646 |
| cls G06F or H04L | 36694 (8.02%) | 142657 | postfilter-ann | 124.2 | 132.9 | 0.946 |
| kw network | 25325 (5.54%) | 105293 | prefilter-exact **(auto)** | 17.4 | 17.9 | 1.000 |
| kw network | 25325 (5.54%) | 105293 | filtered-ann | 2.5 | 3.4 | 0.728 |
| kw network | 25325 (5.54%) | 105293 | postfilter-ann | 227.2 | 295.7 | 0.996 |
| cls H01M + kw electrode* + 2021-22 | 337 (0.07%) | 897 | prefilter-exact **(auto)** | 0.1 | 0.1 | 1.000 |
| cls H01M + kw electrode* + 2021-22 | 337 (0.07%) | 897 | filtered-ann | 1.9 | 3.0 | 0.112 |
| cls H01M + kw electrode* + 2021-22 | 337 (0.07%) | 897 | postfilter-ann | 324.9 | 355.7 | 0.212 |
| title phrase | 9915 (2.17%) | 34479 | prefilter-exact **(auto)** | 6.3 | 7.6 | 1.000 |
| title phrase | 9915 (2.17%) | 34479 | filtered-ann | 2.3 | 3.1 | 0.376 |
| title phrase | 9915 (2.17%) | 34479 | postfilter-ann | 305.7 | 334.0 | 0.850 |
| kw data (broad) | 60992 (13.33%) | 246166 | prefilter-exact **(auto)** | 38.6 | 40.7 | 1.000 |
| kw data (broad) | 60992 (13.33%) | 246166 | filtered-ann | 3.5 | 4.4 | 0.836 |
| kw data (broad) | 60992 (13.33%) | 246166 | postfilter-ann | 46.7 | 52.4 | 0.986 |
| year >= 2010 (very broad) | 329028 (71.94%) | 1129236 | prefilter-exact | skipped (> 400000 chunks) | | |
| year >= 2010 (very broad) | 329028 (71.94%) | 1129236 | filtered-ann | 10.3 | 11.8 | 0.992 |
| year >= 2010 (very broad) | 329028 (71.94%) | 1129236 | postfilter-ann **(auto)** | 12.8 | 14.5 | 0.998 |
