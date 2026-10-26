# Ingestion benchmark - 10 files (684 MB, 10000 patents, 3.66 chunks/patent, 'core' granularity)

Extrapolations are to the full deduplicated dataset (457385 patents, ~1.67M chunks). Measured on an Apple M2 Pro (10 cores, 16 GB), warm page cache.

## Parse + clean + chunk + tokenize

| variant | time | patents/s | MB/s | full dataset (est.) | speed-up |
|---|---|---|---|---|---|
| v0 json, 1 proc | 7.85s | 1,273 | 87 | 6.0 min | 1.0x |
| v1 orjson, 1 proc | 6.94s | 1,441 | 99 | 5.3 min | 1.1x |
| v2 orjson, 8 procs | 2.36s | 4,233 | 290 | 1.8 min | 3.3x |

## Embedding (bge-small-en-v1.5, max 256 tokens)

| variant | texts timed | texts/s | full dataset (est.) | speed-up |
|---|---|---|---|---|
| v0 CPU fp32, per-patent calls | 782 | 76 | 6.13 h | 1.0x |
| v1 MPS fp32, per-patent calls | 3997 | 168 | 2.76 h | 2.2x |
| v2 MPS fp32, one length-sorted call | 3997 | 312 | 1.49 h | 4.1x |
| v3 MPS fp16, one length-sorted call | 3997 | 359 | 1.29 h | 4.7x |

fp16 vs fp32 embeddings: mean cosine similarity 1.00016 (no measurable quality loss).

## BM25 index

| variant | build | query (avg of 5) |
|---|---|---|
| v0 pure-Python dicts | 0.67s | 0.42 ms |
| v1 vectorized CSC | 0.68s | 0.15 ms |
