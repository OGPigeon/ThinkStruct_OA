# Hybrid patent prior-art search with claim mapping

Thinkstruct coding task.

| Part | Contents |
|---|---|
| **1** Search engine | `patent_search/`: hybrid engine (BM25 + vectors + filters, RRF), CLI, HTTP API + UI, Google Patents scraper |
| **2** System at scale | [`DESIGN.md`](DESIGN.md) design doc. The proof of concept is [`poc/pipeline.py`](poc/pipeline.py): a job queue with leases, retries and a dead-letter state, a parse→embed DAG and a live status dashboard |
| **3** Enhancement | **Efficiency and large data**: the 471k-patent / 32 GB sample, measured before and after optimization (`benchmarks/`) |

## Problem statement

> An examiner or patent attorney has either an invention description, a single claim, or an existing
> patent. They need to find the most relevant prior art in the corpus, see *why* each result matches
> (which claim or paragraph corresponds to which of their claims), and narrow the results by
> classification, keywords, title and date. Results must come back fast enough to use interactively,
> and that must still hold on the ~0.5M-patent sample, not just the 640-patent toy set.

So the engine supports:

* **Inputs:**
  * a natural-language query
  * a claim text
  * a **patent number**. Its independent claims each become a query, and the results come back as a
    claim chart. If the number is not in the index, it is scraped from Google Patents
    (`patent_search/fetch.py`).
* **Outputs:** ranked patents with metadata, a semantic rank and a BM25 rank, matched terms, and
  **evidence**: for every query row, the single best-matching passage (abstract, a specific claim, or a
  description paragraph) and its similarity.
* **Hybrid constraints:**
  * classification prefix(es) (`B60B`, `H01L21/`, OR-ed)
  * required or excluded **keywords in title/abstract** (`tire*` wildcards; multi-word keywords
    are AND-ed)
  * a **title** phrase, or the exact title
  * a date range (`2024`, `2024-03`, `2024-03-15`)

## Quick start

```bash
python3.14 -m venv .venv && source .venv/bin/activate   # any Python ≥ 3.10
pip install -r requirements.txt

# data: put the small set at data/small/*.json and the large set at data/large/job_*/*.json
# (or pass --data <dir>; files are found recursively)

# 1) build the small index (~70 s on an M2 Pro: embedding 28k passages dominates)
python -m patent_search.build --data data/small --out indexes/small --granularity full

# 2) search from the CLI
python -m patent_search.cli "tire pressure sensor mounted inside the wheel rim"
python -m patent_search.cli "airless tire" --cls B60C --kw "spoke*" --from 2024-06
python -m patent_search.cli "lightweight rim" --title "vehicle wheel"
python -m patent_search.cli --title "VEHICLE WHEEL" --title-exact --from 2024-06
python -m patent_search.cli --patent US20240051333A1          # claim mapping for an indexed patent
python -m patent_search.cli --patent US20230001234A1          # not indexed -> fetched from Google Patents
python -m patent_search.cli --claim "a hub body having a first end and a second end; ..." --mode dense

# 3) or the web UI / JSON API
python -m patent_search.server --index small=indexes/small     # http://127.0.0.1:8000
```

The first query in a process takes ~3 s (model load and MPS kernel compilation). After that, query
embedding takes 15–20 ms. The server warms up before it starts serving.

**Large dataset (Part 3).** Expect about 80 minutes on an M2 Pro, almost all of it embedding:

```bash
python -m patent_search.build --data data/large --out indexes/large --granularity core --ann hnsw --workers 6
python -m patent_search.cli --index indexes/large "solid state battery electrolyte" --cls H01M
```

Every build stage is resumable. Shards and per-shard vectors are checkpointed under `<out>/work`, so
an interrupted build continues where it stopped (`--stages prepare|embed|assemble` re-runs one stage).

## Part 1: the search engine

### Data handling (missing fields, duplicates, formats)

The choice made for incomplete records: **keep every patent that can be identified and has
something to search. Drop only records without a `doc_number`, or with no title, no abstract and
no claims.** The logic is in `patent_search/data.py`.

| case | handling | small set | large sample |
|---|---|---|---|
| missing/empty `detailed_description` | kept; searchable by title, abstract and claims | 0 | 4,950 |
| classification `N/A` / missing | kept; never matches a classification filter | 0 | 179,812 (38%) |
| missing title or abstract | kept; that field contributes nothing | 0 | 0 |
| no doc_number, or nothing searchable | dropped, counted in `meta.json` | 0 | 0 |
| duplicate `doc_number` | **keep the most complete copy** (most claim + description text) | 0 | 13,615 removed |
| corrupt / unparsable file | skipped and reported; the build continues | 0 | 0 |

Other cleaning steps:

* **Claims.** The raw claims carry their own numbering (`"2 . The insert of claim 1 , wherein"`),
  and canceled claims can cover ranges (`"1 - 15 . (canceled)"`). Numbers are parsed, canceled
  claims are dropped, and claims are tagged independent or dependent (no reference to "claim N").
* **Classification codes.** The two datasets use different formats (`B60B304FI` vs `H01L21/0234`).
  Both are normalized to a search key that keeps the subclass, so `B60B` works on either.
* **Dates.** Small set: the publication date comes from the filename. Large set: only the year is
  available, from the doc number. Date filters respect that precision.
* **Description text.** Reference numerals glued to words (`insert100according`) are separated
  before embedding.

### Retrieval

* **Passages ("chunks").** Each patent becomes chunks: `title. abstract`, claims, and description
  passages (paragraphs merged to ≥400 characters, split at ~1,200).
  * `full` granularity (small set): everything, 44 chunks per patent.
  * `core` granularity (large set): title+abstract plus *independent* claims, 3.7 chunks per patent.
    That is what fits a laptop; see Part 3.
* **Semantic.** `BAAI/bge-small-en-v1.5` (384-d), fp16 on Apple MPS. Short queries use bge's
  retrieval instruction; claims and patents are embedded as passages. A patent's score is its best
  chunk's cosine similarity, averaged over query rows. With several claims as queries, patents that
  match many of them rank first.
* **Lexical.** BM25 (k1 = 1.2, b = 0.75) over title + abstract + claims. Term weights are precomputed
  into a column-major sparse matrix, so scoring a query is just summing a few posting-list slices
  (`np.bincount`).
* **Fusion.** Reciprocal rank fusion (k = 60) of the top-100 semantic and top-100 BM25 patents.
  `--mode dense|bm25` is available for comparison.
* **Evidence.** For the final top-k only, each query row is compared against every chunk of the
  patent exactly, and the best chunk's text is fetched from SQLite.

**Index layout** (`indexes/<name>/`):

* `patents.sqlite`: text store, used only to display the top-k
* `vectors.npy`: fp16
* `chunk_patent` / `patent_offsets`: chunks are contiguous per patent
* BM25 CSC arrays, and keyword posting lists for title+abstract and for title alone
* sorted classification keys, date arrays, a sorted vocabulary, and the optional `ann_*.faiss`
* `meta.json`: build statistics

### Hybrid search, and how it is kept fast

The naive way to combine semantic search with constraints is to score everything, then walk the
ranking and test each patent's raw text and metadata until k survive. This engine filters
**first**, using structures built at index time, and never touches document text:

| constraint | structure | cost |
|---|---|---|
| classification prefix | sorted array of normalized codes plus a permutation; a prefix is a contiguous range found by binary search | O(log n) |
| keywords in title/abstract | inverted index (CSC matrix of title+abstract terms). `tire*` uses a sorted-vocabulary range. Multi-word keywords and multiple keywords intersect posting lists, smallest first | O(postings) |
| title phrase / exact title | title-only posting lists to get "all words present", then the word order is checked only on those few survivors | O(postings + survivors) |
| date range | two int32 arrays, vectorized comparison | O(n), about 1 ms at 0.5M |

The candidate set then drives a **selectivity-based plan** for the vector search (`PatentIndex.dense_search`):

* **prefilter-exact:** few candidate chunks (≤ 60k). Brute force over just those vectors; per-patent
  maxima with `np.maximum.reduceat` over each patent's contiguous chunk range. Exact, and for very selective
  filters the vector step is cheaper than an unfiltered ANN search.
* **filtered-ann:** a moderate number of candidates. HNSW/IVF search restricted by a faiss
  `IDSelectorBitmap`, so the graph only returns allowed chunks.
* **postfilter-ann:** most chunks pass (≥ 50%). ANN with over-fetch (k / selectivity × 1.5), then drop
  disallowed chunks.
* **exact-masked:** no ANN index and a broad filter. One contiguous matmul with a patent-level mask
  beats gathering a large subset.

BM25 applies the same candidate mask before `bincount`, so both rankings that go into RRF obey
exactly the same filter.

#### Timing: with and without hybrid filters

Command: `python benchmarks/bench_hybrid.py --index indexes/<small|large> --suite <small|large>`.

Measurement setup:

* 20 queries, best of 3 runs each.
* Retrieval latency only (filter + dense + BM25 + fusion + evidence). Query embedding is excluded
  because it is the same for every variant and is reported separately.
* The naive baseline produces the same result payload. Its keyword check is a plain substring test,
  the cheapest possible, so the speed-ups are conservative.

**Small set:** 640 patents, 27,999 passages, exact vectors with no ANN index. Query embedding takes ~20 ms.

| filter | patents passing | naive post-filter median / p95 | filter-first median / p95 | speed-up (median) | plan |
|---|---|---|---|---|---|
| **none (no filters)** | all | - | 1.86 / 1.99 | baseline | exact |
| cls B60B | 298 (46.56%) | 1.98 / 2.19 | 1.92 / 2.03 | 1.0x | exact-masked |
| kw sensor* | 32 (5.00%) | 2.50 / 2.77 | 1.14 / 1.28 | 2.2x | prefilter-exact |
| cls B60C + kw noise + date | 2 (0.31%) | 1.79 / 2.02 | 0.28 / 0.32 | 6.3x | prefilter-exact |
| title phrase | 29 (4.53%) | 4.26 / 4.73 | 1.16 / 1.26 | 3.7x | prefilter-exact |

**Large sample:** 457,385 patents, 1.67M passages, HNSW. Query embedding takes ~11 ms.

| filter | patents passing | naive post-filter median / p95 | filter-first median / p95 | speed-up (median) | plan |
|---|---|---|---|---|---|
| **none (no filters)** | all | - | 4.73 / 6.31 | baseline | hnsw |
| cls H01L | 15481 (3.38%) | 107.47 / 116.73 | 13.38 / 14.39 | 8.0x | prefilter-exact |
| cls G06F or H04L | 36694 (8.02%) | 96.05 / 108.49 | 28.71 / 30.30 | 3.3x | prefilter-exact |
| kw network | 25325 (5.54%) | 98.37 / 113.99 | 19.93 / 21.17 | 4.9x | prefilter-exact |
| cls H01M + kw electrode* + 2021-22 | 337 (0.07%) | 252.25 / 366.01 | 5.44 / 5.70 | 46.4x | prefilter-exact |
| title phrase | 9915 (2.17%) | 234.59 / 338.61 | 11.78 / 12.97 | 19.9x | prefilter-exact |

**Commentary**

* **At 640 patents everything takes 1–4 ms.** Filtering first still wins 2–6× for selective
  filters. For a broad filter (47% pass) it only breaks even, which is why the planner switches to
  a masked full scan there.
* **At 457k patents the naive post-filter costs 96–250 ms.** It has to score all 1.67M vectors and
  then walk deep into the ranking (for the 0.07% filter, tens of thousands of patents). That is
  regardless of how cheap the predicates are.
* **Filter-first costs 5–29 ms and returns the same exact results** (3–46× faster). A very
  selective filter costs about the same as no filter: 337 candidates take 5.4 ms end to end, against
  4.7 ms unfiltered, and the vector step alone is 0.1 ms vs 1.5 ms for HNSW.
* **Correctness was the deciding factor, not speed.** The tempting option for mid-selectivity
  filters is HNSW with a bitmap filter (2–5 ms). But its recall@10 collapses to 0.11–0.84 when
  3–15% of vectors pass, because the graph walk runs out of allowed neighbours (table in Part 3).
  The planner therefore brute-forces up to 300k candidate chunks, which costs ≤ 40 ms with recall
  1.0. Filtered HNSW is used only above that (measured 0.995 recall from ~35% selectivity), and
  over-fetch + post-filter only when most vectors pass.

Things that made it efficient:

* Each patent's chunks are contiguous, so a candidate set maps to row ranges and per-patent maxima
  are a single `np.maximum.reduceat`.
* Classification codes are pre-sorted, so prefix search is a binary search.
* The vocabulary is sorted, so wildcards are a range.
* Posting lists are intersected smallest-first.
* BM25 weights are precomputed, so lexical scoring is a `bincount` over posting slices.
* Document text is never touched until the final top-k is displayed.

What I would add at larger scale:

* Partition the vector index by classification section, so the most common filters become a
  smaller index instead of a filter.
* Bitmap-compressed posting lists (Roaring).
* Run the candidate brute force on the GPU, or with int8 vectors.
* Raise HNSW `efSearch` in proportion to 1/selectivity for filtered queries.
* Cache filter results for repeated filter combinations, which is common in examiner sessions.

## Part 2: implementation at scale

See **[DESIGN.md](DESIGN.md)**. It covers:

* the architecture: S3, Postgres, SQS queues with CPU and GPU workers, OpenSearch, and a stateless
  query service
* the backfill, incremental, re-embedding and query pipelines
* error handling, how to track what is in the system, and alarms
* a cost estimate: ≈ $2.1k/month steady state, under $300 one-time
* known challenges, and a step-by-step implementation plan

**Proof of concept: the ingestion control plane.** It runs the real Part 1 parsing and embedding
code over the large dataset:

```bash
python poc/pipeline.py init --data data/large --limit 24      # 24 parse jobs + 1 deliberately corrupt file
python poc/pipeline.py dashboard &                             # live status page: http://127.0.0.1:8001
python poc/pipeline.py run --parse-workers 4 --embed-workers 1 --chaos 0.15
```

What it demonstrates:

* SQLite stands in for Postgres. The atomic claim is a single `UPDATE ... RETURNING`.
* Leases are renewed by a heartbeat thread. Enqueueing is idempotent.
* Each finished `parse` job enqueues an `embed` job for its shard. The two stages have separate
  worker pools.
* Transient errors retry with exponential backoff and are dead-lettered after 4 attempts. Permanent
  errors (the corrupt file) are dead-lettered immediately.
* A supervisor restarts crashed workers.
* `--chaos` randomly injects transient errors and **hard-kills workers mid-job**. Their jobs are
  taken over by other workers once the lease expires.

As the task allows, this is quick code written in limited time.

## Part 3: efficiency and large data

**Why this enhancement.** Part 1 only becomes a real search engine once it holds up at a size where
naive choices hurt: a full scan, per-record Python loops, unfiltered ANN plus post-filtering, fp32
everywhere. The large sample (471k records, 32 GB of JSON, 1.7M vectors) makes those costs
measurable on a laptop. It also feeds directly into the Part 2 design, because cost and RAM at 10M
are extrapolated from these numbers.

All numbers below were measured on an Apple M2 Pro (10 cores, 16 GB RAM, no discrete GPU), with a
warm OS page cache.

### 1. Ingestion: baseline vs. improved (`benchmarks/bench_ingest.py`, 10 files / 10k patents, extrapolated)

Extrapolations are to the full deduplicated dataset (457385 patents, ~1.67M chunks). Measured on an Apple M2 Pro (10 cores, 16 GB), warm page cache.

#### Parse + clean + chunk + tokenize

| variant | time | patents/s | MB/s | full dataset (est.) | speed-up |
|---|---|---|---|---|---|
| v0 json, 1 proc | 7.85s | 1,273 | 87 | 6.0 min | 1.0x |
| v1 orjson, 1 proc | 6.94s | 1,441 | 99 | 5.3 min | 1.1x |
| v2 orjson, 8 procs | 2.36s | 4,233 | 290 | 1.8 min | 3.3x |

#### Embedding (bge-small-en-v1.5, max 256 tokens)

| variant | texts timed | texts/s | full dataset (est.) | speed-up |
|---|---|---|---|---|
| v0 CPU fp32, per-patent calls | 782 | 76 | 6.13 h | 1.0x |
| v1 MPS fp32, per-patent calls | 3997 | 168 | 2.76 h | 2.2x |
| v2 MPS fp32, one length-sorted call | 3997 | 312 | 1.49 h | 4.1x |
| v3 MPS fp16, one length-sorted call | 3997 | 359 | 1.29 h | 4.7x |

fp16 vs fp32 embeddings: mean cosine similarity ≈ 1.000 (1.00016, the excess over 1 is fp16 rounding), so there is no measurable quality loss.

#### BM25 index

| variant | build | query (avg of 5) |
|---|---|---|
| v0 pure-Python dicts | 0.67s | 0.42 ms |
| v1 vectorized CSC | 0.68s | 0.15 ms |


### 2. The real full build with the improved pipeline (`indexes/large/meta.json`)

| stage | what | time |
|---|---|---|
| prepare | 471 files / 32.2 GB of JSON: parse, clean, chunk, tokenize; 13,615 duplicates removed (6 worker processes) | **65 s** (≈ 495 MB/s, 7,200 patents/s) |
| embed | 1,668,005 chunks (title+abstract and independent claims), MPS fp16 | **81 min** (342 chunks/s) |
| assemble | SQLite text store, BM25 and keyword postings (472k-term vocabulary, 1.3 s), vectors, HNSW | **115 s** (HNSW alone: 98 s) |
| **total** | | **≈ 84 min**, vs. **≈ 6.3 h** estimated for the baseline (6 min parse + 6.1 h CPU embedding + Python-dict BM25) |

* Embedding is 96% of the build. That is why the fp16, MPS and batching work mattered most, and why
  the large set uses `core` granularity: about 120 chunks per patent would take about 45 h here.
* The build is resumable at shard granularity. During development the original run segfaulted in
  the very last step (a macOS libomp clash between torch and faiss, now fixed in
  `patent_search/__init__.py`). Re-running only `--stages assemble` took 2 minutes, instead of
  repeating 81 minutes of embedding.
* Small set for comparison: 0.4 s to prepare, 68 s to embed 28k passages (413/s), 0.1 s to assemble.

### 3. Search: exact vs. ANN, and filtered plans (`benchmarks/bench_search.py`)

Measured with 50 queries: 20 natural-language queries plus 30 patent abstracts used as queries.
Recall@10 is at patent level against exact search, and times are dense retrieval only, in ms.

#### Unfiltered

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

#### Filtered: every plan, forced, at different selectivities ("(auto)" = what the planner picks)

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

### Summary of improvements (all shipped as the defaults)

| area | baseline | improved | gain |
|---|---|---|---|
| parse + clean + tokenize | stdlib `json`, 1 process: 1.3k patents/s | `orjson` + 6–8 worker processes, compact shards: up to 7.2k patents/s | 3.3–5.6× |
| embedding | CPU fp32, one call per patent: 76 texts/s | MPS + fp16 + one length-sorted batch per shard: 359 texts/s | **4.7×**, quality unchanged |
| full ingest of the large sample | ≈ 6.3 h (estimated) | 84 min (measured), resumable per shard | ≈ 4.5× |
| unfiltered vector search | exact brute force: 52 ms | HNSW (M=32, ef=128): 1.5 ms, recall@10 0.994 | **34×** |
| filtered (hybrid) search | score everything, post-filter: 96–252 ms | filter-first + selectivity planner: 5–29 ms, exact results | **3–46×** |
| filtered ANN correctness | bitmap HNSW for every filter: recall 0.11–0.84 | planner thresholds calibrated on this data: recall ≥ 0.998 everywhere | correctness fix |
| memory option | HNSW: 3.0 GB | IVF-SQ8: 0.66 GB at 0.98 recall (nprobe=128, 3.4 ms) | 4.6× smaller, for when RAM matters more (DESIGN.md) |

How this was approached:

1. Profile the raw data first (`scripts/profile_large.py`). That surfaced the duplicates, empty
   descriptions, `N/A` placeholders and the mixed classification formats before they could corrupt
   the index.
2. Measure each stage separately. Embedding dominated the build at 96%.
3. Fix the biggest cost first.
4. Calibrate every speed trick against an exact baseline. The filtered-HNSW recall collapse was
   only found because every plan is checked for recall, not just latency.

Two known leftovers:

* 42,448 large-set titles were the literal string `"N/A"`. That was found after embedding, so it is
  fixed in the text store and the filters, and those abstract chunks keep a harmless `"N/A. "`
  prefix in their vectors until the next re-embed.
* The first full scan of the freshly unzipped data ran at 16 MB/s because Spotlight was indexing
  it. All numbers above use a warm page cache.

## Reproducing every number

```bash
python -m patent_search.build --data data/small --out indexes/small --granularity full
python -m patent_search.build --data data/large --out indexes/large --granularity core --ann hnsw --workers 6
python benchmarks/bench_hybrid.py --index indexes/small --suite small     # -> benchmarks/results/hybrid_small.md
python benchmarks/bench_hybrid.py --index indexes/large --suite large     # -> benchmarks/results/hybrid_large.md
python benchmarks/bench_ingest.py --data data/large --files 10            # -> benchmarks/results/ingest.md
python benchmarks/bench_search.py --index indexes/large --ivf             # -> benchmarks/results/search.md
python scripts/profile_large.py data/large                                # data-quality profile of the raw files
```

Notes:

* Build statistics (records in, dropped, duplicates, stage timings) are written to
  `indexes/<name>/meta.json`.
* On macOS, exclude `data/` from Spotlight before timing. While Spotlight indexed the freshly
  unzipped 32 GB, the first full scan ran at 16 MB/s instead of ~500 MB/s.

## Repository layout

```
patent_search/
  text.py      tokenizer, claim parsing, classification normalization
  data.py      raw JSON -> cleaned records -> chunks/tokens; missing-field + dedup policy; worker entry point
  embed.py     bge-small wrapper (MPS/CUDA/CPU, fp16)
  build.py     resumable prepare -> embed -> assemble pipeline (SQLite, numpy, BM25/CSC, faiss)
  index.py     PatentIndex: filters, planner, dense/BM25 search, RRF, evidence
  fetch.py     Google Patents scraper for patent-number queries not in the index
  cli.py       command line
  server.py    FastAPI JSON API + static/index.html UI
benchmarks/    bench_hybrid.py (Part 1 timing), bench_ingest.py and bench_search.py (Part 3), results/*.md
poc/           pipeline.py: Part 2 proof of concept (job queue, workers, dashboard)
scripts/       profile_large.py: one-pass data-quality scan of the raw files
DESIGN.md      Part 2 system design
```

## Limitations and next steps

* **Large set coverage.** It is indexed at `core` granularity (no description passages), so its
  evidence is the abstract or an independent claim. Embedding all ~120 chunks per patent would take
  about 45 h on this laptop, but only about $50 of spot GPU time (see DESIGN.md).
* **Truncation.** Inputs are cut at 256 tokens, which affects 13.7% of independent claims. Windowed
  claim chunks would fix this.
* **No relevance evaluation.** There is no labelled test set; ANN quality is measured only as recall
  against exact search. The natural next step is the evaluation-and-training enhancement, for
  example using examiner citations as positives.
* **Thread safety.** One process serves one search at a time (a lock in `server.py`). Scaling out is
  covered in DESIGN.md.
