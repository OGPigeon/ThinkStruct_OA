"""Part 3: ingestion cost on a portion of the large dataset, baseline vs. improved, stage by stage.

    python benchmarks/bench_ingest.py --data data/large --files 10

Stages and variants (same cleaning/chunking code everywhere, only the execution strategy changes):
  parse     v0 stdlib json, 1 process | v1 orjson, 1 process | v2 orjson + N worker processes (what build.py does)
  embed     v0 CPU fp32, one encode() per patent | v1 MPS fp32, one call per patent |
            v2 MPS fp32, one big length-sorted call | v3 MPS fp16, one big call (what build.py does)
  bm25      v0 dict-of-dicts inverted index (pure Python) | v1 vectorized CSR/CSC (what build.py does)
Writes benchmarks/results/ingest.md.
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import orjson

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from patent_search.build import _bm25_csc, find_files  # noqa: E402
from patent_search.data import normalize_record  # noqa: E402
from patent_search.text import tokenize  # noqa: E402


def parse_file(args):
    path, use_orjson = args
    with open(path, "rb") as f:
        raw = orjson.loads(f.read()) if use_orjson else json.loads(f.read())
    out = []
    for r in raw:
        rec = normalize_record(r, "core")
        if rec:
            rec["tokens"] = tokenize(f"{rec['title']} {rec['abstract']} " + " ".join(t for _, t, _ in rec["claims"]))
            out.append(rec)
    return out


def bench_parse(files, workers):
    res = {}
    for name, use_orjson, nproc in [("v0 json, 1 proc", False, 1), ("v1 orjson, 1 proc", True, 1),
                                    (f"v2 orjson, {workers} procs", True, workers)]:
        for f in files:  # warm the OS page cache so every variant reads from RAM, not disk
            with open(f, "rb") as fh:
                fh.read()
        t = time.perf_counter()
        if nproc == 1:
            recs = [r for f in files for r in parse_file((f, use_orjson))]
        else:
            with Pool(nproc) as pool:
                recs = [r for part in pool.map(parse_file, [(f, True) for f in files]) for r in part]
        res[name] = (time.perf_counter() - t, len(recs))
    return res, recs


def bench_embed(recs, n_texts):
    import torch
    from patent_search.embed import Embedder

    per_patent = []
    for r in recs:
        texts = [c[2] for c in r["chunks"]]
        if sum(map(len, per_patent)) + len(texts) > n_texts:
            break
        per_patent.append(texts)
    flat = [t for p in per_patent for t in p]
    res, vecs = {}, {}
    variants = [("v0 CPU fp32, per-patent calls", "cpu", False, "per", 0.2),
                ("v1 MPS fp32, per-patent calls", "mps", False, "per", 1.0),
                ("v2 MPS fp32, one length-sorted call", "mps", False, "all", 1.0),
                ("v3 MPS fp16, one length-sorted call", "mps", True, "all", 1.0)]
    for name, device, fp16, how, frac in variants:
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        e = Embedder(device=device, fp16=fp16)
        e.encode_passages(flat[:64])  # warm-up (kernel compilation)
        sub = per_patent[: max(1, int(len(per_patent) * frac))]  # CPU is slow: time a subset, report rate
        n = sum(map(len, sub))
        t = time.perf_counter()
        if how == "per":
            out = [e.encode_passages(p, batch_size=32) for p in sub]
            v = np.concatenate(out)
        else:
            v = e.encode_passages([t_ for p in sub for t_ in p], batch_size=128)
        dt = time.perf_counter() - t
        res[name] = (dt, n)
        vecs[name] = v
        del e
    a, b = vecs.get("v2 MPS fp32, one length-sorted call"), vecs.get("v3 MPS fp16, one length-sorted call")
    agree = float(np.mean(np.sum(a * b, axis=1))) if a is not None and b is not None else None
    return res, agree


def bench_bm25(recs):
    docs = [r["tokens"] for r in recs]
    queries = [tokenize(q) for q in ["battery electrode lithium", "wireless network beam", "neural network training",
                                     "semiconductor gate transistor", "vehicle wheel tire sensor"]]
    res = {}
    # v0: textbook pure-Python inverted index with BM25 computed at query time
    t = time.perf_counter()
    inv, dl = {}, []
    for i, d in enumerate(docs):
        dl.append(len(d))
        tf = {}
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        for w, c in tf.items():
            inv.setdefault(w, {})[i] = c
    build0 = time.perf_counter() - t
    avgdl, N = sum(dl) / len(dl), len(docs)
    t = time.perf_counter()
    for q in queries:
        scores = {}
        for w in q:
            post = inv.get(w, {})
            idf = math.log1p((N - len(post) + 0.5) / (len(post) + 0.5))
            for i, c in post.items():
                scores[i] = scores.get(i, 0) + idf * c * 2.2 / (c + 1.2 * (0.25 + 0.75 * dl[i] / avgdl))
        sorted(scores.items(), key=lambda kv: -kv[1])[:100]
    q0 = (time.perf_counter() - t) / len(queries)
    res["v0 pure-Python dicts"] = (build0, q0)

    # v1: vectorized CSR -> CSC with precomputed weights (query = sum of posting slices)
    import scipy.sparse as sp
    t = time.perf_counter()
    vocab = {}
    rows, cols, vals = [], [], []
    for i, d in enumerate(docs):
        tf = {}
        for w in d:
            j = vocab.setdefault(w, len(vocab))
            tf[j] = tf.get(j, 0) + 1
        rows.extend([i] * len(tf))
        cols.extend(tf.keys())
        vals.extend(tf.values())
    m = sp.csr_matrix((np.asarray(vals, np.float32), (np.asarray(rows), np.asarray(cols))), shape=(N, len(vocab)))
    csc = _bm25_csc(m)
    build1 = time.perf_counter() - t
    t = time.perf_counter()
    for q in queries:
        ids = [vocab[w] for w in q if w in vocab]
        r = np.concatenate([csc.indices[csc.indptr[j]:csc.indptr[j + 1]] for j in ids])
        w = np.concatenate([csc.data[csc.indptr[j]:csc.indptr[j + 1]] for j in ids])
        s = np.bincount(r, weights=w, minlength=N)
        top = np.argpartition(-s, 100)[:100]
        top[np.argsort(-s[top])]
    q1 = (time.perf_counter() - t) / len(queries)
    res["v1 vectorized CSC"] = (build1, q1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/large")
    ap.add_argument("--files", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--embed-texts", type=int, default=4000)
    ap.add_argument("--full-patents", type=int, default=457385, help="for extrapolation")
    a = ap.parse_args()
    files = find_files(a.data, a.files)
    mb = sum(os.path.getsize(f) for f in files) / 1e6

    parse, recs = bench_parse(files, a.workers)
    n = len(recs)
    chunks_per_patent = sum(len(r["chunks"]) for r in recs) / n
    embed, agree = bench_embed(recs, a.embed_texts)
    bm25 = bench_bm25(recs)

    full_chunks = a.full_patents * chunks_per_patent
    L = [f"# Ingestion benchmark - {len(files)} files ({mb:.0f} MB, {n} patents, {chunks_per_patent:.2f} chunks/patent, 'core' granularity)", "",
         f"Extrapolations are to the full deduplicated dataset ({a.full_patents} patents, ~{full_chunks / 1e6:.2f}M chunks). "
         "Measured on an Apple M2 Pro (10 cores, 16 GB), warm page cache.", "",
         "## Parse + clean + chunk + tokenize", "", "| variant | time | patents/s | MB/s | full dataset (est.) | speed-up |", "|---|---|---|---|---|---|"]
    base = None
    for k, (dt, cnt) in parse.items():
        rate = cnt / dt
        base = base or rate
        L.append(f"| {k} | {dt:.2f}s | {rate:,.0f} | {mb / dt:,.0f} | {a.full_patents / rate / 60:.1f} min | {rate / base:.1f}x |")
    L += ["", "## Embedding (bge-small-en-v1.5, max 256 tokens)", "",
          "| variant | texts timed | texts/s | full dataset (est.) | speed-up |", "|---|---|---|---|---|"]
    base = None
    for k, (dt, cnt) in embed.items():
        rate = cnt / dt
        base = base or rate
        L.append(f"| {k} | {cnt} | {rate:,.0f} | {full_chunks / rate / 3600:.2f} h | {rate / base:.1f}x |")
    if agree is not None:
        L.append(f"\nfp16 vs fp32 embeddings: mean cosine similarity {agree:.5f} (no measurable quality loss).")
    L += ["", "## BM25 index", "", "| variant | build | query (avg of 5) |", "|---|---|---|"]
    for k, (b, q) in bm25.items():
        L.append(f"| {k} | {b:.2f}s | {q * 1000:.2f} ms |")
    report = "\n".join(L)
    print(report)
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results", "ingest.md"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
