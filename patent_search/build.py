"""Build a search index from raw patent JSON files.

    python -m patent_search.build --data data/small --out indexes/small --granularity full
    python -m patent_search.build --data data/large --out indexes/large --granularity core --ann hnsw

Three resumable stages (each can be re-run on its own with --stages):
  prepare   raw JSON -> cleaned/tokenized shards, in parallel worker processes; dedup by doc_number
  embed     shard chunks -> fp16 vectors, one checkpoint file per shard (a crash loses at most one shard)
  assemble  shards + vectors -> final index (SQLite text store, numpy arrays, BM25 + keyword postings, ANN)
"""
import argparse
import glob
import json
import os
import sqlite3
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp

from .data import GRANULARITIES, load_shard, process_file

BM25_K1, BM25_B = 1.2, 0.75


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_files(data_dir, limit=None):
    files = sorted(p for p in glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
                   if os.path.basename(p) != "sample.json")
    return files[:limit] if limit else files


# --------------------------------------------------------------------------------------------- prepare
def stage_prepare(files, work, granularity, workers):
    shard_dir = os.path.join(work, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    jobs, summaries = [], {}
    for i, src in enumerate(files):
        shard = os.path.join(shard_dir, f"{i:05d}.pkl")
        summ = shard + ".json"
        if os.path.exists(shard) and os.path.exists(summ):  # resume: shard already done
            with open(summ) as f:
                s = json.load(f)
            if s["source"] == src:
                summaries[i] = s
                continue
        jobs.append((src, shard, granularity))

    t = time.time()
    if jobs:
        log(f"prepare: {len(jobs)} files to process ({len(summaries)} already done), {workers} workers")
        with Pool(workers) as pool:
            for n, s in enumerate(pool.imap_unordered(process_file, jobs), 1):
                i = int(os.path.basename(s["shard"]).split(".")[0])
                summaries[i] = s
                if s["error"] is None:
                    with open(s["shard"] + ".json", "w") as f:
                        json.dump(s, f)
                else:
                    log(f"  CORRUPT FILE skipped: {s['source']}: {s['error']}")
                if n % 25 == 0 or n == len(jobs):
                    log(f"  {n}/{len(jobs)} files ({time.time() - t:.1f}s)")
    parse_s = time.time() - t

    # Dedup: keep the most complete copy of each doc_number (ties -> first seen in file order).
    best = {}
    order = sorted(i for i, s in summaries.items() if s["error"] is None)
    for i in order:
        s = summaries[i]
        for j, (doc, q) in enumerate(zip(s["doc_number"], s["quality"])):
            cur = best.get(doc)
            if cur is None or q > cur[0]:
                best[doc] = (q, i, j)
    keep = {i: np.zeros(len(summaries[i]["doc_number"]), dtype=bool) for i in order}
    for _, i, j in best.values():
        keep[i][j] = True

    stats = {
        "files": len(files),
        "corrupt_files": [s["source"] for s in summaries.values() if s["error"]],
        "records_in": sum(s["n_in"] for s in summaries.values()),
        "records_dropped_invalid": sum(s["n_dropped"] for s in summaries.values()),
        "records_valid": sum(len(s["doc_number"]) for s in summaries.values() if s["error"] is None),
        "records_kept": len(best),
        "parse_seconds": round(parse_s, 2),
    }
    stats["duplicates_removed"] = stats["records_valid"] - stats["records_kept"]
    np.savez(os.path.join(work, "keep.npz"), **{str(i): m for i, m in keep.items()})
    with open(os.path.join(work, "prepare.json"), "w") as f:
        json.dump({"order": order, "stats": stats}, f, indent=1)
    log(f"prepare: {stats}")
    return order, keep, stats


def load_prepare(work):
    with open(os.path.join(work, "prepare.json")) as f:
        p = json.load(f)
    keep_npz = np.load(os.path.join(work, "keep.npz"))
    return p["order"], {i: keep_npz[str(i)] for i in p["order"]}, p["stats"]


def kept_chunk_mask(shard, keep):
    counts = np.diff(shard["chunk_indptr"])
    return np.repeat(keep, counts)


# ----------------------------------------------------------------------------------------------- embed
def stage_embed(work, order, keep, model, device, fp16, batch_size):
    from .embed import Embedder

    emb_dir = os.path.join(work, "emb")
    os.makedirs(emb_dir, exist_ok=True)
    todo = [i for i in order if not os.path.exists(os.path.join(emb_dir, f"{i:05d}.npy"))]
    if not todo:
        log("embed: all shards already embedded")
        return {}
    embedder = Embedder(model, device=device, fp16=fp16)
    log(f"embed: {len(todo)} shards to embed with {model} on {embedder.device} (fp16={fp16})")
    t0, n_done = time.time(), 0
    for n, i in enumerate(todo, 1):
        shard = load_shard(os.path.join(work, "shards", f"{i:05d}.pkl"))
        mask = kept_chunk_mask(shard, keep[i])
        texts = [t for t, m in zip(shard["chunk_text"], mask) if m]
        vecs = embedder.encode_passages(texts, batch_size=batch_size).astype(np.float16)
        out = os.path.join(emb_dir, f"{i:05d}.npy")
        np.save(out + ".tmp.npy", vecs)
        os.replace(out + ".tmp.npy", out)
        n_done += len(texts)
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t0
            log(f"  {n}/{len(todo)} shards, {n_done} chunks, {n_done / el:.0f} chunks/s, "
                f"ETA {el / n * (len(todo) - n) / 60:.1f} min")
    el = time.time() - t0
    return {"embed_seconds": round(el, 2), "chunks_embedded": n_done, "chunks_per_second": round(n_done / el, 1)}


# -------------------------------------------------------------------------------------------- assemble
def _bm25_csc(counts_csr):
    """Precompute BM25 term weights so query-time scoring is just summing posting lists."""
    n_docs = counts_csr.shape[0]
    tf = counts_csr.data.astype(np.float32)
    dl = np.asarray(counts_csr.sum(axis=1)).ravel().astype(np.float32)
    avgdl = max(float(dl.mean()), 1.0)
    df = np.bincount(counts_csr.indices, minlength=counts_csr.shape[1]).astype(np.float32)
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    rows = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(counts_csr.indptr))
    w = idf[counts_csr.indices] * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl[rows] / avgdl))
    m = sp.csr_matrix((w.astype(np.float32), counts_csr.indices, counts_csr.indptr), shape=counts_csr.shape).tocsc()
    m.sort_indices()
    return m


def _save_csc(out, name, m, with_data=True):
    np.save(os.path.join(out, f"{name}_indptr.npy"), m.indptr.astype(np.int64))
    np.save(os.path.join(out, f"{name}_indices.npy"), m.indices.astype(np.int32))
    if with_data:
        np.save(os.path.join(out, f"{name}_data.npy"), m.data.astype(np.float32))


def build_ann(vecs_f16, kind, out):
    import faiss

    n, d = vecs_f16.shape
    t = time.time()
    if kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
    elif kind == "ivf":
        nlist = int(min(65536, max(256, 4 * np.sqrt(n))))
        index = faiss.IndexIVFScalarQuantizer(faiss.IndexFlatIP(d), d, nlist, faiss.ScalarQuantizer.QT_8bit,
                                              faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(0)
        sample = vecs_f16[np.sort(rng.choice(n, size=min(n, 50 * nlist), replace=False))].astype(np.float32)
        index.train(sample)
    else:
        raise ValueError(kind)
    step = 200_000
    for s in range(0, n, step):
        index.add(np.ascontiguousarray(vecs_f16[s:s + step], dtype=np.float32))
        log(f"  ann: added {min(n, s + step)}/{n}")
    faiss.write_index(index, os.path.join(out, f"ann_{kind}.faiss"))
    return round(time.time() - t, 2)


def stage_assemble(work, order, keep, out, ann, model):
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    db_path = os.path.join(out, "patents.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)
    db = sqlite3.connect(db_path)
    db.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE patents(id INTEGER PRIMARY KEY, doc_number TEXT, title TEXT, abstract TEXT,
                             classification TEXT, date_lo INTEGER, date_hi INTEGER, claims BLOB, source TEXT);
        CREATE TABLE chunks(id INTEGER PRIMARY KEY, patent_id INTEGER, kind INTEGER, num INTEGER, text TEXT);
    """)

    vocab = {}
    fields = {"T": [], "A": [], "C": []}
    cls_keys, date_lo, date_hi, chunk_counts, chunk_kind = [], [], [], [], []
    emb_parts, pid, cid = [], 0, 0
    for i in order:
        shard = load_shard(os.path.join(work, "shards", f"{i:05d}.pkl"))
        k = keep[i]
        idx = np.flatnonzero(k)
        local_to_global = np.fromiter((vocab.setdefault(t, len(vocab)) for t in shard["vocab"]),
                                      dtype=np.int32, count=len(shard["vocab"]))
        for f_name, (indptr, ids, cnt) in shard["tok"].items():
            lens = np.diff(indptr)
            sel = np.repeat(k, lens)
            fields[f_name].append((lens[k], local_to_global[ids[sel]], cnt[sel]))

        ci = shard["chunk_indptr"]
        cmask = kept_chunk_mask(shard, k)
        chunk_counts.append(np.diff(ci)[k])
        chunk_kind.append(shard["chunk_kind"][cmask])
        cls_keys.extend(shard["cls_key"][j] for j in idx)
        date_lo.append(shard["date_lo"][k])
        date_hi.append(shard["date_hi"][k])

        rows, crow = [], []
        for j in idx:
            rows.append((pid, shard["doc_number"][j], shard["title"][j], shard["abstract"][j],
                         shard["classification"][j], int(shard["date_lo"][j]), int(shard["date_hi"][j]),
                         shard["claims_blob"][j], os.path.basename(shard.get("source", "")) or None))
            for c in range(ci[j], ci[j + 1]):
                crow.append((cid, pid, int(shard["chunk_kind"][c]), int(shard["chunk_num"][c]), shard["chunk_text"][c]))
                cid += 1
            pid += 1
        db.executemany("INSERT INTO patents VALUES (?,?,?,?,?,?,?,?,?)", rows)
        db.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)", crow)
        emb_parts.append(os.path.join(work, "emb", f"{i:05d}.npy"))
    db.execute("CREATE INDEX idx_chunks_patent ON chunks(patent_id)")
    db.execute("CREATE UNIQUE INDEX idx_doc ON patents(doc_number)")
    db.commit()
    db.close()
    n_pat, n_chunks = pid, cid
    log(f"assemble: {n_pat} patents, {n_chunks} chunks, vocab {len(vocab)} written to SQLite ({time.time() - t0:.1f}s)")

    # Vocabulary sorted alphabetically so wildcard keywords ("tire*") are a contiguous id range.
    terms = np.array(list(vocab), dtype=object)
    sort_idx = np.argsort(terms.astype(str))
    remap = np.empty(len(terms), dtype=np.int32)
    remap[sort_idx] = np.arange(len(terms), dtype=np.int32)
    with open(os.path.join(out, "vocab.txt"), "w") as f:
        f.write("\n".join(terms[sort_idx]))

    def field_csr(parts):
        lens = np.concatenate([p[0] for p in parts])
        indptr = np.concatenate([[0], np.cumsum(lens, dtype=np.int64)])
        ids = remap[np.concatenate([p[1] for p in parts])]
        cnt = np.concatenate([p[2] for p in parts]).astype(np.float32)
        return sp.csr_matrix((cnt, ids, indptr), shape=(n_pat, len(terms)))

    t = time.time()
    T, A, C = (field_csr(fields[f]) for f in ("T", "A", "C"))
    del fields
    _save_csc(out, "bm25", _bm25_csc(T + A + C))
    del C
    ta = (T + A).tocsc()
    ta.sort_indices()
    _save_csc(out, "kw_ta", ta, with_data=False)
    tt = T.tocsc()
    tt.sort_indices()
    _save_csc(out, "kw_title", tt, with_data=False)
    del T, A, ta, tt
    log(f"assemble: BM25 + keyword postings built ({time.time() - t:.1f}s)")

    # Classification codes sorted for O(log n) prefix lookups.
    has = np.array([c is not None for c in cls_keys])
    ids_with = np.flatnonzero(has).astype(np.int32)
    keys = np.array([cls_keys[j] for j in ids_with], dtype=str)
    o = np.argsort(keys, kind="stable")
    np.save(os.path.join(out, "cls_keys_sorted.npy"), keys[o])
    np.save(os.path.join(out, "cls_order.npy"), ids_with[o])
    np.save(os.path.join(out, "date_lo.npy"), np.concatenate(date_lo).astype(np.int32))
    np.save(os.path.join(out, "date_hi.npy"), np.concatenate(date_hi).astype(np.int32))
    counts = np.concatenate(chunk_counts)
    np.save(os.path.join(out, "patent_offsets.npy"), np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]))
    np.save(os.path.join(out, "chunk_patent.npy"), np.repeat(np.arange(n_pat, dtype=np.int32), counts))
    np.save(os.path.join(out, "chunk_kind.npy"), np.concatenate(chunk_kind))

    t = time.time()
    d = np.load(emb_parts[0], mmap_mode="r").shape[1]
    vecs = np.lib.format.open_memmap(os.path.join(out, "vectors.npy"), mode="w+", dtype=np.float16, shape=(n_chunks, d))
    pos = 0
    for p in emb_parts:
        v = np.load(p)
        vecs[pos:pos + len(v)] = v
        pos += len(v)
    assert pos == n_chunks, f"vector count {pos} != chunk count {n_chunks} (re-run the embed stage)"
    vecs.flush()
    log(f"assemble: vectors written ({time.time() - t:.1f}s)")

    if ann == "auto":
        ann = "flat" if n_chunks < 300_000 else "hnsw"
    ann_s = build_ann(vecs, ann, out) if ann != "flat" else 0.0
    return {"assemble_seconds": round(time.time() - t0, 2), "ann": ann, "ann_build_seconds": ann_s,
            "patents": n_pat, "chunks": n_chunks, "vocab": len(terms), "dim": int(d), "model": model}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="directory with patent JSON files (searched recursively)")
    ap.add_argument("--out", required=True, help="output index directory")
    ap.add_argument("--work", help="scratch dir for shards/vectors (default: <out>/work)")
    ap.add_argument("--granularity", choices=GRANULARITIES, default="full")
    ap.add_argument("--limit-files", type=int, help="only use the first N files (sorted)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fp32", action="store_true", help="disable fp16 inference")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ann", choices=["auto", "flat", "hnsw", "ivf"], default="auto")
    ap.add_argument("--stages", default="prepare,embed,assemble")
    args = ap.parse_args()

    from .embed import DEFAULT_MODEL
    model = args.model or DEFAULT_MODEL
    work = args.work or os.path.join(args.out, "work")
    os.makedirs(work, exist_ok=True)
    cfg_path = os.path.join(work, "config.json")
    cfg = {"data": os.path.abspath(args.data), "granularity": args.granularity, "limit_files": args.limit_files, "model": model}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            old = json.load(f)
        if old != cfg:
            raise SystemExit(f"work dir {work} was built with {old}; delete it or pass a different --work")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    stages = args.stages.split(",")
    meta_path = os.path.join(args.out, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta.update({"granularity": args.granularity, "model": model, "data": cfg["data"]})
    build = meta.setdefault("build", {})

    if "prepare" in stages:
        files = find_files(args.data, args.limit_files)
        log(f"found {len(files)} files under {args.data}")
        order, keep, stats = stage_prepare(files, work, args.granularity, args.workers)
        build["prepare"] = stats
    else:
        order, keep, _ = load_prepare(work)
    if "embed" in stages:
        r = stage_embed(work, order, keep, model, args.device, not args.fp32, args.batch_size)
        if r:
            build["embed"] = r
    if "assemble" in stages:
        r = stage_assemble(work, order, keep, args.out, args.ann, model)
        build["assemble"] = r
        meta.update({k: r[k] for k in ("patents", "chunks", "vocab", "dim", "ann")})
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    log(f"done -> {args.out}")


if __name__ == "__main__":
    main()
