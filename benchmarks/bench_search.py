"""Part 3: vector-search options on the large index - latency, recall and memory - and filtered-search plans.

    python benchmarks/bench_search.py --index indexes/large [--ivf]

1. Unfiltered: exact brute force vs HNSW (several efSearch) vs IVF-SQ8 (several nprobe).
   Recall@10 is measured at PATENT level (the unit users see) against exact search.
2. Filtered: for filters of decreasing selectivity, compare the plans the engine can choose from
   (prefilter-exact, ANN + IDSelector bitmap, ANN + over-fetch post-filter) on latency and recall vs.
   exact filtered results. This is what the planner thresholds in index.py are based on.
Ground truth is computed block-wise from the fp16 vectors so memory stays bounded on a 16 GB laptop.
Writes benchmarks/results/search.md (and prints rows as they are produced).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench_hybrid import SUITES  # noqa: E402
from patent_search.index import Filters, PatentIndex  # noqa: E402

N_PAT = 100   # patents per query the fusion stage consumes
K = 10        # what is evaluated
BLOCK = 200_000
EXACT_GATHER_MAX = 400_000  # don't time prefilter-exact above this many chunks (the planner never picks it there)


def recall(pred, truth):
    return len(set(pred[:K]) & set(truth[:K])) / max(1, min(K, len(truth)))


def exact_truth(idx, Q, cand=None):
    """Exact top-N_PAT patents per query: all chunk scores block-wise from fp16, per-patent max, optional mask."""
    S = np.empty((len(Q), idx.n_chunks), dtype=np.float32)
    for s in range(0, idx.n_chunks, BLOCK):
        S[:, s:s + BLOCK] = Q @ idx.vectors[s:s + BLOCK].astype(np.float32).T
    P = np.maximum.reduceat(S, idx.offsets[:-1], axis=1)
    if cand is not None:
        mask = np.ones(idx.n, bool)
        mask[cand] = False
        P[:, mask] = -np.inf
    out = []
    for row in P:
        top = np.argpartition(-row, N_PAT)[:N_PAT]
        top = top[np.isfinite(row[top])]
        out.append(top[np.argsort(-row[top])].tolist())
    return out


def timed(fn, Q, repeat):
    out, times = [], []
    for q in range(len(Q)):
        best = None
        for _ in range(repeat):
            t = time.perf_counter()
            r = fn(Q[q:q + 1])
            dt = (time.perf_counter() - t) * 1000
            best = dt if best is None else min(best, dt)
        out.append(r)
        times.append(best)
    return out, float(np.median(times)), float(np.percentile(times, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="indexes/large")
    ap.add_argument("--ivf", action="store_true", help="also build + test an IVF-SQ8 index (~2-4 min)")
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()
    import faiss

    lines = []

    def emit(line=""):
        lines.append(line)
        print(line, flush=True)

    idx = PatentIndex(a.index)
    queries = SUITES["large"]["queries"]
    rng = np.random.default_rng(0)
    sample = rng.choice(idx.n, 30, replace=False)
    abstracts = [(idx.db.execute("SELECT abstract FROM patents WHERE id=?", (int(p),)).fetchone()[0] or "")[:600] for p in sample]
    Q = np.vstack([idx.embedder.encode_queries(queries), idx.embedder.encode_passages(abstracts)])
    to_list = lambda d: list(d.keys())
    truth = exact_truth(idx, Q)

    emit(f"# Vector search benchmark - `{a.index}` ({idx.n} patents, {idx.n_chunks} vectors x {idx.dim}d)")
    emit()
    emit(f"{len(Q)} queries (20 natural-language + 30 patent abstracts used as queries), batch size 1, best of "
         f"{a.repeat}; recall@{K} at patent level vs exact search; dense retrieval only (ms).")
    emit()
    emit("## Unfiltered")
    emit()
    emit("| method | index memory | build | median | p95 | recall@10 |")
    emit("|---|---|---|---|---|---|")

    # exact brute force (fp32 copy in RAM, one matmul + reduceat over per-patent chunk ranges)
    ann = idx.ann
    idx.ann, idx.ann_kind = None, "flat"
    t = time.perf_counter()
    flat = idx.flat32
    load_s = time.perf_counter() - t
    res, med, p95 = timed(lambda q: to_list(idx.dense_search(q, None, N_PAT)[0]), Q, a.repeat)
    rec = np.mean([recall(r, t_) for r, t_ in zip(res, truth)])
    emit(f"| exact brute force (numpy fp32) | {flat.nbytes / 1e9:.2f} GB | {load_s:.1f}s | {med:.1f} | {p95:.1f} | {rec:.3f} |")
    idx._flat32 = flat = None

    idx.ann, idx.ann_kind = ann, "hnsw"
    hnsw_gb = os.path.getsize(os.path.join(a.index, "ann_hnsw.faiss")) / 1e9
    hnsw_build = idx.meta.get("build", {}).get("assemble", {}).get("ann_build_seconds", "?")
    orig = idx._ann_search
    for ef in (32, 64, 128, 256):
        def ann_ef(Qr, k, sel=None, ef=ef):
            params = faiss.SearchParametersHNSW(sel=sel, efSearch=max(ef, int(k)))
            sc, ids = idx.ann.search(np.ascontiguousarray(Qr, dtype=np.float32), int(k), params=params)
            return ids, sc
        idx._ann_search = ann_ef
        res, med, p95 = timed(lambda q: to_list(idx.dense_search(q, None, N_PAT)[0]), Q, a.repeat)
        rec = np.mean([recall(r, t_) for r, t_ in zip(res, truth)])
        emit(f"| HNSW M=32, efSearch=max({ef}, k) | {hnsw_gb:.2f} GB | {hnsw_build}s | {med:.1f} | {p95:.1f} | {rec:.3f} |")
    idx._ann_search = orig

    if a.ivf:
        nlist = int(4 * np.sqrt(idx.n_chunks))
        t = time.perf_counter()
        ivf = faiss.IndexIVFScalarQuantizer(faiss.IndexFlatIP(idx.dim), idx.dim, nlist, faiss.ScalarQuantizer.QT_8bit,
                                            faiss.METRIC_INNER_PRODUCT)
        train = np.sort(rng.choice(idx.n_chunks, min(idx.n_chunks, 40 * nlist), replace=False))
        ivf.train(idx.vectors[train].astype(np.float32))
        for s in range(0, idx.n_chunks, BLOCK):
            ivf.add(idx.vectors[s:s + BLOCK].astype(np.float32))
        ivf_build = time.perf_counter() - t
        ivf_gb = (idx.n_chunks * (idx.dim + 8) + nlist * idx.dim * 4) / 1e9
        idx.ann, idx.ann_kind = ivf, "ivf"
        for nprobe in (8, 32, 128):
            def ann_np(Qr, k, sel=None, nprobe=nprobe):
                params = faiss.SearchParametersIVF(sel=sel, nprobe=nprobe)
                sc, ids = ivf.search(np.ascontiguousarray(Qr, dtype=np.float32), int(k), params=params)
                return ids, sc
            idx._ann_search = ann_np
            res, med, p95 = timed(lambda q: to_list(idx.dense_search(q, None, N_PAT)[0]), Q, a.repeat)
            rec = np.mean([recall(r, t_) for r, t_ in zip(res, truth)])
            emit(f"| IVF{nlist}-SQ8 (int8), nprobe={nprobe} | ~{ivf_gb:.2f} GB | {ivf_build:.0f}s | {med:.1f} | {p95:.1f} | {rec:.3f} |")
        idx._ann_search = orig
        idx.ann, idx.ann_kind = ann, "hnsw"
        del ivf

    # ---------------------------------------------------------------------------------- filtered plans
    emit()
    emit("## Filtered: which plan wins at which selectivity (HNSW, efSearch=max(128, k))")
    emit()
    emit("| filter | patents passing | chunks passing | plan | median | p95 | recall@10 vs exact filtered |")
    emit("|---|---|---|---|---|---|---|")
    filters = dict(SUITES["large"]["filters"])
    filters["kw data (broad)"] = Filters(keywords=["data"])
    filters["year >= 2010 (very broad)"] = Filters(date_from="2010")
    for name, f in filters.items():
        cand = idx.filter_ids(f)
        n_cids = len(idx.chunk_ids_for(cand))
        truth_f = exact_truth(idx, Q, cand)
        auto_plan = idx.dense_search(Q[:1], cand, N_PAT, strategy="auto")[1].split("(")[0]
        for strat, label in (("exact", "prefilter-exact"), ("filtered", "filtered-ann"), ("postfilter", "postfilter-ann")):
            if strat == "exact" and n_cids > EXACT_GATHER_MAX:
                emit(f"| {name} | {len(cand)} ({100 * len(cand) / idx.n:.2f}%) | {n_cids} | {label} | skipped (> {EXACT_GATHER_MAX} chunks) | | |")
                continue
            res, med, p95 = timed(lambda q: to_list(idx.dense_search(q, cand, N_PAT, strategy=strat)[0]), Q, 1)
            rec = np.mean([recall(r, t_) for r, t_ in zip(res, truth_f)])
            star = " **(auto)**" if label == auto_plan else ""
            emit(f"| {name} | {len(cand)} ({100 * len(cand) / idx.n:.2f}%) | {n_cids} | {label}{star} | {med:.1f} | {p95:.1f} | {rec:.3f} |")

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results", "search.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
