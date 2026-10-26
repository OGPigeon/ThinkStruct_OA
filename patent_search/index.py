"""Hybrid patent search engine: metadata filters + BM25 + dense vectors, fused with reciprocal rank fusion.

Query execution is *filter-first*:
  1. Filters (classification prefix, title/abstract keywords, title phrase, date range) are answered from
     precomputed structures - sorted arrays (binary search) and inverted posting lists - and intersected
     smallest-first into a candidate set of patent ids. This never touches document text.
  2. Dense search is planned on the candidate set's size ("selectivity"):
       - few candidate chunks  -> exact brute force over just those vectors (cheaper than an unfiltered search)
       - many candidate chunks -> ANN search restricted by a faiss IDSelector bitmap
       - almost everything     -> ANN search with over-fetch, then drop non-candidates
  3. BM25 sums precomputed term weights over posting lists, keeping only candidate rows.
  4. Patent-level rankings are fused with RRF; evidence passages are computed exactly for the final top-k.
"""
import bisect
import json
import os
import re
import sqlite3
import time
import zlib
from dataclasses import dataclass, field

import numpy as np
import orjson

from .data import KIND_NAMES
from .text import normalize_classification, tokenize

RRF_K = 60
# Candidate chunks up to which the planner brute-forces exactly instead of using filtered HNSW. Calibrated on
# the 1.67M-vector index (benchmarks/results/search.md): exact costs ~10-40 ms up to 300k chunks with recall 1.0,
# while bitmap-filtered HNSW drops to 0.4-0.9 recall@10 when only 3-15% of chunks pass (the graph walk runs out
# of allowed neighbours); from ~35% selectivity upward filtered HNSW is back at 0.995 recall in 6-12 ms.
PREFILTER_EXACT_MAX = 300_000
POSTFILTER_MIN_SELECTIVITY = 0.5  # above this fraction of chunks, over-fetch + post-filter is cheapest
MAX_WILDCARD_TERMS = 1000
MAX_OVERFETCH = 20_000            # cap for post-filter over-fetch: HNSW cost grows with k (efSearch >= k)


@dataclass
class Filters:
    cls: list = field(default_factory=list)        # classification prefixes, OR-ed ("B60B", "H01L21/")
    keywords: list = field(default_factory=list)   # every keyword must appear in title or abstract; "tire*" ok
    exclude: list = field(default_factory=list)    # none of these may appear in title or abstract
    title: str = None                              # phrase that must appear in the title (case-insensitive)
    title_exact: bool = False                      # title must equal `title` exactly (case-insensitive)
    date_from: str = None                          # "2024", "2024-03", "2024-03-15"
    date_to: str = None

    def active(self):
        return bool(self.cls or self.keywords or self.exclude or self.title or self.date_from or self.date_to)

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        as_list = lambda v: [x for x in (v if isinstance(v, list) else str(v).split(",")) if str(x).strip()] if v else []
        return cls(cls=as_list(d.get("cls")), keywords=as_list(d.get("keywords")), exclude=as_list(d.get("exclude")),
                   title=d.get("title") or None, title_exact=bool(d.get("title_exact")),
                   date_from=d.get("date_from") or None, date_to=d.get("date_to") or None)


def _parse_date(s, end):
    digits = re.sub(r"\D", "", str(s))
    if len(digits) == 4:
        return int(digits + ("1231" if end else "0101"))
    if len(digits) == 6:
        return int(digits + ("31" if end else "01"))
    if len(digits) == 8:
        return int(digits)
    raise ValueError(f"bad date {s!r}; use YYYY, YYYY-MM or YYYY-MM-DD")


def _rrf(*rankings):
    scores = {}
    for ranking in rankings:
        for r, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (RRF_K + r + 1)
    return sorted(scores, key=scores.get, reverse=True), scores


class PatentIndex:
    def __init__(self, path, embedder=None, ann=None, mmap=True):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self._embedder = embedder
        L = lambda name, mm=mmap: np.load(os.path.join(path, f"{name}.npy"), mmap_mode="r" if mm else None)

        self.db = sqlite3.connect(os.path.join(path, "patents.sqlite"), check_same_thread=False)
        rows = self.db.execute("SELECT doc_number, title, classification FROM patents ORDER BY id").fetchall()
        self.doc_numbers = [r[0] for r in rows]
        self.titles = [r[1] or "" for r in rows]
        self.classifications = [r[2] for r in rows]
        self.doc_to_id = {d: i for i, d in enumerate(self.doc_numbers)}
        self.n = len(rows)
        self._title_norm = None

        self.date_lo, self.date_hi = L("date_lo", False), L("date_hi", False)
        self.cls_keys, self.cls_order = L("cls_keys_sorted", False), L("cls_order", False)
        self.offsets = L("patent_offsets", False)
        self.chunk_patent = L("chunk_patent", False)
        self.chunk_kind = L("chunk_kind", False)
        self.vectors = L("vectors", False)  # float16, L2-normalized
        self.n_chunks, self.dim = self.vectors.shape

        self.bm25 = (L("bm25_indptr", False), L("bm25_indices"), L("bm25_data"))
        self.kw_ta = (L("kw_ta_indptr", False), L("kw_ta_indices"))
        self.kw_title = (L("kw_title_indptr", False), L("kw_title_indices"))
        with open(os.path.join(path, "vocab.txt")) as f:
            self.vocab = f.read().split("\n")
        self.term_id = {t: i for i, t in enumerate(self.vocab)}

        self.ann_kind = ann or self.meta.get("ann", "flat")
        self.ann = None
        self._flat32 = None
        if self.ann_kind != "flat":
            import faiss
            p = os.path.join(path, f"ann_{self.ann_kind}.faiss")
            self.ann = faiss.read_index(p)
            if self.ann_kind == "hnsw":
                self.ann.hnsw.efSearch = 128
            elif self.ann_kind == "ivf":
                self.ann.nprobe = 32

    # ----------------------------------------------------------------------------------------- helpers
    @property
    def embedder(self):
        if self._embedder is None:
            from .embed import Embedder
            self._embedder = Embedder(self.meta.get("model"))
        return self._embedder

    @property
    def flat32(self):
        if self._flat32 is None:
            self._flat32 = self.vectors.astype(np.float32)
        return self._flat32

    def _postings(self, index, term_ids):
        indptr, indices = index[0], index[1]
        if len(term_ids) == 1:
            t = term_ids[0]
            return np.asarray(indices[indptr[t]:indptr[t + 1]])
        parts = [np.asarray(indices[indptr[t]:indptr[t + 1]]) for t in term_ids]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int32)

    def _term_ids(self, token):
        """Exact term or a 'prefix*' wildcard -> vocabulary ids (vocab is sorted, so a wildcard is a range)."""
        if token.endswith("*"):
            prefix = token[:-1]
            lo = bisect.bisect_left(self.vocab, prefix)
            hi = bisect.bisect_left(self.vocab, prefix + "￿")
            return list(range(lo, min(hi, lo + MAX_WILDCARD_TERMS)))
        t = self.term_id.get(token)
        return [] if t is None else [t]

    def _keyword_ids(self, keyword, index):
        """Patents whose field contains every token of `keyword` (multi-word keywords are AND-ed)."""
        wildcard = keyword.strip().endswith("*")
        toks = tokenize(keyword)
        if not toks:
            return None  # only stopwords -> no constraint
        if wildcard:
            toks[-1] += "*"
        sets = []
        for tok in toks:
            ids = self._term_ids(tok)
            if not ids:
                return np.zeros(0, np.int32)
            sets.append(self._postings(index, ids))
        return _intersect(sets)

    def chunk_ids_for(self, pids):
        """Contiguous chunk ranges of the given patents, concatenated (vectorized np.arange-of-ranges)."""
        starts, ends = self.offsets[pids], self.offsets[np.asarray(pids) + 1]
        lens = ends - starts
        total = int(lens.sum())
        if total == 0:
            return np.zeros(0, np.int64)
        rel = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
        return np.repeat(starts, lens) + rel

    # ----------------------------------------------------------------------------------------- filters
    def filter_ids(self, f: Filters):
        """Sorted int array of patent ids passing every filter, or None when no filter is active."""
        if f is None or not f.active():
            return None
        sets = []
        if f.cls:
            parts = []
            for prefix in f.cls:
                key = normalize_classification(prefix) or ""
                lo = np.searchsorted(self.cls_keys, key, "left")
                hi = np.searchsorted(self.cls_keys, key + "￿", "left")
                parts.append(self.cls_order[lo:hi])
            sets.append(np.unique(np.concatenate(parts)))
        for kw in f.keywords:
            s = self._keyword_ids(kw, self.kw_ta)
            if s is not None:
                sets.append(s)
        if f.title:
            sets.append(self._title_ids(f.title, f.title_exact))
        if f.date_from or f.date_to:
            lo = _parse_date(f.date_from, False) if f.date_from else 0
            hi = _parse_date(f.date_to, True) if f.date_to else 99999999
            sets.append(np.flatnonzero((self.date_hi >= lo) & (self.date_lo <= hi)))
        out = _intersect(sets) if sets else np.arange(self.n)
        for kw in f.exclude:
            s = self._keyword_ids(kw, self.kw_ta)
            if s is not None and len(s):
                out = np.setdiff1d(out, s, assume_unique=True)
        return out.astype(np.int64)

    def _title_ids(self, title, exact):
        toks = tokenize(title)
        if not toks:
            return np.zeros(0, np.int32)
        sets = []
        for tok in toks:
            ids = self._term_ids(tok)
            if not ids:
                return np.zeros(0, np.int32)
            sets.append(self._postings(self.kw_title, ids))
        cand = _intersect(sets)
        # Posting lists give "all words present"; verify word order (or full equality) on the few survivors.
        want = " ".join(toks)
        if self._title_norm is None:
            self._title_norm = {}
        keep = []
        for pid in cand:
            norm = self._title_norm.get(pid)
            if norm is None:
                norm = self._title_norm[pid] = " ".join(tokenize(self.titles[pid]))
            if (norm == want) if exact else (f" {want} " in f" {norm} "):
                keep.append(pid)
        return np.asarray(keep, dtype=np.int64)

    # -------------------------------------------------------------------------------------------- dense
    def dense_search(self, Q, cand, n_patents, strategy="auto", exclude=None):
        """Patent-level semantic scores {patent_id: score} for the best `n_patents`, plus the plan used.

        A patent's score is its best chunk's cosine similarity, averaged over query rows (several claims
        as queries -> patents matching many of them rank first). Exact plans compute this for every
        candidate with one matmul + np.maximum.reduceat over the per-patent chunk ranges; ANN plans
        approximate it from the top chunk hits.
        """
        if cand is not None and len(cand) == 0:
            return {}, "empty"
        if cand is None:
            if self.ann is None or strategy == "exact":
                return self._top_patents(self._patent_max(Q @ self.flat32.T, self.offsets), None, n_patents, exclude), "exact-all"
            return self._aggregate(*self._ann_search(Q, self._chunk_k(n_patents)), n_patents, exclude), f"ann-{self.ann_kind}"

        cids = self.chunk_ids_for(cand)
        selectivity = len(cids) / self.n_chunks
        if self.ann is None and selectivity > 0.25:
            # Exact index and a broad filter: one contiguous matmul, then mask at patent level, beats gathering.
            P = self._patent_max(Q @ self.flat32.T, self.offsets)
            allowed = np.zeros(self.n, bool)
            allowed[cand] = True
            P[~allowed] = -np.inf
            return self._top_patents(P, None, n_patents, exclude), f"exact-masked(sel={selectivity:.2f})"
        if strategy == "exact" or self.ann is None or (strategy == "auto" and len(cids) <= PREFILTER_EXACT_MAX):
            sub = self.vectors[cids].astype(np.float32) if self.ann is not None else self.flat32[cids]
            lens = self.offsets[cand + 1] - self.offsets[cand]
            local_offsets = np.concatenate([[0], np.cumsum(lens)])
            P = self._patent_max(Q @ sub.T, local_offsets)
            return self._top_patents(P, cand, n_patents, exclude), f"prefilter-exact({len(cids)} chunks)"
        chunk_k = self._chunk_k(n_patents)
        if strategy == "postfilter" or (strategy == "auto" and selectivity >= POSTFILTER_MIN_SELECTIVITY):
            over = int(min(self.n_chunks, MAX_OVERFETCH, chunk_k / max(selectivity, 1e-3) * 1.5))
            ids, sc = self._ann_search(Q, over)
            allowed = np.zeros(self.n, bool)
            allowed[cand] = True
            ok = (ids >= 0) & allowed[self.chunk_patent[np.clip(ids, 0, None)]]
            ids, sc = _compact(ids, sc, ok, chunk_k)
            return self._aggregate(ids, sc, n_patents, exclude), f"postfilter-ann(sel={selectivity:.2f})"
        import faiss
        bitmap = np.zeros(self.n_chunks, dtype=bool)
        bitmap[cids] = True
        sel = faiss.IDSelectorBitmap(self.n_chunks, faiss.swig_ptr(np.packbits(bitmap, bitorder="little")))
        ids, sc = self._ann_search(Q, chunk_k, sel)
        return self._aggregate(ids, sc, n_patents, exclude), f"filtered-ann(sel={selectivity:.2f})"

    def _chunk_k(self, n_patents):
        """Chunk hits to request from the ANN index so ~n_patents distinct patents survive grouping."""
        per_patent = max(1.0, self.n_chunks / self.n)
        return int(min(self.n_chunks, max(200, n_patents * per_patent * 2)))

    @staticmethod
    def _patent_max(S, offsets):
        """[m, chunks] similarities -> [n_patents] best chunk per patent, averaged over the m query rows."""
        lens = np.diff(offsets)
        P = np.maximum.reduceat(S, np.minimum(offsets[:-1], S.shape[1] - 1), axis=1).mean(axis=0)
        P[lens == 0] = -np.inf  # reduceat returns a neighbour's value for empty ranges
        return P

    @staticmethod
    def _top_patents(P, ids, n, exclude):
        """Top-n of patent scores P (indexed by position; `ids` maps positions to patent ids if given)."""
        ids = np.arange(len(P)) if ids is None else ids
        if exclude is not None:
            P = np.where(ids == exclude, -np.inf, P)
        n = min(n, len(P))
        top = np.argpartition(-P, n - 1)[:n]
        top = top[np.isfinite(P[top])]
        return {int(ids[i]): float(P[i]) for i in top[np.argsort(-P[top])]}

    def _ann_search(self, Q, k, sel=None):
        import faiss
        k = int(min(k, self.n_chunks))
        if self.ann_kind == "hnsw":
            params = faiss.SearchParametersHNSW(sel=sel, efSearch=max(128, k)) if sel is not None or k > 128 else None
        else:
            params = faiss.SearchParametersIVF(sel=sel, nprobe=32) if sel is not None else None
        sc, ids = self.ann.search(np.ascontiguousarray(Q, dtype=np.float32), k, params=params)
        return ids, sc

    # --------------------------------------------------------------------------------------------- bm25
    def bm25_search(self, text, cand, k):
        terms = sorted({t for tok in tokenize(text) for t in self._term_ids(tok)})
        if not terms:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        indptr, indices, data = self.bm25
        rows = np.concatenate([np.asarray(indices[indptr[t]:indptr[t + 1]]) for t in terms])
        w = np.concatenate([np.asarray(data[indptr[t]:indptr[t + 1]]) for t in terms])
        if cand is not None:
            allowed = np.zeros(self.n, bool)
            allowed[cand] = True
            keep = allowed[rows]
            rows, w = rows[keep], w[keep]
        scores = np.bincount(rows, weights=w, minlength=self.n)
        nz = np.flatnonzero(scores)
        if len(nz) > k:
            nz = nz[np.argpartition(-scores[nz], k)[:k]]
        nz = nz[np.argsort(-scores[nz])]
        return nz, scores[nz]

    # ------------------------------------------------------------------------------------------- search
    def resolve_query(self, query=None, claim=None, patent=None):
        """Turn the user's input into (query vectors, lexical text, labels for each query row, patent to exclude)."""
        if patent:
            doc = re.sub(r"[^0-9]", "", str(patent).upper().removeprefix("US"))[:11] or str(patent)
            pid = self.doc_to_id.get(doc)
            if pid is not None:
                p = self.get_patent(pid)
            else:
                from .fetch import fetch_patent
                p = fetch_patent(patent)
            claims = [c for c in p["claims"] if c[2]] or p["claims"]
            claims = claims[:10]
            texts = [c[1] for c in claims] or [f"{p['title']}. {p['abstract']}"]
            labels = [f"claim {c[0]}" for c in claims] or ["abstract"]
            lexical = " ".join([p["title"], p["abstract"]] + [c[1] for c in claims])
            return self.embedder.encode_passages(texts), lexical, labels, pid, p
        if claim:
            return self.embedder.encode_passages([claim]), claim, ["claim"], None, None
        return self.embedder.encode_queries([query]), query, ["query"], None, None

    def search(self, query=None, claim=None, patent=None, filters=None, k=10, mode="hybrid",
               strategy="auto", n_candidates=100, Q=None):
        t = {}
        t0 = time.perf_counter()
        f = filters if isinstance(filters, Filters) else Filters.from_dict(filters)
        cand = self.filter_ids(f)
        t["filter_ms"] = _ms(t0)

        t1 = time.perf_counter()
        if Q is None:
            Q, lexical, labels, self_pid, source = self.resolve_query(query, claim, patent)
        else:
            lexical, labels, self_pid, source = query or claim or "", ["query"] * len(Q), None, None
        t["embed_ms"] = _ms(t1)
        if self_pid is not None and cand is not None:
            cand = cand[cand != self_pid]

        dense_rank, bm25_rank, dense_score, plan = [], [], {}, "-"
        if mode in ("dense", "hybrid"):
            t2 = time.perf_counter()
            dense_score, plan = self.dense_search(Q, cand, n_candidates, strategy, exclude=self_pid)
            dense_rank = list(dense_score)  # already sorted best-first
            t["dense_ms"] = _ms(t2)
        if mode in ("bm25", "hybrid"):
            t3 = time.perf_counter()
            ids, scores = self.bm25_search(lexical, cand, n_candidates + 1)
            bm25_rank = [int(i) for i in ids if i != self_pid][:n_candidates]
            t["bm25_ms"] = _ms(t3)

        fused, rrf = _rrf(*[r for r in (dense_rank, bm25_rank) if r])
        top = fused[:k]
        t4 = time.perf_counter()
        results = self._results(top, Q, labels, lexical, f, rrf, dense_rank, bm25_rank, dense_score)
        t["evidence_ms"] = _ms(t4)
        t["total_ms"] = _ms(t0)
        t["retrieval_ms"] = round(t["total_ms"] - t["embed_ms"], 2)
        return {"results": results, "timings": t, "plan": plan,
                "n_filtered": None if cand is None else int(len(cand)), "n_patents": self.n,
                "query_rows": labels, "source": _brief(source)}

    def _aggregate(self, ids, sc, n_patents, exclude=None):
        """ANN chunk hits -> patent scores (best chunk per query row, averaged over rows; a row that did not
        retrieve the patent contributes 0)."""
        m = ids.shape[0]
        acc = {}
        for r in range(m):
            valid = ids[r] >= 0
            if not valid.any():
                continue
            pids, s = self.chunk_patent[ids[r][valid]], sc[r][valid]
            order = np.argsort(-s, kind="stable")
            uniq, first = np.unique(pids[order], return_index=True)  # first hit per patent = its best chunk
            for pid, v in zip(uniq.tolist(), (s[order][first] / m).tolist()):
                acc[pid] = acc.get(pid, 0.0) + v
        acc.pop(exclude, None)
        return dict(sorted(acc.items(), key=lambda kv: kv[1], reverse=True)[:n_patents])

    def _results(self, top, Q, labels, lexical, f, rrf, dense_rank, bm25_rank, dense_score):
        if not top:
            return []
        q_terms = set(tokenize(lexical)) | {t for kw in f.keywords for t in tokenize(kw.rstrip("*"))}
        d_pos = {p: i + 1 for i, p in enumerate(dense_rank)}
        b_pos = {p: i + 1 for i, p in enumerate(bm25_rank)}
        out = []
        for rank, pid in enumerate(top, 1):
            cids = self.chunk_ids_for(np.array([pid]))
            sims = Q @ self.vectors[cids].astype(np.float32).T  # [m, n_chunks_of_patent]
            best = sims.argmax(axis=1)
            chunk_rows = {int(cids[b]) for b in best}
            texts = dict(self.db.execute(
                f"SELECT id, kind || '|' || num || '|' || text FROM chunks WHERE id IN ({','.join('?' * len(chunk_rows))})",
                list(chunk_rows)).fetchall())
            evidence = []
            for r, b in enumerate(best):
                kind, num, text = texts[int(cids[b])].split("|", 2)
                evidence.append({"query": labels[r], "kind": KIND_NAMES[int(kind)],
                                 "num": int(num), "score": round(float(sims[r, b]), 4), "text": text})
            abstract = self.db.execute("SELECT abstract FROM patents WHERE id=?", (int(pid),)).fetchone()[0]
            hay = set(tokenize(" ".join([self.titles[pid], abstract] + [e["text"] for e in evidence])))
            out.append({
                "rank": rank, "id": int(pid), "doc_number": self.doc_numbers[pid], "title": self.titles[pid],
                "classification": self.classifications[pid], "date": self._date_str(pid),
                "abstract": abstract, "score": round(rrf.get(pid, 0.0), 5),
                "dense_score": round(dense_score[pid], 4) if pid in dense_score else None,
                "dense_rank": d_pos.get(pid), "bm25_rank": b_pos.get(pid),
                "matched_terms": sorted(q_terms & hay), "evidence": evidence,
            })
        return out

    def _date_str(self, pid):
        lo, hi = int(self.date_lo[pid]), int(self.date_hi[pid])
        if lo > hi:
            return None
        if lo == hi:
            return f"{lo // 10000}-{lo // 100 % 100:02d}-{lo % 100:02d}"
        return str(lo // 10000)

    def get_patent(self, pid):
        row = self.db.execute("SELECT doc_number, title, abstract, classification, claims FROM patents WHERE id=?",
                              (int(pid),)).fetchone()
        return {"id": int(pid), "doc_number": row[0], "title": row[1], "abstract": row[2], "classification": row[3],
                "date": self._date_str(pid), "claims": [tuple(c) for c in orjson.loads(zlib.decompress(row[4]))]}


def _intersect(sets):
    sets = sorted(sets, key=len)
    out = np.asarray(sets[0])
    for s in sets[1:]:
        if len(out) == 0:
            break
        out = np.intersect1d(out, s, assume_unique=True)
    return out


def _compact(ids, sc, ok, k):
    out_i = np.full((ids.shape[0], k), -1, dtype=np.int64)
    out_s = np.full((ids.shape[0], k), -np.inf, dtype=np.float32)
    for r in range(ids.shape[0]):
        keep = np.flatnonzero(ok[r])[:k]
        out_i[r, :len(keep)], out_s[r, :len(keep)] = ids[r, keep], sc[r, keep]
    return out_i, out_s


def _ms(t):
    return round((time.perf_counter() - t) * 1000, 2)


def _brief(p):
    if not p:
        return None
    return {"doc_number": p.get("doc_number"), "title": p.get("title"),
            "claims_used": len([c for c in p.get("claims", []) if c[2]][:10]) or None}
