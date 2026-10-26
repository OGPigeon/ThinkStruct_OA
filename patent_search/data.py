"""Loading, cleaning and chunking raw patent JSON files.

Missing-field policy (applies to both the small and the large dataset):
  * no doc_number                       -> record dropped (it cannot be identified, deduplicated or cited)
  * no title, no abstract and no claims -> record dropped (nothing to search)
  * any other missing / empty field     -> record KEPT; the field becomes "" / [] / None and simply contributes
                                           nothing (e.g. a patent with an empty detailed description is still
                                           searchable by title, abstract and claims; a patent with classification
                                           "N/A" is searchable but never matches a classification filter)
  * duplicated doc_number               -> keep the most complete copy (most claim + description text)

Each input file is turned into one "shard" (a pickle of column arrays) by `process_file`, which is
designed to run in a worker process: it does all the per-record CPU work (JSON parsing, cleaning,
chunking, tokenizing) so that only compact arrays cross the process boundary.
"""
import os
import pickle
import re
import zlib

import numpy as np
import orjson

from .text import normalize_classification, parse_claims, tidy, tokenize, unglue_numerals

KIND_ABSTRACT, KIND_CLAIM, KIND_DESC = 0, 1, 2
KIND_NAMES = {KIND_ABSTRACT: "abstract", KIND_CLAIM: "claim", KIND_DESC: "description"}

# What gets embedded per patent. 'full' is affordable for the small dataset (~50 chunks/patent);
# 'core' is what fits a laptop for the ~460k-patent dataset (~4 chunks/patent).
GRANULARITIES = ("full", "core", "abstract")

DESC_MIN_PARA = 40      # shorter paragraphs are headings ("BRIEF DESCRIPTION OF THE DRAWINGS")
DESC_TARGET = 400       # merge consecutive paragraphs until a chunk has at least this many chars
DESC_MAX = 1200         # split longer paragraphs at sentence boundaries (~256 model tokens)
UNKNOWN_DATE = (99999999, 0)  # (lo, hi) that never overlaps any date range
_PLACEHOLDERS = {"N/A", "NA", "NONE", "NULL"}

_FILE_DATE_RE = re.compile(r"-(\d{8})\.XML$", re.I)
_SENT_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z])")


def record_dates(rec):
    """(date_lo, date_hi) as yyyymmdd ints. Exact publication date from the small dataset's filename
    (US20240051333A1-20240215.XML); the large dataset only gives the year via the doc number."""
    m = _FILE_DATE_RE.search(rec.get("filename") or "")
    if m:
        d = int(m.group(1))
        return d, d
    doc = rec.get("doc_number") or ""
    if len(doc) >= 4 and doc[:4].isdigit() and 1970 <= int(doc[:4]) <= 2100:
        y = int(doc[:4])
        return y * 10000 + 101, y * 10000 + 1231
    return UNKNOWN_DATE


def _split_long(text):
    if len(text) <= DESC_MAX:
        return [text]
    out, buf = [], ""
    for sent in _SENT_SPLIT_RE.split(text):
        if buf and len(buf) + len(sent) > DESC_MAX:
            out.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}" if buf else sent
    if buf:
        out.append(buf)
    return out


def description_chunks(paragraphs):
    chunks, buf = [], ""
    for para in paragraphs or []:
        if not isinstance(para, str):
            continue
        para = tidy(unglue_numerals(para))
        if len(para) < DESC_MIN_PARA:
            continue
        buf = f"{buf} {para}" if buf else para
        if len(buf) >= DESC_TARGET:
            chunks.extend(_split_long(buf))
            buf = ""
    if buf:
        chunks.extend(_split_long(buf))
    return chunks


def normalize_record(raw, granularity):
    """Clean one raw record. Returns None if it must be dropped (see module docstring)."""
    if not isinstance(raw, dict):
        return None
    doc = str(raw.get("doc_number") or "").strip()
    if not doc:
        return None
    title = tidy(raw.get("title") or "") if isinstance(raw.get("title"), str) else ""
    abstract = tidy(raw.get("abstract") or "") if isinstance(raw.get("abstract"), str) else ""
    # the large dataset uses the literal string "N/A" as a placeholder (9.3% of titles): treat as missing
    title = "" if title.upper() in _PLACEHOLDERS else title
    abstract = "" if abstract.upper() in _PLACEHOLDERS else abstract
    raw_claims = raw.get("claims") if isinstance(raw.get("claims"), list) else []
    claims = parse_claims(raw_claims)
    if not (title or abstract or claims):
        return None
    desc = raw.get("detailed_description") if isinstance(raw.get("detailed_description"), list) else []

    code = raw.get("classification") if isinstance(raw.get("classification"), str) else None
    cls_key = normalize_classification(code)
    date_lo, date_hi = record_dates(raw)

    chunks = []
    head = f"{title}. {abstract}" if title and abstract else (title or abstract)
    if head:
        chunks.append((KIND_ABSTRACT, 0, head))
    if granularity in ("full", "core"):
        for num, text, indep in claims:
            if granularity == "full" or indep:
                chunks.append((KIND_CLAIM, num, text))
    if granularity == "full":
        for i, text in enumerate(description_chunks(desc)):
            chunks.append((KIND_DESC, i, text))

    quality = sum(len(t) for _, t, _ in claims) + sum(len(p) for p in desc if isinstance(p, str))
    return {
        "doc_number": doc,
        "title": title,
        "abstract": abstract,
        "classification": code if cls_key else None,
        "cls_key": cls_key,
        "date_lo": date_lo,
        "date_hi": date_hi,
        "claims": claims,
        "chunks": chunks,
        "quality": quality,
    }


class _Vocab:
    def __init__(self):
        self.ids = {}

    def encode(self, tokens):
        """Unique token ids + counts for one field of one document."""
        counts = {}
        for t in tokens:
            i = self.ids.get(t)
            if i is None:
                i = self.ids[t] = len(self.ids)
            counts[i] = counts.get(i, 0) + 1
        return counts


def _pack(per_doc_counts):
    indptr = np.zeros(len(per_doc_counts) + 1, dtype=np.int64)
    np.cumsum([len(c) for c in per_doc_counts], out=indptr[1:])
    ids = np.fromiter((i for c in per_doc_counts for i in c.keys()), dtype=np.int32, count=int(indptr[-1]))
    cnt = np.fromiter((min(n, 65535) for c in per_doc_counts for n in c.values()), dtype=np.uint16, count=int(indptr[-1]))
    return indptr, ids, cnt


def load_json_file(path):
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list, got {type(data).__name__}")
    return data


def process_file(job):
    """Worker entry point: raw JSON file -> shard pickle. Returns a small summary for the parent.

    job = (src_path, shard_path, granularity). Never raises: a corrupt file is reported via 'error'.
    """
    src, shard_path, granularity = job
    summary = {"source": src, "shard": shard_path, "n_in": 0, "n_dropped": 0, "error": None,
               "doc_number": [], "quality": []}
    try:
        raw = load_json_file(src)
    except Exception as e:  # corrupt / truncated file: skip it, report it, keep going
        summary["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return summary

    vocab = _Vocab()
    cols = {k: [] for k in ("doc_number", "title", "abstract", "classification", "cls_key",
                            "date_lo", "date_hi", "quality", "claims_blob")}
    chunk_counts, chunk_kind, chunk_num, chunk_text = [], [], [], []
    tok = {"T": [], "A": [], "C": []}
    summary["n_in"] = len(raw)
    for item in raw:
        try:
            rec = normalize_record(item, granularity)
        except Exception:
            rec = None
        if rec is None:
            summary["n_dropped"] += 1
            continue
        for k in ("doc_number", "title", "abstract", "classification", "cls_key", "date_lo", "date_hi", "quality"):
            cols[k].append(rec[k])
        cols["claims_blob"].append(zlib.compress(orjson.dumps(rec["claims"]), 3))
        chunk_counts.append(len(rec["chunks"]))
        for kind, num, text in rec["chunks"]:
            chunk_kind.append(kind)
            chunk_num.append(num)
            chunk_text.append(text)
        tok["T"].append(vocab.encode(tokenize(rec["title"])))
        tok["A"].append(vocab.encode(tokenize(rec["abstract"])))
        tok["C"].append(vocab.encode(tokenize(" ".join(t for _, t, _ in rec["claims"]))))

    shard = {k: v for k, v in cols.items() if k not in ("date_lo", "date_hi", "quality")}
    shard["source"] = src
    shard["date_lo"] = np.asarray(cols["date_lo"], dtype=np.int32)
    shard["date_hi"] = np.asarray(cols["date_hi"], dtype=np.int32)
    shard["quality"] = np.asarray(cols["quality"], dtype=np.int64)
    shard["chunk_indptr"] = np.concatenate([[0], np.cumsum(chunk_counts, dtype=np.int64)])
    shard["chunk_kind"] = np.asarray(chunk_kind, dtype=np.int8)
    shard["chunk_num"] = np.asarray(chunk_num, dtype=np.int32)
    shard["chunk_text"] = chunk_text
    shard["vocab"] = list(vocab.ids)  # dicts preserve insertion order, so position == local id
    shard["tok"] = {k: _pack(v) for k, v in tok.items()}

    tmp = shard_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, shard_path)  # atomic: a crash never leaves a half-written shard behind
    summary["doc_number"] = cols["doc_number"]
    summary["quality"] = cols["quality"]
    return summary


def load_shard(path):
    with open(path, "rb") as f:
        return pickle.load(f)
