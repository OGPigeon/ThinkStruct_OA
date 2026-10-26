"""Scan the large dataset once and report data-quality issues (corrupt files, missing fields, duplicates)."""
import collections
import glob
import os
import sys
import time
from multiprocessing import Pool

import orjson

FIELDS = ["title", "doc_number", "abstract", "detailed_description", "claims", "classification"]


def scan(path):
    out = {"path": path, "bytes": os.path.getsize(path), "ok": True, "err": None, "n": 0,
           "missing": collections.Counter(), "empty": collections.Counter(), "keys": collections.Counter(),
           "ids": [], "cls": collections.Counter(), "non_dict": 0}
    try:
        with open(path, "rb") as f:
            data = orjson.loads(f.read())
    except Exception as e:
        out["ok"], out["err"] = False, f"{type(e).__name__}: {str(e)[:120]}"
        return out
    if not isinstance(data, list):
        out["ok"], out["err"] = False, f"top-level {type(data).__name__}"
        return out
    for p in data:
        if not isinstance(p, dict):
            out["non_dict"] += 1
            continue
        out["n"] += 1
        out["keys"][tuple(sorted(p))] += 1
        for k in FIELDS:
            v = p.get(k)
            if v is None:
                out["missing"][k] += 1
            elif not v or (isinstance(v, str) and not v.strip()):
                out["empty"][k] += 1
        out["ids"].append(p.get("doc_number"))
        out["cls"][(p.get("classification") or "")[:4]] += 1
    return out


if __name__ == "__main__":
    root = sys.argv[1]
    files = sorted(glob.glob(os.path.join(root, "job_*", "*.json")))
    t = time.time()
    agg = collections.defaultdict(collections.Counter)
    ids = collections.Counter()
    bad, total_bytes, n = [], 0, 0
    with Pool(8) as pool:
        for r in pool.imap_unordered(scan, files):
            total_bytes += r["bytes"]
            if not r["ok"]:
                bad.append((r["path"], r["err"]))
                continue
            n += r["n"]
            for k in ("missing", "empty", "keys", "cls"):
                agg[k].update(r[k])
            agg["non_dict"]["n"] += r["non_dict"]
            ids.update(r["ids"])
    dt = time.time() - t
    print(f"files={len(files)} bytes={total_bytes/1e9:.1f}GB time={dt:.1f}s ({total_bytes/1e6/dt:.0f} MB/s)")
    print(f"patents={n} unique_doc_numbers={len(ids)} duplicated_ids={sum(1 for v in ids.values() if v > 1)} extra_copies={sum(v-1 for v in ids.values() if v > 1)}")
    print("bad files:", len(bad))
    for b in bad:
        print("  ", b)
    print("non-dict entries:", agg["non_dict"]["n"])
    print("missing:", dict(agg["missing"]))
    print("empty:", dict(agg["empty"]))
    print("keysets:", agg["keys"].most_common(6))
    print("top cls4:", agg["cls"].most_common(25))
