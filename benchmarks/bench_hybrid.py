"""Part 1 timing: search latency with and without hybrid filters, naive vs filter-first.

    python benchmarks/bench_hybrid.py --index indexes/small --suite small
    python benchmarks/bench_hybrid.py --index indexes/large --suite large

For every (query, filter set) it measures retrieval latency (query embedding excluded - it is the same
for every variant, ~15-20 ms on MPS) for:
  none          no filters (plain semantic + BM25 hybrid ranking)
  naive         score EVERY patent (exact dense + BM25), sort, then walk the ranking testing each patent's raw
                text/metadata against the filters in Python until k survive (the textbook post-filter)
  filter-first  this engine: filters from posting lists / sorted arrays, then search only the candidates
Results are written to benchmarks/results/hybrid_<suite>.md.
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from patent_search.index import Filters, PatentIndex, _rrf  # noqa: E402
from patent_search.text import normalize_classification, tokenize  # noqa: E402

SUITES = {
    "small": {
        "queries": [
            "tire pressure sensor mounted inside the wheel rim", "airless tire with flexible spokes",
            "tread pattern with circumferential grooves and sipes for wet grip", "wheel hub motor for electric vehicle",
            "lightweight carbon fiber wheel rim", "trailer hitch coupling with ball mount",
            "noise reduction foam attached to tire inner surface", "bicycle wheel spoke tension adjustment",
            "rim with air holes for brake cooling", "tire bead with steel wire core",
            "vehicle wheel cover aerodynamic", "run flat tire insert",
            "rubber composition with silica for tread", "wheel bolt anti-theft lock",
            "tire inflation system on a moving vehicle", "omnidirectional mecanum wheel rollers",
            "self sealing tire puncture", "wheel balancing weight clip",
            "towing vehicle detecting trailer angle", "caster wheel for furniture with brake",
        ],
        "filters": {
            "cls B60B": Filters(cls=["B60B"]),
            "kw sensor*": Filters(keywords=["sensor*"]),
            "cls B60C + kw noise + date": Filters(cls=["B60C"], keywords=["noise"], date_from="2024-06", date_to="2024-12"),
            "title phrase": Filters(title="vehicle wheel"),
        },
    },
    "large": {
        "queries": [
            "tire pressure sensor mounted inside the wheel rim", "lithium ion battery cathode with nickel rich layered oxide",
            "transformer neural network for natural language processing", "semiconductor device with gate all around transistor",
            "wireless communication beam management in 5G", "CRISPR gene editing guide RNA delivery",
            "autonomous vehicle lidar object detection", "blockchain based payment verification",
            "antibody that binds PD-1 for cancer treatment", "augmented reality head mounted display optics",
            "image sensor pixel with photodiode isolation", "drone delivery package release mechanism",
            "solar cell perovskite layer", "medical stent with drug coating",
            "video encoding with motion vector prediction", "electric vehicle charging station power management",
            "memory controller error correction code", "OLED display thin film encapsulation",
            "wind turbine blade pitch control", "smart contract execution on distributed ledger",
        ],
        "filters": {
            "cls H01L": Filters(cls=["H01L"]),
            "cls G06F or H04L": Filters(cls=["G06F", "H04L"]),
            "kw network": Filters(keywords=["network"]),
            "cls H01M + kw electrode* + 2021-22": Filters(cls=["H01M"], keywords=["electrode*"], date_from="2021", date_to="2022"),
            "title phrase": Filters(title="semiconductor device"),
        },
    },
}


class NaiveFilter:
    """Filters evaluated the obvious way: per patent, on raw strings."""

    def __init__(self, idx):
        rows = idx.db.execute("SELECT title, abstract, classification FROM patents ORDER BY id").fetchall()
        self.text = [f"{t or ''} {a or ''}".lower() for t, a, _ in rows]
        self.title = [(t or "").lower() for t, _, _ in rows]
        self.cls = [normalize_classification(c) or "" for _, _, c in rows]
        self.lo, self.hi = idx.date_lo, idx.date_hi

    def ok(self, pid, f, date_rng):
        if f.cls and not any(self.cls[pid].startswith(normalize_classification(c)) for c in f.cls):
            return False
        for kw in f.keywords:  # plain substring test: the cheapest naive check, so speed-ups are conservative
            if kw.lower().rstrip("*") not in self.text[pid]:
                return False
        if f.title and " ".join(tokenize(f.title)) not in " ".join(tokenize(self.title[pid])):
            return False
        if date_rng and not (self.hi[pid] >= date_rng[0] and self.lo[pid] <= date_rng[1]):
            return False
        return True


def naive_search(idx, nf, q, lexical, f, k, flat):
    """Exhaustive scoring of all patents, then post-filter by walking the ranking."""
    from patent_search.index import _parse_date
    sims = flat @ q[0]                                          # every chunk
    per_patent = np.maximum.reduceat(sims, idx.offsets[:-1])    # best chunk per patent
    dense_order = np.argsort(-per_patent)
    bm_ids, _ = idx.bm25_search(lexical, None, idx.n)
    date_rng = None
    if f.date_from or f.date_to:
        date_rng = (_parse_date(f.date_from, False) if f.date_from else 0,
                    _parse_date(f.date_to, True) if f.date_to else 99999999)

    def walk(order, need=100):
        out = []
        for pid in order:
            if nf.ok(int(pid), f, date_rng):
                out.append(int(pid))
                if len(out) == need:
                    break
        return out

    dense, bm = walk(dense_order), walk(bm_ids)
    fused, rrf = _rrf(dense, bm)
    # same result payload as the engine (evidence passages, matched terms) so the comparison is like for like
    return idx._results(fused[:k], q, ["query"], lexical, f, rrf, dense, bm,
                        {p: float(per_patent[p]) for p in dense})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--suite", choices=SUITES, required=True)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    suite = SUITES[a.suite]
    idx = PatentIndex(a.index)
    idx.embedder.encode_queries(["warm up"])
    Q = idx.embedder.encode_queries(suite["queries"])
    t = time.perf_counter()
    for qtext in suite["queries"]:
        idx.embedder.encode_queries([qtext])
    embed_ms = (time.perf_counter() - t) * 1000 / len(suite["queries"])
    nf = NaiveFilter(idx)
    flat = idx.flat32  # exhaustive scoring needs fp32 vectors in RAM (built once, not timed)

    def timed(fn):
        best = []
        for q in range(len(suite["queries"])):
            runs = []
            for _ in range(a.repeat):
                t0 = time.perf_counter()
                fn(q)
                runs.append((time.perf_counter() - t0) * 1000)
            best.append(min(runs))
        return best

    rows = []
    base = timed(lambda q: idx.search(query=suite["queries"][q], Q=Q[q:q + 1], k=a.k))
    rows.append(("none (no filters)", "-", base, base, "-"))
    for name, f in suite["filters"].items():
        n_pass = len(idx.filter_ids(f))
        plans = set()

        def ff(q):
            r = idx.search(query=suite["queries"][q], Q=Q[q:q + 1], filters=f, k=a.k)
            plans.add(r["plan"].split("(")[0])
        fast = timed(ff)
        slow = timed(lambda q: naive_search(idx, nf, Q[q:q + 1], suite["queries"][q], f, a.k, flat))
        # sanity: both strategies must agree on what is allowed
        r = idx.search(query=suite["queries"][0], Q=Q[0:1], filters=f, k=a.k)
        allowed = set(idx.filter_ids(f).tolist())
        assert all(x["id"] in allowed for x in r["results"])
        rows.append((name, f"{n_pass} ({100 * n_pass / idx.n:.2f}%)", slow, fast, ",".join(sorted(plans))))

    med = lambda xs: statistics.median(xs)
    p95 = lambda xs: sorted(xs)[max(0, int(round(0.95 * len(xs))) - 1)]
    lines = [f"# Hybrid search timing - `{a.index}` ({idx.n} patents, {idx.n_chunks} chunks, ann={idx.ann_kind})", "",
             f"{len(suite['queries'])} queries x best of {a.repeat} runs; retrieval only (query embedding "
             f"excluded: {embed_ms:.1f} ms/query, identical for all variants). Times in ms.", "",
             "| filter | patents passing | naive post-filter median / p95 | filter-first median / p95 | speed-up (median) | plan |",
             "|---|---|---|---|---|---|"]
    for name, n_pass, slow, fast, plan in rows:
        if name.startswith("none"):
            lines.append(f"| **{name}** | all | - | {med(fast):.2f} / {p95(fast):.2f} | baseline | {idx.ann_kind if idx.ann else 'exact'} |")
        else:
            lines.append(f"| {name} | {n_pass} | {med(slow):.2f} / {p95(slow):.2f} | {med(fast):.2f} / {p95(fast):.2f} "
                         f"| {med(slow) / med(fast):.1f}x | {plan} |")
    report = "\n".join(lines)
    print(report)
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results", f"hybrid_{a.suite}.md"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
