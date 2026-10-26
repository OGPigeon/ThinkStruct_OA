"""Command-line search.

  python -m patent_search.cli "tire pressure sensor inside the rim"
  python -m patent_search.cli "airless tire" --cls B60C --kw spoke* --from 2024-06
  python -m patent_search.cli --title "vehicle wheel" --title-exact "lightweight rim"
  python -m patent_search.cli --patent US20240051333A1          # claim mapping for a patent (index or Google Patents)
  python -m patent_search.cli --claim "a hub body having a first end ..." --mode dense
"""
import argparse
import json
import re
import sys
import textwrap

from .index import Filters, PatentIndex


def highlight(text, terms, enable):
    if not enable or not terms:
        return text
    pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\w*", re.I)
    return pat.sub(lambda m: f"\033[1;33m{m.group(0)}\033[0m", text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="?", help="natural-language query")
    ap.add_argument("--index", default="indexes/small")
    ap.add_argument("--patent", help="publication number; its independent claims become the queries")
    ap.add_argument("--claim", help="claim text to map against the corpus")
    ap.add_argument("--cls", action="append", default=[], help="classification prefix (repeatable, OR-ed)")
    ap.add_argument("--kw", action="append", default=[], help="keyword required in title/abstract (repeatable, AND-ed; 'tire*' ok)")
    ap.add_argument("--exclude", action="append", default=[], help="keyword that must NOT be in title/abstract")
    ap.add_argument("--title", help="phrase that must appear in the title")
    ap.add_argument("--title-exact", action="store_true", help="title must equal --title")
    ap.add_argument("--from", dest="date_from")
    ap.add_argument("--to", dest="date_to")
    ap.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    ap.add_argument("--strategy", choices=["auto", "exact", "postfilter", "filtered"], default="auto")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    a = ap.parse_args(argv)

    filters = Filters(cls=a.cls, keywords=a.kw, exclude=a.exclude, title=a.title, title_exact=a.title_exact,
                      date_from=a.date_from, date_to=a.date_to)
    query = a.query
    if not (query or a.patent or a.claim):
        if not filters.active():
            ap.error("give a query, --patent, --claim or at least one filter")
        query = a.title or " ".join(a.kw) or " ".join(a.cls)  # filter-only search: rank by the filter terms

    idx = PatentIndex(a.index)
    r = idx.search(query=query, claim=a.claim, patent=a.patent, filters=filters, k=a.k, mode=a.mode, strategy=a.strategy)
    if a.json:
        print(json.dumps(r, indent=1))
        return
    color = sys.stdout.isatty()
    src = f" | query patent: {r['source']['doc_number']} {r['source']['title']!r}" if r["source"] else ""
    passed = "" if r["n_filtered"] is None else f" | {r['n_filtered']} pass filters"
    print(f"{len(r['results'])} results | {r['n_patents']} patents{passed} | plan {r['plan']}{src}")
    print("timings (ms): " + ", ".join(f"{k[:-3]} {v}" for k, v in r["timings"].items()))
    for x in r["results"]:
        terms = set(x["matched_terms"])
        print(f"\n{x['rank']:>2}. {x['doc_number']}  {x['title']}")
        print(f"    {x['classification'] or 'no class'} | {x['date'] or 'no date'} | dense #{x['dense_rank'] or '-'}"
              f" ({x['dense_score']}) | bm25 #{x['bm25_rank'] or '-'} | matched: {', '.join(x['matched_terms'][:8]) or '-'}")
        for e in x["evidence"]:
            body = textwrap.shorten(e["text"], 260)
            print(f"    [{e['query']} -> {e['kind']} {e['num']} | sim {e['score']}] {highlight(body, terms, color)}")


if __name__ == "__main__":
    main()
