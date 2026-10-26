"""HTTP API + single-page UI.

  python -m patent_search.server --index small=indexes/small --index large=indexes/large
  open http://127.0.0.1:8000
"""
import argparse
import os
import threading

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .index import PatentIndex

app = FastAPI(title="Patent search")
INDEXES = {}
LOCK = threading.Lock()  # one search at a time: the SQLite handle and the MPS model are shared
STATIC = os.path.join(os.path.dirname(__file__), "static")


class SearchRequest(BaseModel):
    index: str | None = None
    query: str | None = None
    claim: str | None = None
    patent: str | None = None
    filters: dict = {}
    k: int = 10
    mode: str = "hybrid"
    strategy: str = "auto"


@app.get("/")
def home():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/indexes")
def indexes():
    return [{"name": n, "patents": i.n, "chunks": i.n_chunks, "granularity": i.meta.get("granularity"),
             "ann": i.ann_kind} for n, i in INDEXES.items()]


def _get(name):
    idx = INDEXES.get(name or next(iter(INDEXES)))
    if idx is None:
        raise HTTPException(404, f"unknown index {name!r}")
    return idx


@app.post("/api/search")
def search(req: SearchRequest):
    idx = _get(req.index)
    query = req.query
    if not (query or req.claim or req.patent):
        f = req.filters or {}
        query = f.get("title") or f.get("keywords") or f.get("cls")
        query = " ".join(query) if isinstance(query, list) else query
        if not query:
            raise HTTPException(400, "give a query, claim, patent or filter")
    try:
        with LOCK:
            return idx.search(query=query, claim=req.claim, patent=req.patent, filters=req.filters,
                              k=min(req.k, 100), mode=req.mode, strategy=req.strategy)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # e.g. remote patent fetch failed
        raise HTTPException(502, f"{type(e).__name__}: {e}")


@app.get("/api/patent/{index}/{doc}")
def patent(index: str, doc: str):
    idx = _get(index)
    pid = idx.doc_to_id.get(doc)
    if pid is None:
        raise HTTPException(404, "not in index")
    with LOCK:
        return idx.get_patent(pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    specs = a.index or ["small=indexes/small"]
    embedder = None
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            name, path = os.path.basename(spec.rstrip("/")), spec
        idx = PatentIndex(path, embedder=embedder)
        embedder = embedder or idx.embedder  # all indexes share one model (they must use the same one)
        INDEXES[name] = idx
        print(f"loaded {name}: {idx.n} patents, {idx.n_chunks} chunks, ann={idx.ann_kind}")
    embedder.encode_queries(["warm up"])  # first MPS call compiles kernels (~3s); do it before serving
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
