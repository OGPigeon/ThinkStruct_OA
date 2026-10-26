"""Part 2 proof of concept: a durable job queue + worker pools + live status dashboard for ingestion.

SQLite stands in for Postgres (same SQL: atomic claim with UPDATE ... RETURNING, leases, retries).
Two stages run as a DAG: every finished `parse` job enqueues an `embed` job for its shard, and each stage
has its own worker pool (CPU parsers vs. a single GPU embedder), like the at-scale design.

    python poc/pipeline.py init --data data/large --limit 24          # one parse job per raw file
    python poc/pipeline.py dashboard &                                 # http://127.0.0.1:8001
    python poc/pipeline.py run --parse-workers 4 --embed-workers 1 --chaos 0.15

--chaos p makes each job attempt fail with probability p: half the failures raise (transient error ->
retried with backoff), half hard-kill the worker process mid-job (its lease expires and another worker
picks the job up; the supervisor restarts the dead worker). After max attempts a job goes to 'failed'
(the dead-letter state) with its last error, visible on the dashboard.
"""
import argparse
import json
import os
import random
import socket
import sqlite3
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
STATE = os.path.join(ROOT, "poc", "state")
DB = os.path.join(STATE, "manifest.sqlite")
LEASE_S = 10
MAX_ATTEMPTS = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY, stage TEXT NOT NULL, input TEXT NOT NULL, output TEXT,
  status TEXT NOT NULL DEFAULT 'pending',           -- pending | running | done | failed
  attempts INTEGER NOT NULL DEFAULT 0, not_before REAL NOT NULL DEFAULT 0,
  lease_owner TEXT, lease_expires REAL, error TEXT,
  n_in INTEGER, n_out INTEGER, n_dropped INTEGER,
  created_at REAL, started_at REAL, finished_at REAL,
  UNIQUE(stage, input)                               -- enqueueing is idempotent
);
CREATE TABLE IF NOT EXISTS events (ts REAL, job_id INTEGER, worker TEXT, kind TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS workers (name TEXT PRIMARY KEY, stage TEXT, pid INTEGER, heartbeat REAL, current_job INTEGER, done INTEGER DEFAULT 0);
"""


def connect():
    os.makedirs(STATE, exist_ok=True)
    db = sqlite3.connect(DB, timeout=30, isolation_level=None)  # autocommit; explicit BEGIN IMMEDIATE when needed
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db


def event(db, job_id, worker, kind, detail=""):
    db.execute("INSERT INTO events VALUES (?,?,?,?,?)", (time.time(), job_id, worker, kind, detail[:300]))


def enqueue(db, stage, inp):
    db.execute("INSERT OR IGNORE INTO jobs(stage, input, created_at) VALUES (?,?,?)", (stage, inp, time.time()))


def claim(db, stage, worker):
    """Atomically take one runnable job: pending and past its backoff, or running with an expired lease."""
    now = time.time()
    row = db.execute("""
        UPDATE jobs SET status='running', lease_owner=?, lease_expires=?, attempts=attempts+1, started_at=?
        WHERE id = (SELECT id FROM jobs WHERE stage=? AND
                    ((status='pending' AND not_before<=?) OR (status='running' AND lease_expires<?))
                    ORDER BY id LIMIT 1)
        RETURNING id, input, attempts""", (worker, now + LEASE_S, now, stage, now, now)).fetchone()
    return row


# ------------------------------------------------------------------------------------------- job bodies
class PermanentError(Exception):
    """Retrying cannot help (e.g. corrupt input): dead-letter immediately instead of burning attempts."""


def run_parse(inp, job_id):
    from patent_search.data import process_file
    out = os.path.join(STATE, "shards", f"{job_id:05d}.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    s = process_file((inp, out, "core"))
    if s["error"]:
        raise PermanentError(f"corrupt input: {s['error']}")
    return out, s["n_in"], s["n_in"] - s["n_dropped"], s["n_dropped"]


_EMBEDDER = None


def run_embed(inp, job_id):
    global _EMBEDDER
    import numpy as np
    from patent_search.data import load_shard
    if _EMBEDDER is None:
        from patent_search.embed import Embedder
        _EMBEDDER = Embedder()
    shard = load_shard(inp)
    vecs = _EMBEDDER.encode_passages(shard["chunk_text"]).astype(np.float16)
    out = inp.replace(".pkl", ".emb.npy")
    np.save(out + ".tmp.npy", vecs)
    os.replace(out + ".tmp.npy", out)  # atomic: re-running a job never leaves a partial file
    return out, len(shard["chunk_text"]), len(vecs), 0


BODIES = {"parse": run_parse, "embed": run_embed}
NEXT_STAGE = {"parse": "embed"}


def worker_loop(stage, chaos, idle_exit):
    name = f"{stage}-{socket.gethostname().split('.')[0]}-{os.getpid()}"
    db = connect()
    db.execute("INSERT OR REPLACE INTO workers VALUES (?,?,?,?,NULL,0)", (name, stage, os.getpid(), time.time()))
    current = {"job": None}
    stop = threading.Event()

    def heartbeat():  # keeps the lease alive while a long job runs; a dead process stops renewing it
        hb = connect()
        while not stop.wait(LEASE_S / 4):
            hb.execute("UPDATE workers SET heartbeat=?, current_job=? WHERE name=?", (time.time(), current["job"], name))
            if current["job"] is not None:
                hb.execute("UPDATE jobs SET lease_expires=? WHERE id=? AND lease_owner=?",
                           (time.time() + LEASE_S, current["job"], name))
    threading.Thread(target=heartbeat, daemon=True).start()

    idle_since = time.time()
    while True:
        job = claim(db, stage, name)
        if job is None:
            if time.time() - idle_since > idle_exit:
                break
            time.sleep(0.3)
            continue
        idle_since = time.time()
        job_id, inp, attempts = job
        current["job"] = job_id
        event(db, job_id, name, "start", f"attempt {attempts}")
        try:
            if chaos and random.random() < chaos:
                if random.random() < 0.5:
                    time.sleep(random.random())
                    event(db, job_id, name, "crash", "simulated hard crash (process killed mid-job)")
                    os._exit(1)
                raise RuntimeError("simulated transient error (e.g. S3 timeout)")
            out, n_in, n_out, n_drop = BODIES[stage](inp, job_id)
            db.execute("BEGIN IMMEDIATE")
            db.execute("""UPDATE jobs SET status='done', output=?, n_in=?, n_out=?, n_dropped=?, finished_at=?, error=NULL
                          WHERE id=? AND lease_owner=?""", (out, n_in, n_out, n_drop, time.time(), job_id, name))
            if stage in NEXT_STAGE:
                enqueue(db, NEXT_STAGE[stage], out)
            db.execute("UPDATE workers SET done=done+1 WHERE name=?", (name,))
            db.execute("COMMIT")
            event(db, job_id, name, "done", f"{n_out} items")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if attempts >= MAX_ATTEMPTS or isinstance(e, PermanentError):
                db.execute("UPDATE jobs SET status='failed', error=?, finished_at=? WHERE id=?", (err, time.time(), job_id))
                event(db, job_id, name, "dead-letter", err)
            else:
                backoff = 2 ** attempts * 0.5 + random.random()
                db.execute("UPDATE jobs SET status='pending', error=?, not_before=?, lease_owner=NULL WHERE id=?",
                           (err, time.time() + backoff, job_id))
                event(db, job_id, name, "retry", f"{err}; backoff {backoff:.1f}s")
        current["job"] = None
    stop.set()
    db.execute("DELETE FROM workers WHERE name=?", (name,))


# ------------------------------------------------------------------------------------------- commands
def cmd_init(a):
    from patent_search.build import find_files
    if os.path.exists(DB) and a.reset:
        os.remove(DB)
    db = connect()
    files = find_files(a.data, a.limit)
    for f in files:
        enqueue(db, "parse", os.path.abspath(f))
    if a.corrupt:  # add a deliberately broken file to show dead-lettering
        bad = os.path.join(STATE, "corrupt_example.json")
        with open(bad, "w") as fh:
            fh.write('[{"doc_number": "1", "title": "truncated...')
        enqueue(db, "parse", bad)
    print(f"enqueued {len(files) + bool(a.corrupt)} parse jobs into {DB}")


def cmd_run(a):
    """Supervisor: keeps N workers per stage alive (restarting crashed ones) until the queue drains."""
    connect()
    procs = {}
    pools = {"parse": a.parse_workers, "embed": a.embed_workers}

    def spawn(stage):
        # workers stay up while idle (an embed worker waits for parse output); the supervisor stops them at the end
        return subprocess.Popen([sys.executable, __file__, "worker", "--stage", stage, "--chaos", str(a.chaos),
                                 "--idle-exit", "86400"])
    for stage, n in pools.items():
        for i in range(n):
            procs[(stage, i)] = spawn(stage)
    db = connect()
    while True:
        time.sleep(1)
        active = [st for st, n in pools.items() if n > 0]
        open_jobs = db.execute(f"SELECT COUNT(*) FROM jobs WHERE status IN ('pending','running') AND stage IN "
                               f"({','.join('?' * len(active))})", active).fetchone()[0]
        for key, p in list(procs.items()):
            if p.poll() is not None and open_jobs:
                print(f"supervisor: worker {key} died (exit {p.returncode}), restarting", flush=True)
                procs[key] = spawn(key[0])
        if not open_jobs:
            break
    for p in procs.values():
        p.terminate()
    summary = db.execute("SELECT stage, status, COUNT(*), SUM(n_out) FROM jobs GROUP BY stage, status").fetchall()
    print("queue drained:", summary)


def status(db):
    now = time.time()
    by = {}
    for stage, st, n, items in db.execute("SELECT stage, status, COUNT(*), COALESCE(SUM(n_out),0) FROM jobs GROUP BY stage, status"):
        by.setdefault(stage, {}).update({st: n, f"{st}_items": items})
    t0 = db.execute("SELECT MIN(started_at) FROM jobs").fetchone()[0]
    stages = []
    for stage in ("parse", "embed"):
        s = by.get(stage, {})
        total = sum(s.get(k, 0) for k in ("pending", "running", "done", "failed"))
        done_t = db.execute("SELECT COUNT(*), MIN(started_at), MAX(finished_at) FROM jobs WHERE stage=? AND status='done'", (stage,)).fetchone()
        rate = done_t[0] / max(1e-6, (done_t[2] - done_t[1])) if done_t[0] and done_t[2] else 0
        retries = db.execute("SELECT COALESCE(SUM(attempts-1),0) FROM jobs WHERE stage=? AND attempts>1", (stage,)).fetchone()[0]
        stages.append({"stage": stage, "total": total, "pending": s.get("pending", 0), "running": s.get("running", 0),
                       "done": s.get("done", 0), "failed": s.get("failed", 0), "items": s.get("done_items", 0),
                       "jobs_per_min": round(rate * 60, 1), "retries": retries,
                       "eta_s": round((s.get("pending", 0) + s.get("running", 0)) / rate) if rate else None})
    workers = [dict(zip(("name", "stage", "pid", "age_s", "job", "done"), (r[0], r[1], r[2], round(now - r[3], 1), r[4], r[5])))
               for r in db.execute("SELECT name, stage, pid, heartbeat, current_job, done FROM workers ORDER BY name")]
    events = [dict(zip(("ts", "job", "worker", "kind", "detail"), (time.strftime("%H:%M:%S", time.localtime(r[0])), *r[1:])))
              for r in db.execute("SELECT * FROM events ORDER BY ts DESC LIMIT 25")]
    failed = [dict(zip(("id", "stage", "input", "attempts", "error"), r)) for r in
              db.execute("SELECT id, stage, input, attempts, error FROM jobs WHERE status='failed'")]
    return {"stages": stages, "workers": workers, "events": events, "failed": failed,
            "elapsed_s": round(now - t0, 1) if t0 else 0}


DASH = """<!doctype html><html><head><meta charset="utf-8"><title>Ingestion Status</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
:root{--bg:#f7f7f5;--p:#fff;--ink:#1d1d1b;--mut:#6b6b66;--line:#e3e2dd;--ok:#2e7d32;--run:#2f5bd3;--bad:#c62828;--wait:#9e9e9e}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--p:#1d1d1b;--ink:#ecebe6;--mut:#9b9a94;--line:#34332f;--ok:#81c784;--run:#7c9cff;--bad:#ff8a80;--wait:#6b6b66}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,Segoe UI,sans-serif;padding:18px}
h1{font-size:17px;margin:0 0 14px}h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:20px 0 8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}.card{background:var(--p);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.bar{display:flex;height:10px;border-radius:5px;overflow:hidden;background:var(--line);margin:8px 0}.bar div{height:100%}
.kv{display:flex;gap:14px;flex-wrap:wrap;color:var(--mut)}.kv b{color:var(--ink);font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:4px 6px;border-bottom:1px solid var(--line)}th{color:var(--mut);font-weight:500}
.k-done{color:var(--ok)}.k-retry,.k-crash{color:#e8a200}.k-dead-letter{color:var(--bad);font-weight:600}.k-start{color:var(--mut)}
</style></head><body><h1>Ingestion pipeline <span id="el" style="color:var(--mut);font-weight:400"></span></h1>
<div class="grid" id="stages"></div><h2>Workers</h2><div class="card"><table id="workers"></table></div>
<h2>Dead-lettered jobs</h2><div class="card"><table id="failed"></table></div>
<h2>Recent events</h2><div class="card"><table id="events"></table></div>
<script>
const esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
async function tick(){const r=await fetch("/api/status").then(r=>r.json());
document.getElementById("el").textContent=`· ${r.elapsed_s}s since first job`;
document.getElementById("stages").innerHTML=r.stages.map(s=>{const t=Math.max(1,s.total),w=x=>100*x/t+"%";return `<div class="card"><b>${s.stage}</b>
<div class="bar"><div style="width:${w(s.done)};background:var(--ok)"></div><div style="width:${w(s.running)};background:var(--run)"></div><div style="width:${w(s.failed)};background:var(--bad)"></div></div>
<div class="kv"><span>done <b>${s.done}</b>/${s.total}</span><span>running <b>${s.running}</b></span><span>pending <b>${s.pending}</b></span><span>failed <b>${s.failed}</b></span>
<span>retries <b>${s.retries}</b></span><span><b>${s.items.toLocaleString()}</b> ${s.stage==="parse"?"patents":"vectors"}</span><span><b>${s.jobs_per_min}</b> jobs/min</span><span>ETA <b>${s.eta_s==null?"–":s.eta_s+"s"}</b></span></div></div>`}).join("");
document.getElementById("workers").innerHTML="<tr><th>worker</th><th>stage</th><th>job</th><th>done</th><th>heartbeat</th></tr>"+r.workers.map(w=>`<tr><td>${esc(w.name)}</td><td>${w.stage}</td><td>${w.job??"idle"}</td><td>${w.done}</td><td>${w.age_s}s ago</td></tr>`).join("");
document.getElementById("failed").innerHTML=r.failed.length?"<tr><th>job</th><th>stage</th><th>input</th><th>attempts</th><th>last error</th></tr>"+r.failed.map(f=>`<tr><td>${f.id}</td><td>${f.stage}</td><td>${esc(f.input.split("/").slice(-2).join("/"))}</td><td>${f.attempts}</td><td>${esc(f.error)}</td></tr>`).join(""):"<tr><td>none</td></tr>";
document.getElementById("events").innerHTML=r.events.map(e=>`<tr><td>${e.ts}</td><td>job ${e.job}</td><td>${esc(e.worker)}</td><td class="k-${e.kind}">${e.kind}</td><td>${esc(e.detail)}</td></tr>`).join("");}
tick();setInterval(tick,1000);</script></body></html>"""


def cmd_dashboard(a):
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    app = FastAPI()
    app.get("/")(lambda: HTMLResponse(DASH))
    app.get("/api/status")(lambda: status(connect()))
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--data", default=os.path.join(ROOT, "data", "large"))
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--no-corrupt", dest="corrupt", action="store_false", help="don't add the broken demo file")
    p.add_argument("--keep", dest="reset", action="store_false", help="keep the existing manifest")
    p = sub.add_parser("run")
    p.add_argument("--parse-workers", type=int, default=4)
    p.add_argument("--embed-workers", type=int, default=1)
    p.add_argument("--chaos", type=float, default=0.0)
    p = sub.add_parser("worker")
    p.add_argument("--stage", required=True)
    p.add_argument("--chaos", type=float, default=0.0)
    p.add_argument("--idle-exit", type=float, default=5.0)
    p = sub.add_parser("dashboard")
    p.add_argument("--port", type=int, default=8001)
    p = sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "init":
        cmd_init(a)
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "worker":
        worker_loop(a.stage, a.chaos, a.idle_exit)
    elif a.cmd == "dashboard":
        cmd_dashboard(a)
    elif a.cmd == "status":
        print(json.dumps(status(connect()), indent=1))


if __name__ == "__main__":
    main()
