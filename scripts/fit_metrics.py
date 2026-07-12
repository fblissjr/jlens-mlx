"""Offline, additive fit-metrics analytics tool.

Reads a fit run's progress log + JSON checkpoints (written under jlens-mlx's out/
directory by the long-running background fit job) and ingests them into a shared
DuckDB store for analysis. This is a standalone script: stdlib + duckdb only. It is
NOT imported by the running fit process anywhere, and never touches any fit artifact
other than reading it (the DuckDB file it creates/updates lives alongside them under
out/, which is gitignored).

Schema:
    dim_run          -- one row per fit run (upserted on run_id).
    fact_item_fit     -- one row per (run, item), append-only.
    v_peak_vs_seq     -- peak memory vs. sequence length, per fact row.
    v_throughput      -- throughput aggregated per chunk_size.

Usage:
    python scripts/fit_metrics.py --out out/band-n14-fixed
    python scripts/fit_metrics.py --out out/band-n14-fixed --db /path/to/fit_metrics.duckdb
    python scripts/fit_metrics.py --out out/band-n14-fixed --query peak_vs_seq
    python scripts/fit_metrics.py --out out/band-n14-fixed --query throughput

Run (this repo's venv lacks duckdb -- use heylookitsanllm's venv, which has it):
    cd /Users/fredbliss/workspace/heylookitsanllm && uv run python /Users/fredbliss/workspace/jlens-mlx/scripts/fit_metrics.py --out /Users/fredbliss/workspace/jlens-mlx/out/band-n14-fixed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

RECORD_SOURCE = "fit_metrics.py"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DB_PATH = _REPO_ROOT / "out" / "fit_metrics.duckdb"

# Optional leading "[HH:MM:SS] " prefix -- some logs lack it.
_TS_PREFIX = r"(?:\[\d{2}:\d{2}:\d{2}\]\s+)?"

_HEADER_RE = re.compile(
    _TS_PREFIX + r"load\s+\S+\s+n_layers=(?P<n_layers>\d+)\s+d_model=(?P<d_model>\d+)",
)

_HEARTBEAT_RE = re.compile(
    _TS_PREFIX + r"item\s+(?P<item_index>\d+)/(?P<n_items>\d+)\s+chunk\s+\d+/(?P<n_chunks>\d+)\s+\(",
)

_DONE_RE = re.compile(
    _TS_PREFIX
    + r"item\s+(?P<item_index>\d+)/(?P<n_items>\d+)\s+done\s+in\s+(?P<wall_time_s>\d+(?:\.\d+)?)s"
    + r"\s+\((?P<n_positions>\d+)\s+pos,\s+(?P<sec_per_pos>\d+(?:\.\d+)?)\s+s/pos\)"
    + r".*?peak\s+(?P<peak_gb>\d+(?:\.\d+)?)GB",
)

_MODEL_RE = re.compile(r"JLENS_MODEL[=:\s]+(?P<path>\S+)")

_DEFAULT_D_MODEL = 5120


# --- surrogate keys --------------------------------------------------------------------------

def compute_run_id(out_dir: Path) -> str:
    """Deterministic surrogate key: md5 hex of the out-dir basename."""
    return hashlib.md5(out_dir.name.encode()).hexdigest()


def compute_item_fit_key(run_id: str, item_index: int) -> str:
    """Deterministic surrogate key: md5 hex of f"{run_id}:{item_index}"."""
    return hashlib.md5(f"{run_id}:{item_index}".encode()).hexdigest()


# --- log parsing ------------------------------------------------------------------------------

def parse_log_header(log_text: str) -> dict[str, Any]:
    """Extracts header fields (currently just d_model) from the log text.

    Returns {} (all fields absent) if no header line is found or it doesn't parse --
    callers fall back to _DEFAULT_D_MODEL in that case.
    """
    m = _HEADER_RE.search(log_text)
    if not m:
        return {}
    return {"d_model": int(m.group("d_model"))}


def parse_model_path(log_text: str) -> str | None:
    """Best-effort JLENS_MODEL path extraction -> basename. None if not present."""
    m = _MODEL_RE.search(log_text)
    if not m:
        return None
    return Path(m.group("path")).name


def _find_chunk_size_and_n_chunks(
    lines: list[str], done_line_idx: int, item_index: int, d_model: int
) -> tuple[int | None, int | None]:
    """Scans backward from done_line_idx for the nearest heartbeat matching item_index.

    Returns (chunk_size, n_chunks) derived from that heartbeat's n_chunks, or (None, None)
    if no matching heartbeat is found (caller falls back to ckpt.json's chunk_size).
    """
    for i in range(done_line_idx - 1, -1, -1):
        m = _HEARTBEAT_RE.search(lines[i])
        if m and int(m.group("item_index")) == item_index:
            n_chunks = int(m.group("n_chunks"))
            if n_chunks <= 0:
                return None, None
            chunk_size = round(d_model / n_chunks)
            return chunk_size, n_chunks
    return None, None


def parse_done_lines(log_text: str, d_model: int | None = None) -> list[dict[str, Any]]:
    """Extracts one record per "done" line, with per-item chunk_size/n_chunks derived
    from the nearest preceding heartbeat line for that same item index.
    """
    if d_model is None:
        header = parse_log_header(log_text)
        d_model = header.get("d_model", _DEFAULT_D_MODEL)

    lines = log_text.splitlines()
    records: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        m = _DONE_RE.search(line)
        if not m:
            continue
        item_index = int(m.group("item_index"))
        chunk_size, n_chunks = _find_chunk_size_and_n_chunks(lines, idx, item_index, d_model)
        records.append(
            {
                "item_index": item_index,
                "n_items": int(m.group("n_items")),
                "wall_time_s": float(m.group("wall_time_s")),
                "n_positions": int(m.group("n_positions")),
                "sec_per_pos": float(m.group("sec_per_pos")),
                "peak_gb": float(m.group("peak_gb")),
                "chunk_size": chunk_size,
                "n_chunks": n_chunks,
            }
        )
    return records


# --- json artifact loading ----------------------------------------------------------------------

def load_corpus(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / "ckpt" / "corpus.json").read_text())


def load_ckpt(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / "ckpt" / "ckpt.json").read_text())


# --- git sha ----------------------------------------------------------------------------------

def get_git_sha() -> str | None:
    """Best-effort short git SHA of this repo (jlens-mlx). None on any failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


# --- schema ------------------------------------------------------------------------------------

_DIM_RUN_DDL = """
CREATE TABLE IF NOT EXISTS dim_run (
    run_id TEXT PRIMARY KEY,
    out_dir TEXT,
    model TEXT,
    band_start INTEGER,
    band_end INTEGER,
    target INTEGER,
    n_items INTEGER,
    enable_thinking BOOLEAN,
    max_seq_len INTEGER,
    shared_fraction_overall DOUBLE,
    shared_fraction_onpolicy DOUBLE,
    dropped_over_len INTEGER,
    use_chain BOOLEAN,
    git_sha TEXT,
    record_source TEXT,
    inserted_at TIMESTAMP
)
"""

_FACT_ITEM_FIT_DDL = """
CREATE TABLE IF NOT EXISTS fact_item_fit (
    item_fit_key TEXT PRIMARY KEY,
    run_id TEXT,
    item_index INTEGER,
    seq_len INTEGER,
    n_positions INTEGER,
    stratum TEXT,
    on_policy BOOLEAN,
    chunk_size INTEGER,
    n_chunks INTEGER,
    wall_time_s DOUBLE,
    sec_per_pos DOUBLE,
    sec_per_chunk DOUBLE,
    peak_gb DOUBLE,
    record_source TEXT,
    inserted_at TIMESTAMP
)
"""

_V_PEAK_VS_SEQ_DDL = """
CREATE OR REPLACE VIEW v_peak_vs_seq AS
SELECT run_id, item_index, seq_len, chunk_size, peak_gb, on_policy, stratum
FROM fact_item_fit
ORDER BY seq_len
"""

_V_THROUGHPUT_DDL = """
CREATE OR REPLACE VIEW v_throughput AS
SELECT
    chunk_size,
    count(*) AS n,
    round(avg(sec_per_pos), 2) AS avg_sec_per_pos,
    round(avg(sec_per_chunk), 2) AS avg_sec_per_chunk,
    round(avg(peak_gb), 1) AS avg_peak_gb
FROM fact_item_fit
GROUP BY chunk_size
ORDER BY chunk_size
"""

_DIM_RUN_MUTABLE_FIELDS = [
    "out_dir",
    "model",
    "band_start",
    "band_end",
    "target",
    "n_items",
    "enable_thinking",
    "max_seq_len",
    "shared_fraction_overall",
    "shared_fraction_onpolicy",
    "dropped_over_len",
    "use_chain",
    "git_sha",
    "record_source",
    "inserted_at",
]


def ensure_schema(con) -> None:
    con.execute(_DIM_RUN_DDL)
    con.execute(_FACT_ITEM_FIT_DDL)
    con.execute(_V_PEAK_VS_SEQ_DDL)
    con.execute(_V_THROUGHPUT_DDL)


# --- ingest ------------------------------------------------------------------------------------

def build_dim_run_row(out_dir: Path, log_text: str, corpus: dict, ckpt: dict) -> dict[str, Any]:
    run_id = compute_run_id(out_dir)
    provenance = corpus.get("provenance", {})
    diversity = provenance.get("diversity", {})
    on_policy_diversity = diversity.get("on_policy", {})

    layers = ckpt.get("layers") or []
    model_name = parse_model_path(log_text) or "unknown"

    return {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "model": model_name,
        "band_start": min(layers) if layers else None,
        "band_end": max(layers) if layers else None,
        "target": ckpt.get("target"),
        "n_items": ckpt.get("n_total"),
        "enable_thinking": provenance.get("enable_thinking"),
        "max_seq_len": provenance.get("max_seq_len"),
        "shared_fraction_overall": diversity.get("shared_fraction"),
        "shared_fraction_onpolicy": on_policy_diversity.get("shared_fraction"),
        "dropped_over_len": provenance.get("dropped_over_len"),
        "use_chain": ckpt.get("use_chain"),
        "git_sha": get_git_sha(),
        "record_source": RECORD_SOURCE,
        "inserted_at": datetime.now(timezone.utc),
    }


def build_fact_rows(
    out_dir: Path, log_text: str, corpus: dict, ckpt: dict
) -> list[dict[str, Any]]:
    run_id = compute_run_id(out_dir)
    header = parse_log_header(log_text)
    d_model = header.get("d_model", _DEFAULT_D_MODEL)
    fallback_chunk_size = ckpt.get("chunk_size")

    items = corpus.get("items", [])
    done_records = parse_done_lines(log_text, d_model=d_model)

    rows = []
    now = datetime.now(timezone.utc)
    for rec in done_records:
        item_index = rec["item_index"]
        corpus_item = items[item_index - 1] if 0 <= item_index - 1 < len(items) else {}

        chunk_size = rec["chunk_size"]
        n_chunks = rec["n_chunks"]
        if chunk_size is None:
            chunk_size = fallback_chunk_size

        sec_per_chunk = (
            rec["wall_time_s"] / n_chunks if n_chunks else None
        )

        rows.append(
            {
                "item_fit_key": compute_item_fit_key(run_id, item_index),
                "run_id": run_id,
                "item_index": item_index,
                "seq_len": len(corpus_item.get("ids", [])) if corpus_item else None,
                "n_positions": rec["n_positions"],
                "stratum": corpus_item.get("stratum"),
                "on_policy": corpus_item.get("on_policy"),
                "chunk_size": chunk_size,
                "n_chunks": n_chunks,
                "wall_time_s": rec["wall_time_s"],
                "sec_per_pos": rec["sec_per_pos"],
                "sec_per_chunk": sec_per_chunk,
                "peak_gb": rec["peak_gb"],
                "record_source": RECORD_SOURCE,
                "inserted_at": now,
            }
        )
    return rows


def upsert_dim_run(con, row: dict[str, Any]) -> None:
    columns = ["run_id"] + _DIM_RUN_MUTABLE_FIELDS
    placeholders = ", ".join("?" for _ in columns)
    set_clause = ", ".join(f"{c} = excluded.{c}" for c in _DIM_RUN_MUTABLE_FIELDS)
    sql = (
        f"INSERT INTO dim_run ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT (run_id) DO UPDATE SET {set_clause}"
    )
    con.execute(sql, [row[c] for c in columns])


def insert_fact_rows(con, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    sql = (
        f"INSERT INTO fact_item_fit ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT (item_fit_key) DO NOTHING"
    )
    for row in rows:
        con.execute(sql, [row[c] for c in columns])


def ingest(out_dir: Path, db_path: Path) -> dict[str, Any]:
    """Ingests one fit run's artifacts into the shared DuckDB store. Idempotent.

    Returns a small summary dict: {"run_id": ..., "fact_count": ...}.
    """
    out_dir = out_dir.resolve()
    log_path = out_dir.parent / f"{out_dir.name}.log"
    log_text = log_path.read_text() if log_path.exists() else ""
    corpus = load_corpus(out_dir)
    ckpt = load_ckpt(out_dir)

    dim_row = build_dim_run_row(out_dir, log_text, corpus, ckpt)
    fact_rows = build_fact_rows(out_dir, log_text, corpus, ckpt)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        ensure_schema(con)
        upsert_dim_run(con, dim_row)
        insert_fact_rows(con, fact_rows)
        fact_count = con.execute(
            "SELECT count(*) FROM fact_item_fit WHERE run_id = ?", [dim_row["run_id"]]
        ).fetchone()[0]
    finally:
        con.close()

    return {"run_id": dim_row["run_id"], "fact_count": fact_count}


# --- CLI ---------------------------------------------------------------------------------------

def run_query(db_path: Path, query_name: str) -> None:
    view_by_name = {"peak_vs_seq": "v_peak_vs_seq", "throughput": "v_throughput"}
    view = view_by_name.get(query_name)
    if view is None:
        raise SystemExit(f"unknown --query {query_name!r}; expected one of {sorted(view_by_name)}")

    con = duckdb.connect(str(db_path))
    try:
        ensure_schema(con)
        df = con.execute(f"SELECT * FROM {view}").df()
        print(df.to_string(index=False))
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", required=False, help="Path to a fit run's out dir (e.g. out/band-n14-fixed)"
    )
    parser.add_argument("--db", default=str(_DEFAULT_DB_PATH), help="Path to the shared DuckDB store")
    parser.add_argument(
        "--query",
        choices=["peak_vs_seq", "throughput"],
        default=None,
        help="Print a view's contents and exit without ingesting",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)

    if args.query:
        run_query(db_path, args.query)
        return 0

    if not args.out:
        parser.error("--out is required unless --query is given")

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir

    summary = ingest(out_dir=out_dir, db_path=db_path)
    print(f"run_id: {summary['run_id']}")
    print(f"fact rows for this run: {summary['fact_count']}")

    con = duckdb.connect(str(db_path))
    try:
        df = con.execute(
            "SELECT * FROM v_peak_vs_seq WHERE run_id = ?", [summary["run_id"]]
        ).df()
        print(df.to_string(index=False))
    finally:
        con.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
