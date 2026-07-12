"""Unit + smoke tests for scripts/fit_metrics_ui.py -- the read-only fit-metrics dashboard.

Builds a tiny fixture DuckDB (`dim_run` + `fact_item_fit` + the `v_peak_vs_seq`/`v_throughput`
views, matching the exact schema scripts/fit_metrics.py is expected to produce) in `tmp_path`,
spins up the real handler on an ephemeral port, and hits each endpoint over HTTP. NEVER touches
the real out/fit_metrics.duckdb (a live fit may be writing to it in the main checkout).

Run:  uv run pytest tests/test_fit_metrics_ui.py -q
"""
from __future__ import annotations

import datetime
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.fit_metrics_ui import (  # noqa: E402
    build_handler, fetch_peak_vs_seq, fetch_runs, fetch_throughput, open_ro,
)


def _build_fixture_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    try:
        con.execute("""
            CREATE TABLE dim_run (
                run_id VARCHAR, out_dir VARCHAR, model VARCHAR, band_start INTEGER,
                band_end INTEGER, target INTEGER, n_items INTEGER, enable_thinking BOOLEAN,
                max_seq_len INTEGER, shared_fraction_overall DOUBLE, shared_fraction_onpolicy DOUBLE,
                dropped_over_len INTEGER, use_chain BOOLEAN, git_sha VARCHAR, inserted_at TIMESTAMP
            )
        """)
        con.execute("""
            CREATE TABLE fact_item_fit (
                item_fit_key VARCHAR, run_id VARCHAR, item_index INTEGER, seq_len INTEGER,
                n_positions INTEGER, stratum VARCHAR, on_policy BOOLEAN, chunk_size INTEGER,
                n_chunks INTEGER, wall_time_s DOUBLE, sec_per_pos DOUBLE, sec_per_chunk DOUBLE,
                peak_gb DOUBLE, inserted_at TIMESTAMP
            )
        """)
        con.execute("""
            CREATE VIEW v_peak_vs_seq AS
            SELECT run_id, item_index, seq_len, chunk_size, peak_gb, on_policy, stratum
            FROM fact_item_fit
        """)
        con.execute("""
            CREATE VIEW v_throughput AS
            SELECT chunk_size, count(*) AS n, avg(sec_per_pos) AS avg_sec_per_pos,
                   avg(sec_per_chunk) AS avg_sec_per_chunk, avg(peak_gb) AS avg_peak_gb
            FROM fact_item_fit GROUP BY chunk_size
        """)

        now = datetime.datetime(2026, 7, 12, 15, 0, 0)
        con.execute(
            "INSERT INTO dim_run VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["run-a", "out/band-n14-fixed", "gemma-4-26B-A4B", 16, 48, 63, 3, True, 4096,
             0.22, 0.18, 0, True, "abc12345", now],
        )
        con.execute(
            "INSERT INTO dim_run VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["run-b", "out/band-n12b", "gemma-4-26B-A4B", 16, 48, 63, 2, False, 4096,
             0.55, 0.60, 1, False, "def67890", now],
        )

        rows = [
            # run-a: chunk 128, three seq lengths spanning a range
            ("run-a", 0, 512, 32, "safety", True, 128, 4, 20.0, 0.04, 5.0, 60.0, now),
            ("run-a", 1, 2048, 128, "benign", True, 128, 16, 90.0, 0.044, 5.6, 120.0, now),
            ("run-a", 2, 4096, 256, "other", False, 128, 32, 200.0, 0.049, 6.25, 190.0, now),
            # run-b: chunk 64, same seq lengths -- the "chunk 64 halves memory" story
            ("run-b", 0, 512, 32, "safety", True, 64, 8, 21.0, 0.041, 2.6, 32.0, now),
            ("run-b", 1, 2048, 128, "benign", True, 64, 32, 92.0, 0.045, 2.9, 68.0, now),
        ]
        for i, r in enumerate(rows):
            con.execute(
                "INSERT INTO fact_item_fit VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [f"key-{i}", r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10],
                 r[11], r[12]],
            )
    finally:
        con.close()


def _start_server(db_path: Path):
    handler_cls = build_handler(db_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _get(port: int, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
        return resp.status, resp.read()


# --- pure functions (no server) ------------------------------------------------------------

def test_open_ro_missing_file_returns_none(tmp_path):
    assert open_ro(tmp_path / "nope.duckdb") is None


def test_fetch_functions_degrade_gracefully_on_none_connection():
    assert fetch_runs(None) == []
    assert fetch_peak_vs_seq(None) == []
    assert fetch_throughput(None) == []


def test_fetch_functions_against_fixture(tmp_path):
    db_path = tmp_path / "fixture.duckdb"
    _build_fixture_db(db_path)
    con = open_ro(db_path)
    try:
        runs = fetch_runs(con)
        assert len(runs) == 2
        assert {r["run_id"] for r in runs} == {"run-a", "run-b"}

        points = fetch_peak_vs_seq(con)
        assert len(points) == 5
        run_a_points = fetch_peak_vs_seq(con, run_id="run-a")
        assert len(run_a_points) == 3
        assert all(p["run_id"] == "run-a" for p in run_a_points)

        throughput = fetch_throughput(con)
        assert {r["chunk_size"] for r in throughput} == {128, 64}
    finally:
        con.close()


# --- HTTP smoke tests ------------------------------------------------------------------------

def test_smoke_all_endpoints(tmp_path):
    db_path = tmp_path / "fixture.duckdb"
    _build_fixture_db(db_path)
    server, thread = _start_server(db_path)
    port = server.server_port
    try:
        status, body = _get(port, "/")
        assert status == 200
        assert b"jlens-mlx fit metrics" in body
        assert b"192 GB" in body  # the ceiling line must be present in the shipped page

        status, body = _get(port, "/api/runs")
        assert status == 200
        runs = json.loads(body)["runs"]
        assert len(runs) == 2
        assert runs[0]["model"] == "gemma-4-26B-A4B"

        status, body = _get(port, "/api/peak_vs_seq")
        assert status == 200
        points = json.loads(body)["points"]
        assert len(points) == 5
        assert {p["chunk_size"] for p in points} == {128, 64}
        first = points[0]
        assert set(first) >= {"run_id", "item_index", "seq_len", "chunk_size", "peak_gb",
                              "on_policy", "stratum"}

        status, body = _get(port, "/api/peak_vs_seq?run_id=run-a")
        assert status == 200
        filtered = json.loads(body)["points"]
        assert len(filtered) == 3
        assert all(p["run_id"] == "run-a" for p in filtered)

        status, body = _get(port, "/api/throughput")
        assert status == 200
        rows = json.loads(body)["rows"]
        assert {r["chunk_size"] for r in rows} == {128, 64}
        by_chunk = {r["chunk_size"]: r for r in rows}
        assert by_chunk[64]["avg_peak_gb"] < by_chunk[128]["avg_peak_gb"]  # the halved-memory story

        # unknown path -> clean 404, never a 500/traceback
        try:
            _get(port, "/nope")
            assert False, "expected HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 404

        # traversal-shaped path -> still a clean 404 (there is no file-serving route to reach)
        try:
            _get(port, "/../../../etc/passwd")
            assert False, "expected HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 404

        try:
            _get(port, "/api/../../etc/passwd")
            assert False, "expected HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_smoke_missing_db_file_does_not_500(tmp_path):
    db_path = tmp_path / "does-not-exist.duckdb"
    server, thread = _start_server(db_path)
    port = server.server_port
    try:
        status, body = _get(port, "/")
        assert status == 200

        status, body = _get(port, "/api/runs")
        assert status == 200 and json.loads(body) == {"runs": []}

        status, body = _get(port, "/api/peak_vs_seq")
        assert status == 200 and json.loads(body) == {"points": []}

        status, body = _get(port, "/api/throughput")
        assert status == 200 and json.loads(body) == {"rows": []}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_smoke_empty_db_no_tables_does_not_500(tmp_path):
    # DB file exists (e.g. touched by an earlier duckdb.connect) but fit_metrics.py hasn't run
    # its schema-creation yet -- no tables/views at all.
    db_path = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(db_path))
    con.close()

    server, thread = _start_server(db_path)
    port = server.server_port
    try:
        status, body = _get(port, "/api/runs")
        assert status == 200 and json.loads(body) == {"runs": []}

        status, body = _get(port, "/api/peak_vs_seq")
        assert status == 200 and json.loads(body) == {"points": []}

        status, body = _get(port, "/api/throughput")
        assert status == 200 and json.loads(body) == {"rows": []}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
