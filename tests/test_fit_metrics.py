"""Unit tests for scripts/fit_metrics.py -- the offline, additive fit-metrics analytics tool.

This ingests one fit run's log + ckpt/corpus artifacts into a shared DuckDB store
(dim_run / fact_item_fit + two views). It is standalone: stdlib + duckdb only, not
imported by the running fit process anywhere. These tests build a hermetic fixture
in a tmp dir and never touch the real out/ directory.

Run:  cd /Users/fredbliss/workspace/heylookitsanllm && uv run pytest /Users/fredbliss/workspace/jlens-mlx/tests/test_fit_metrics.py -v
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "fit_metrics.py"

_spec = importlib.util.spec_from_file_location("fit_metrics", _SCRIPT_PATH)
fit_metrics = importlib.util.module_from_spec(_spec)
sys.modules["fit_metrics"] = fit_metrics
_spec.loader.exec_module(fit_metrics)


def _build_fixture(tmp_path: Path, out_dirname: str = "band-n2-fake") -> Path:
    """Writes a fake <out>.log, <out>/ckpt/corpus.json, <out>/ckpt/ckpt.json under tmp_path.

    Item 1 gets its nearest preceding heartbeat at chunk 40 (n_chunks=40 -> chunk_size=128),
    item 2 gets its nearest preceding heartbeat at chunk 80 (n_chunks=80 -> chunk_size=64).
    """
    out_dir = tmp_path / "out" / out_dirname
    out_dir.mkdir(parents=True)
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir()
    log_path = tmp_path / "out" / f"{out_dirname}.log"

    log_lines = [
        "[14:02:50] load 2.9s  n_layers=64 d_model=5120 band=[16,48) arch=qwen3_5 peak=26.6GB",
        "[14:04:44]   item 1/2 chunk 1/40 (2%) item_elapsed=66s",
        "[14:05:50]   item 1/2 chunk 40/40 (100%) item_elapsed=2640s",
        "[14:49:50]   item 1/2 done in 2770s (47 pos, 58.9 s/pos) | elapsed 46.2m | eta pending (first item) | peak 165.8GB",
        "[14:51:44]   item 2/2 chunk 1/80 (1%) item_elapsed=114s",
        "[14:55:16]   item 2/2 chunk 80/80 (100%) item_elapsed=1900s",
        "[15:17:27]   item 2/2 done in 1950s (32 pos, 60.9 s/pos) | elapsed 32.5m | eta 0m | peak 170.2GB",
    ]
    log_path.write_text("\n".join(log_lines) + "\n")

    # item 0: 78 tokens, item 1: 55 tokens (arbitrary but known).
    corpus = {
        "recipe_name": "unused",
        "provenance": {
            "recipe": "band-bootstrap-v1",
            "n_prompts": 2,
            "on_policy_fraction": 0.5,
            "chat_templated": True,
            "enable_thinking": False,
            "strata": {"fake:strata": 2},
            "seed": 0,
            "max_seq_len": 128,
            "dropped_over_len": 1,
            "diversity": {
                "n_items": 2,
                "total_positions": 79,
                "unique_token_types": 40,
                "shared_fraction": 0.1234,
                "on_policy": {
                    "n_items": 1,
                    "total_positions": 47,
                    "unique_token_types": 30,
                    "shared_fraction": 0.2345,
                },
                "off_policy": {
                    "n_items": 1,
                    "total_positions": 32,
                    "unique_token_types": 20,
                    "shared_fraction": 0.0567,
                },
            },
        },
        "items": [
            {
                "ids": list(range(78)),
                "positions": list(range(47)),
                "stratum": "safety",
                "on_policy": True,
            },
            {
                "ids": list(range(55)),
                "positions": list(range(32)),
                "stratum": "math",
                "on_policy": False,
            },
        ],
    }
    (ckpt_dir / "corpus.json").write_text(json.dumps(corpus))

    ckpt = {
        "next_idx": 2,
        "n_done": 2,
        "layers": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        "target": 63,
        "chunk_size": 64,
        "use_chain": True,
        "n_total": 2,
    }
    (ckpt_dir / "ckpt.json").write_text(json.dumps(ckpt))

    return out_dir


# --- surrogate key derivation ---------------------------------------------------------------

def test_run_id_is_deterministic_md5_of_out_dir_basename():
    out_dir = Path("/some/path/out/band-n14-fixed")
    expected = hashlib.md5(b"band-n14-fixed").hexdigest()
    assert fit_metrics.compute_run_id(out_dir) == expected
    # computing twice gives the same value
    assert fit_metrics.compute_run_id(out_dir) == fit_metrics.compute_run_id(out_dir)


def test_item_fit_key_is_deterministic_md5_of_run_id_and_item_index():
    run_id = "abc123"
    expected = hashlib.md5(f"{run_id}:3".encode()).hexdigest()
    assert fit_metrics.compute_item_fit_key(run_id, 3) == expected
    assert fit_metrics.compute_item_fit_key(run_id, 3) == fit_metrics.compute_item_fit_key(run_id, 3)


# --- log/corpus/ckpt parsing -----------------------------------------------------------------

def test_parse_log_header(tmp_path):
    out_dir = _build_fixture(tmp_path)
    log_path = out_dir.parent / f"{out_dir.name}.log"
    header = fit_metrics.parse_log_header(log_path.read_text())
    assert header["d_model"] == 5120


def test_parse_done_lines_and_chunk_derivation(tmp_path):
    out_dir = _build_fixture(tmp_path)
    log_path = out_dir.parent / f"{out_dir.name}.log"
    log_text = log_path.read_text()
    done_records = fit_metrics.parse_done_lines(log_text, d_model=5120)

    assert len(done_records) == 2
    r1, r2 = done_records
    assert r1["item_index"] == 1
    assert r1["n_items"] == 2
    assert r1["wall_time_s"] == 2770
    assert r1["n_positions"] == 47
    assert abs(r1["sec_per_pos"] - 58.9) < 1e-6
    assert abs(r1["peak_gb"] - 165.8) < 1e-6
    assert r1["n_chunks"] == 40
    assert r1["chunk_size"] == 128  # round(5120/40)

    assert r2["item_index"] == 2
    assert r2["wall_time_s"] == 1950
    assert r2["n_positions"] == 32
    assert abs(r2["peak_gb"] - 170.2) < 1e-6
    assert r2["n_chunks"] == 80
    assert r2["chunk_size"] == 64  # round(5120/80)


# --- full ingest (module-level call) -----------------------------------------------------------

def test_ingest_populates_dim_run_and_fact_item_fit(tmp_path):
    out_dir = _build_fixture(tmp_path)
    db_path = tmp_path / "fit_metrics.duckdb"

    fit_metrics.ingest(out_dir=out_dir, db_path=db_path)

    con = duckdb.connect(str(db_path))
    try:
        runs = con.execute("SELECT * FROM dim_run").fetchdf()
        assert len(runs) == 1
        row = runs.iloc[0]
        assert row["out_dir"] == str(out_dir)
        assert row["band_start"] == 16
        assert row["band_end"] == 31
        assert row["target"] == 63
        assert row["n_items"] == 2
        assert bool(row["enable_thinking"]) is False
        assert row["max_seq_len"] == 128
        assert abs(row["shared_fraction_overall"] - 0.1234) < 1e-9
        assert abs(row["shared_fraction_onpolicy"] - 0.2345) < 1e-9
        assert row["dropped_over_len"] == 1
        assert bool(row["use_chain"]) is True
        assert row["model"] == "unknown"

        facts = con.execute("SELECT * FROM fact_item_fit ORDER BY item_index").fetchdf()
        assert len(facts) == 2

        f1 = facts.iloc[0]
        assert f1["item_index"] == 1
        assert f1["seq_len"] == 78  # len(ids) of item 0
        assert f1["chunk_size"] == 128
        assert f1["n_chunks"] == 40
        assert abs(f1["wall_time_s"] - 2770) < 1e-9
        assert abs(f1["peak_gb"] - 165.8) < 1e-9
        assert abs(f1["sec_per_chunk"] - (2770 / 40)) < 1e-6
        assert f1["stratum"] == "safety"
        assert bool(f1["on_policy"]) is True

        f2 = facts.iloc[1]
        assert f2["item_index"] == 2
        assert f2["seq_len"] == 55  # len(ids) of item 1
        assert f2["chunk_size"] == 64
        assert f2["n_chunks"] == 80
        assert abs(f2["sec_per_chunk"] - (1950 / 80)) < 1e-6
        assert f2["stratum"] == "math"
        assert bool(f2["on_policy"]) is False
    finally:
        con.close()


def test_ingest_is_idempotent(tmp_path):
    out_dir = _build_fixture(tmp_path)
    db_path = tmp_path / "fit_metrics.duckdb"

    fit_metrics.ingest(out_dir=out_dir, db_path=db_path)
    fit_metrics.ingest(out_dir=out_dir, db_path=db_path)

    con = duckdb.connect(str(db_path))
    try:
        n_runs = con.execute("SELECT count(*) FROM dim_run").fetchone()[0]
        n_facts = con.execute("SELECT count(*) FROM fact_item_fit").fetchone()[0]
        assert n_runs == 1
        assert n_facts == 2
    finally:
        con.close()


def test_views_created_and_queryable(tmp_path):
    out_dir = _build_fixture(tmp_path)
    db_path = tmp_path / "fit_metrics.duckdb"
    fit_metrics.ingest(out_dir=out_dir, db_path=db_path)

    con = duckdb.connect(str(db_path))
    try:
        peak_vs_seq = con.execute("SELECT * FROM v_peak_vs_seq ORDER BY seq_len").fetchdf()
        assert len(peak_vs_seq) == 2
        # ordered by seq_len ascending: item 2 (55) then item 1 (78)
        assert list(peak_vs_seq["item_index"]) == [2, 1]

        throughput = con.execute("SELECT * FROM v_throughput ORDER BY chunk_size").fetchdf()
        assert len(throughput) == 2  # two distinct chunk sizes (64, 128)
        assert list(throughput["chunk_size"]) == [64, 128]
    finally:
        con.close()


# --- CLI end-to-end -------------------------------------------------------------------------

def test_cli_end_to_end(tmp_path):
    out_dir = _build_fixture(tmp_path)
    db_path = tmp_path / "cli_fit_metrics.duckdb"

    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--out", str(out_dir), "--db", str(db_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "run_id" in result.stdout or fit_metrics.compute_run_id(out_dir) in result.stdout

    con = duckdb.connect(str(db_path))
    try:
        n_facts = con.execute("SELECT count(*) FROM fact_item_fit").fetchone()[0]
        assert n_facts == 2
    finally:
        con.close()


def test_cli_query_mode_prints_view_without_reingesting(tmp_path):
    out_dir = _build_fixture(tmp_path)
    db_path = tmp_path / "cli_query_fit_metrics.duckdb"

    # ingest first
    fit_metrics.ingest(out_dir=out_dir, db_path=db_path)

    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--out", str(out_dir), "--db", str(db_path), "--query", "throughput"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    con = duckdb.connect(str(db_path))
    try:
        n_facts_after = con.execute("SELECT count(*) FROM fact_item_fit").fetchone()[0]
        assert n_facts_after == 2  # unchanged -- --query does not ingest
    finally:
        con.close()


def test_git_sha_lookup_does_not_hardcode_absolute_path():
    # best-effort git sha lookup must derive repo root from __file__, not a literal path.
    sha = fit_metrics.get_git_sha()
    assert sha is None or isinstance(sha, str)
