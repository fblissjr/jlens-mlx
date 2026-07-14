"""Unit + smoke tests for scripts/fit_monitor.py -- the read-only, stdlib-only local web monitor
for a live band-fit run's `out/<run>/ckpt/*` + sibling `.log` file.

Pure-Python logic (chunk-progress log parsing, corpus summarization, decoded-md parsing, the
path-traversal guard) is unit tested directly, no server involved. The smoke test spins up the
real `http.server` handler on an ephemeral port against a hermetic `tmp_path` fixture it
populates itself -- NEVER the real `out/band-n14-fixed` (a live fit runs against that directory
in the main checkout; this test must not touch it).

Run:  uv run pytest tests/test_fit_monitor.py -q
"""
from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.fit_monitor import (  # noqa: E402
    build_handler, corpus_items_summary, discover_run_log, parse_chunk_progress, parse_decoded_md,
    safe_path, summarize_corpus,
)


# --- parse_chunk_progress ----------------------------------------------------------------------
# Real sample lines straight from a live run's .log (see the task spec / a real out/*/.log file).

CHUNK_LINES = [
    "[14:02:50] load 2.9s  n_layers=64 d_model=5120 band=[16,48) arch=qwen3_5 peak=26.6GB",
    "[14:03:38] fitting band layers [16, 17, ..., 47] (target=63, chunk=128) over the corpus...",
    "[14:04:44]   item 1/12 chunk 1/40 (2%) item_elapsed=66s",
    "[14:05:50]   item 1/12 chunk 2/40 (5%) item_elapsed=132s",
    "[14:06:56]   item 1/12 chunk 3/40 (8%) item_elapsed=198s",
]


def test_parse_chunk_progress_deltas():
    parsed = parse_chunk_progress(CHUNK_LINES)
    assert len(parsed) == 3  # only the 3 chunk-progress lines match; the other 2 are ignored
    assert parsed[0]["item"] == 1 and parsed[0]["chunk"] == 1 and parsed[0]["item_elapsed"] == 66
    assert parsed[0]["delta"] is None  # first chunk seen for item 1 -- no prior value to diff
    assert parsed[1]["item_elapsed"] == 132 and parsed[1]["delta"] == 66
    assert parsed[2]["item_elapsed"] == 198 and parsed[2]["delta"] == 66
    assert all(not p["stall"] for p in parsed)


def test_parse_chunk_progress_item_boundary_not_a_stall():
    # item_elapsed resets (drops) at a new item -- delta is None there, never flagged as a stall.
    lines = [
        "[10:00:00]   item 1/2 chunk 40/40 (100%) item_elapsed=500s",
        "[10:00:05]   item 2/2 chunk 1/40 (2%) item_elapsed=5s",
    ]
    parsed = parse_chunk_progress(lines)
    assert parsed[1]["item"] == 2 and parsed[1]["item_elapsed"] == 5
    assert parsed[1]["delta"] is None
    assert parsed[1]["stall"] is False


def test_parse_chunk_progress_frozen_delta_is_a_stall():
    lines = [
        "[10:00:00]   item 1/2 chunk 1/40 (2%) item_elapsed=10s",
        "[10:05:00]   item 1/2 chunk 1/40 (2%) item_elapsed=10s",  # frozen -- same value again
    ]
    parsed = parse_chunk_progress(lines)
    assert parsed[1]["delta"] == 0
    assert parsed[1]["stall"] is True


def test_parse_chunk_progress_ignores_non_matching_lines():
    assert parse_chunk_progress(["diversity: total_positions=394 unique_types=223", "", "junk"]) == []


# --- summarize_corpus --------------------------------------------------------------------------

def _sample_corpus() -> dict:
    return {
        "recipe_name": "band-bootstrap-v1",
        "provenance": {
            "recipe": "band-bootstrap-v1", "n_prompts": 12, "on_policy_fraction": 0.6,
            "chat_templated": True, "enable_thinking": False,
            "strata": {"JailbreakBench/JBB-Behaviors:behaviors:harmful": 4,
                       "JailbreakBench/JBB-Behaviors:behaviors:benign": 5,
                       "open-r1/OpenR1-Math-220k:train": 5},
            "seed": 0, "max_seq_len": 128, "dropped_over_len": 2,
            "diversity": {
                "n_items": 12, "total_positions": 394, "unique_token_types": 223,
                "shared_fraction": 0.0710659,
                "on_policy": {"n_items": 6, "total_positions": 282, "unique_token_types": 180,
                              "shared_fraction": 0.2730496},
                "off_policy": {"n_items": 6, "total_positions": 112, "unique_token_types": 56,
                               "shared_fraction": 0.0714285},
            },
        },
        "items": [
            {"ids": [101, 102, 103, 9981234], "positions": [1, 2], "stratum": "safety",
             "on_policy": True},
            {"ids": [201, 202], "positions": [0], "stratum": "benign", "on_policy": False},
        ],
    }


def test_summarize_corpus_excludes_items_and_token_ids():
    summary = summarize_corpus(_sample_corpus())
    assert "items" not in summary
    dumped = json.dumps(summary)
    assert "9981234" not in dumped  # a real token id from the fake items list must never leak
    assert "ids" not in summary


def test_summarize_corpus_carries_diversity_fields():
    summary = summarize_corpus(_sample_corpus())
    assert summary["n_items"] == 12
    assert summary["diversity"]["shared_fraction"] == 0.0710659
    assert summary["diversity"]["on_policy"]["shared_fraction"] == 0.2730496
    assert summary["strata"]["JailbreakBench/JBB-Behaviors:behaviors:harmful"] == 4
    assert summary["dropped_over_len"] == 2


def test_summarize_corpus_empty_input():
    assert summarize_corpus({}) == {}


def test_summarize_corpus_missing_keys_degrade_gracefully():
    # No 'provenance' at all -- must not raise, everything degrades to None/empty.
    summary = summarize_corpus({"items": []})
    assert summary["n_items"] is None
    assert summary["diversity"]["shared_fraction"] is None
    assert summary["strata"] == {}


# --- corpus_items_summary ------------------------------------------------------------------------

def test_corpus_items_summary_no_raw_ids_or_positions():
    items = corpus_items_summary(_sample_corpus())
    assert len(items) == 2
    dumped = json.dumps(items)
    assert "9981234" not in dumped
    for it in items:
        assert "ids" not in it and "positions" not in it
    assert items[0] == {"index": 0, "stratum": "safety", "on_policy": True,
                        "n_tokens": 4, "n_fitted_positions": 2}
    assert items[1] == {"index": 1, "stratum": "benign", "on_policy": False,
                        "n_tokens": 2, "n_fitted_positions": 1}


def test_corpus_items_summary_empty_input():
    assert corpus_items_summary({}) == []


# --- parse_decoded_md ----------------------------------------------------------------------------

SAMPLE_DECODED_MD = """# Decoded corpus: test-recipe  (2 items, 1 on-policy)
# strata={'a:b:harmful': 1, 'a:b:benign': 1}  seed=0  max_seq_len=128  dropped_over_len=0
# LOCAL-ONLY inspection artifact: raw dataset prompts + the model's on-policy completions

## item 0  [stratum=safety  on_policy=True  tokens=4  masked_positions=2]
--- prompt (through the generation prompt) ---
<|im_start|>user
hello there<|im_end|>
<|im_start|>assistant
<think>

</think>


--- model's on-policy response (the span J reads) ---
hi, how can I help?

## item 1  [stratum=benign  on_policy=False  tokens=2  masked_positions=1]
<|im_start|>user
plain human text item<|im_end|>
"""


def test_parse_decoded_md_item_count():
    parsed = parse_decoded_md(SAMPLE_DECODED_MD)
    assert len(parsed) == 2
    assert [p["index"] for p in parsed] == [0, 1]


def test_parse_decoded_md_on_policy_prompt_completion_split():
    parsed = parse_decoded_md(SAMPLE_DECODED_MD)
    item0 = parsed[0]
    assert item0["stratum"] == "safety" and item0["on_policy"] is True
    assert item0["tokens"] == 4 and item0["masked_positions"] == 2
    assert "hello there" in item0["prompt"]
    assert "<|im_start|>assistant" in item0["prompt"]
    assert item0["completion"] == "hi, how can I help?"
    # the completion marker text itself must not leak into the split fields
    assert "the span J reads" not in item0["prompt"]
    assert "the span J reads" not in item0["completion"]


def test_parse_decoded_md_off_policy_has_no_completion():
    parsed = parse_decoded_md(SAMPLE_DECODED_MD)
    item1 = parsed[1]
    assert item1["stratum"] == "benign" and item1["on_policy"] is False
    assert item1["completion"] is None
    assert "plain human text item" in item1["prompt"]


def test_parse_decoded_md_empty_input():
    assert parse_decoded_md("") == []


def test_parse_decoded_md_prompt_marker_without_response_marker_is_stripped():
    # Edge case: an item block has the "--- prompt ... ---" marker but no matching
    # "--- model's on-policy response ..." marker (e.g. the last on-policy item in a
    # partially-written file). The marker line itself must still be stripped from the
    # prompt text, not leaked in as literal content.
    md = (
        "## item 0  [stratum=safety  on_policy=True  tokens=3  masked_positions=2]\n"
        "--- prompt (through the generation prompt) ---\n"
        "foo\n"
    )
    parsed = parse_decoded_md(md)
    assert len(parsed) == 1
    assert parsed[0]["prompt"] == "foo"
    assert parsed[0]["completion"] is None
    assert "---" not in parsed[0]["prompt"]


# --- safe_path (path-traversal guard) -----------------------------------------------------------

def test_safe_path_accepts_legit_in_root_filename(tmp_path):
    (tmp_path / "ckpt").mkdir()
    result = safe_path(tmp_path, "ckpt/progress.json")
    assert result == (tmp_path / "ckpt" / "progress.json").resolve()


def test_safe_path_rejects_dotdot_traversal(tmp_path):
    assert safe_path(tmp_path, "../../etc/passwd") is None
    assert safe_path(tmp_path, "ckpt/../../secret.txt") is None


def test_safe_path_rejects_absolute_path_outside_root(tmp_path):
    assert safe_path(tmp_path, "/etc/passwd") is None


def test_safe_path_accepts_absolute_path_that_happens_to_be_inside_root(tmp_path):
    # An "absolute" path string that is actually still under root should still resolve fine --
    # the guard only cares about the RESOLVED location, not the string shape.
    inside = str(tmp_path / "ckpt" / "corpus.json")
    result = safe_path(tmp_path, inside)
    assert result == (tmp_path / "ckpt" / "corpus.json").resolve()


# --- discover_run_log (matched-pair driver log resolution) --------------------------------------

def test_discover_run_log_prefers_per_run_sibling(tmp_path):
    out_root = tmp_path / "band-solo"
    out_root.mkdir()
    per_run = tmp_path / "band-solo.log"
    per_run.write_text("x\n")
    # even with a driver log present, the exact per-run sibling wins when it exists
    (tmp_path / "matched_pair_fits.log").write_text("y\n")
    assert discover_run_log(out_root) == per_run


def test_discover_run_log_falls_back_to_driver_fits_log(tmp_path):
    # matched-pair shape: no per-run <name>.log; ONE combined driver log next to the out dirs.
    out_root = tmp_path / "pair-heretic-deep"
    out_root.mkdir()
    driver = tmp_path / "deep_band_fits.log"
    driver.write_text("fitting...\n")
    assert discover_run_log(out_root) == driver


def test_discover_run_log_picks_newest_fits_log(tmp_path):
    out_root = tmp_path / "pair-base-deep"
    out_root.mkdir()
    older = tmp_path / "matched_pair_fits.log"
    older.write_text("old\n")
    newer = tmp_path / "deep_band_fits.log"
    newer.write_text("new\n")
    import os
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    assert discover_run_log(out_root) == newer


def test_discover_run_log_no_log_returns_per_run_path(tmp_path):
    # nothing on disk -> the per-run path (which simply doesn't exist yet); never raises.
    out_root = tmp_path / "band-not-started"
    out_root.mkdir()
    result = discover_run_log(out_root)
    assert result == tmp_path / "band-not-started.log"
    assert not result.exists()


# --- smoke test: real server, hermetic temp dir, every endpoint ---------------------------------

def _populate_run_dir(run_dir: Path) -> None:
    ckpt = run_dir / "ckpt"
    ckpt.mkdir(parents=True)

    (ckpt / "progress.json").write_text(json.dumps({
        "ts": "2026-07-12T14:13:30", "item": 1, "n_items": 2, "chunk": 3, "n_chunks": 10,
        "positions_done": 5, "positions_total": 50, "sec_per_pos": 1.5, "eta_s": 42.0,
        "peak_gb": 10.0,
    }))

    (ckpt / "ckpt.json").write_text(json.dumps({
        "next_idx": 1, "n_done": 1, "layers": [1, 2, 3], "target": 5, "chunk_size": 10,
        "use_chain": True, "n_total": 2,
    }))

    (ckpt / "corpus.json").write_text(json.dumps(_sample_corpus()))
    (ckpt / "corpus_decoded.md").write_text(SAMPLE_DECODED_MD)

    # sibling .log file, one level up from ckpt/ (i.e. next to run_dir, matching run_dir.name)
    log_path = run_dir.parent / f"{run_dir.name}.log"
    log_path.write_text("\n".join(CHUNK_LINES) + "\n")


def _start_server(run_dir: Path):
    handler_cls = build_handler(run_dir)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _get(port: int, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
        body = resp.read()
        return resp.status, body


def test_smoke_all_endpoints(tmp_path):
    run_dir = tmp_path / "band-test-run"
    _populate_run_dir(run_dir)
    server, thread = _start_server(run_dir)
    port = server.server_port
    try:
        status, body = _get(port, "/")
        assert status == 200
        assert b"jlens-mlx fit monitor" in body

        status, body = _get(port, "/api/progress")
        assert status == 200
        progress = json.loads(body)
        assert progress["item"] == 1 and progress["n_items"] == 2
        assert progress["eta_s"] == 42.0

        status, body = _get(port, "/api/ckpt")
        assert status == 200
        ckpt = json.loads(body)
        assert ckpt["n_done"] == 1 and ckpt["n_total"] == 2

        status, body = _get(port, "/api/corpus")
        assert status == 200
        corpus_summary = json.loads(body)
        assert corpus_summary["n_items"] == 12
        assert "items" not in corpus_summary

        status, body = _get(port, "/api/corpus_items")
        assert status == 200
        corpus_items = json.loads(body)
        assert len(corpus_items["items"]) == 2
        dumped = body.decode()
        assert "9981234" not in dumped  # fake token id must never leak over the wire
        first = corpus_items["items"][0]
        assert set(first) >= {"index", "stratum", "on_policy", "n_tokens", "n_fitted_positions",
                              "prompt", "completion"}
        assert "ids" not in first and "positions" not in first
        assert first["completion"] == "hi, how can I help?"  # joined from corpus_decoded.md
        assert corpus_items["items"][1]["completion"] is None  # off-policy item

        status, body = _get(port, "/api/log?n=5")
        assert status == 200
        log_resp = json.loads(body)
        assert len(log_resp["lines"]) == 5
        assert any("item 1/12 chunk 1/40" in line for line in log_resp["lines"])
        assert len(log_resp["chunk_progress"]) == 3  # 3 real chunk-progress lines in the fixture
        assert log_resp["chunk_progress"][-1]["delta"] == 66

        status, body = _get(port, "/decoded")
        assert status == 200
        assert b"Decoded corpus: test-recipe" in body

        # unknown path -> clean 404, never a 500/traceback
        try:
            _get(port, "/nope")
            assert False, "expected HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 404

        # traversal attempt against a fixed route -> still a clean 404
        try:
            _get(port, "/../../../etc/passwd")
            assert False, "expected HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_smoke_progress_and_ckpt_absent_returns_empty_dict(tmp_path):
    # A fit that hasn't started yet (or between items): no progress.json/ckpt.json at all.
    run_dir = tmp_path / "band-not-started"
    (run_dir / "ckpt").mkdir(parents=True)
    server, thread = _start_server(run_dir)
    port = server.server_port
    try:
        status, body = _get(port, "/api/progress")
        assert status == 200 and json.loads(body) == {}

        status, body = _get(port, "/api/ckpt")
        assert status == 200 and json.loads(body) == {}

        status, body = _get(port, "/api/corpus")
        assert status == 200 and json.loads(body) == {}

        status, body = _get(port, "/api/corpus_items")
        assert status == 200
        assert json.loads(body) == {"items": [], "provenance": {}}

        status, body = _get(port, "/api/log")
        assert status == 200
        log_resp = json.loads(body)
        assert log_resp["lines"] == [] and log_resp["chunk_progress"] == []
        # no log file anywhere -> resolves to the per-run path, flagged not-yet-present
        assert log_resp["log_name"] == "band-not-started.log"
        assert log_resp["log_exists"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_smoke_decoded_404_when_absent(tmp_path):
    run_dir = tmp_path / "band-no-decoded"
    (run_dir / "ckpt").mkdir(parents=True)
    server, thread = _start_server(run_dir)
    port = server.server_port
    try:
        try:
            _get(port, "/decoded")
            assert False, "expected HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_smoke_matched_pair_driver_log_auto_discovered(tmp_path):
    # The matched-pair driver writes ONE combined *_fits.log and no per-run <name>.log. Pointing
    # --out at the active pair dir must still surface that combined log's tail + stall detection.
    run_dir = tmp_path / "pair-heretic-deep"
    (run_dir / "ckpt").mkdir(parents=True)
    (tmp_path / "deep_band_fits.log").write_text("\n".join(CHUNK_LINES) + "\n")
    server, thread = _start_server(run_dir)
    port = server.server_port
    try:
        status, body = _get(port, "/api/log?n=200")
        assert status == 200
        resp = json.loads(body)
        assert resp["log_name"] == "deep_band_fits.log" and resp["log_exists"] is True
        assert any("item 1/12 chunk 1/40" in line for line in resp["lines"])
        assert len(resp["chunk_progress"]) == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_smoke_explicit_log_override(tmp_path):
    # --log wins over both the per-run sibling and any driver-log discovery.
    run_dir = tmp_path / "pair-base-deep"
    (run_dir / "ckpt").mkdir(parents=True)
    (run_dir.parent / f"{run_dir.name}.log").write_text("ignored per-run log\n")
    explicit = tmp_path / "somewhere_else.log"
    explicit.write_text("\n".join(CHUNK_LINES) + "\n")
    handler_cls = build_handler(run_dir, log_override=str(explicit))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    try:
        status, body = _get(port, "/api/log?n=200")
        assert status == 200
        resp = json.loads(body)
        assert resp["log_name"] == "somewhere_else.log"
        assert any("item 1/12 chunk 1/40" in line for line in resp["lines"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_smoke_log_n_param_clamped(tmp_path):
    run_dir = tmp_path / "band-log-clamp"
    (run_dir / "ckpt").mkdir(parents=True)
    log_path = run_dir.parent / f"{run_dir.name}.log"
    log_path.write_text("\n".join(f"line {i}" for i in range(50)) + "\n")
    server, thread = _start_server(run_dir)
    port = server.server_port
    try:
        # invalid n -> default 200 (all 50 lines returned, well under the default)
        status, body = _get(port, "/api/log?n=notanumber")
        assert status == 200
        assert len(json.loads(body)["lines"]) == 50

        # requested cap far above the hard cap -> clamped to 2000, not an error, still just the
        # 50 lines that actually exist
        status, body = _get(port, "/api/log?n=999999")
        assert status == 200
        assert len(json.loads(body)["lines"]) == 50

        # n=3 -> exactly the last 3 lines, in order
        status, body = _get(port, "/api/log?n=3")
        lines = json.loads(body)["lines"]
        assert lines == ["line 47", "line 48", "line 49"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
