"""Unit tests for the fit_corpus visibility/resilience plumbing added on top of the fitter:
positions-weighted ETA (`_positions_eta`), the `progress.json` sidecar (`_write_progress_json`),
and the rate-limited intra-item chunk heartbeat (`_chunk_heartbeat`).

These are pure-Python / CPU-safe -- no model, no real VJP (the chunk-callback WIRING through the
real cotangent loop is Metal-gated and covered separately by scripts/check_progress_callback.py).
`_now`/`_print` injection lets the 30s rate limit be tested without real sleeps.

Run:  uv run pytest tests/test_fit_progress.py -q
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import mlx.core as mx

# Device-agnostic (mx.get_peak_memory() is called by _chunk_heartbeat); CPU keeps this fast and
# sandboxed-safe, matching test_fit.py's convention.
mx.set_default_device(mx.cpu)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jlens_mlx.fit as fit  # noqa: E402


# --- _hms ----------------------------------------------------------------------------------------

def test_hms_format():
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2}", fit._hms())
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2}", fit._hms(0))  # explicit epoch -- same HH:MM:SS shape


# --- _positions_eta ----------------------------------------------------------------------------

def test_positions_eta_first_item_pending():
    # items_timed == 0 -> no rate yet, regardless of the other args.
    assert fit._positions_eta(123.4, 50, 0, 999) == (None, None)


def test_positions_eta_zero_positions_completed_is_pending():
    assert fit._positions_eta(0.0, 0, 1, 100) == (None, None)


def test_positions_eta_computes_rate_and_extrapolates():
    # 100s over 50 positions -> 2 s/pos; 30 remaining positions -> eta 60s.
    sec_per_pos, eta = fit._positions_eta(100.0, 50, 2, 30)
    assert abs(sec_per_pos - 2.0) < 1e-9
    assert abs(eta - 60.0) < 1e-9


def test_positions_eta_zero_remaining_is_zero_not_none():
    sec_per_pos, eta = fit._positions_eta(100.0, 50, 2, 0)
    assert sec_per_pos is not None and eta == 0.0


# --- _write_progress_json -----------------------------------------------------------------------

def test_write_progress_json_atomic_content(tmp_path):
    info = {"ts": "2026-07-12T00:00:00", "item": 3, "n_items": 10, "chunk": 5, "n_chunks": 40,
            "positions_done": 12, "positions_total": 40, "sec_per_pos": 1.5, "eta_s": 42.0,
            "peak_gb": 3.2}
    fit._write_progress_json(str(tmp_path), info)
    p = tmp_path / "progress.json"
    assert p.exists()
    loaded = json.loads(p.read_text())
    assert loaded == info
    # tmp+rename: no leftover .progress.<pid>.tmp files.
    assert not list(tmp_path.glob(".progress.*.tmp"))


def test_write_progress_json_overwrites_in_place(tmp_path):
    fit._write_progress_json(str(tmp_path), {"item": 1})
    fit._write_progress_json(str(tmp_path), {"item": 2})
    p = tmp_path / "progress.json"
    assert json.loads(p.read_text()) == {"item": 2}
    assert len(list(tmp_path.glob("progress.json"))) == 1


# --- _chunk_heartbeat rate limiting --------------------------------------------------------------

class _FakeClock:
    """Injectable `_now` -- a manually-advanced counter, no real sleeps."""
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _make_cb(tmp_path, clock, prints, ticks, **kw):
    return fit._chunk_heartbeat(
        i=0, n_total=3, total_fit_seconds=10.0, positions_completed=5, items_timed=1,
        positions_remaining=20, positions_total=25, checkpoint_dir=str(tmp_path),
        on_tick=ticks.append, min_interval=30.0, _now=clock, _print=lambda *a, **k: prints.append(a[0]),
        **kw)


def test_chunk_heartbeat_first_chunk_always_prints(tmp_path):
    clock = _FakeClock()
    prints, ticks = [], []
    cb = _make_cb(tmp_path, clock, prints, ticks)
    cb(1, 40)
    assert len(prints) == 1
    assert len(ticks) == 1
    assert "item 1/3 chunk 1/40" in prints[0]


def test_chunk_heartbeat_suppresses_within_interval(tmp_path):
    clock = _FakeClock()
    prints, ticks = [], []
    cb = _make_cb(tmp_path, clock, prints, ticks)
    cb(1, 40)                 # prints (first tick)
    clock.advance(10.0)       # < 30s
    cb(2, 40)                 # suppressed
    clock.advance(10.0)       # still < 30s cumulative from last PRINT (20s)
    cb(3, 40)                 # suppressed
    assert len(prints) == 1
    assert len(ticks) == 1


def test_chunk_heartbeat_prints_after_interval_elapses(tmp_path):
    clock = _FakeClock()
    prints, ticks = [], []
    cb = _make_cb(tmp_path, clock, prints, ticks)
    cb(1, 40)
    clock.advance(31.0)       # past the 30s throttle
    cb(2, 40)
    assert len(prints) == 2
    assert len(ticks) == 2


def test_chunk_heartbeat_final_chunk_always_prints(tmp_path):
    clock = _FakeClock()
    prints, ticks = [], []
    cb = _make_cb(tmp_path, clock, prints, ticks)
    cb(1, 40)
    clock.advance(1.0)        # well within the throttle window
    cb(40, 40)                # final chunk -- must print regardless
    assert len(prints) == 2
    assert "chunk 40/40" in prints[-1]


def test_chunk_heartbeat_writes_progress_json_at_same_cadence(tmp_path):
    clock = _FakeClock()
    prints, ticks = [], []
    cb = _make_cb(tmp_path, clock, prints, ticks)
    cb(1, 40)
    p = tmp_path / "progress.json"
    assert p.exists()
    info = json.loads(p.read_text())
    assert info["item"] == 1 and info["n_items"] == 3 and info["chunk"] == 1 and info["n_chunks"] == 40
    assert info["positions_done"] == 5 and info["positions_total"] == 25
    assert "peak_gb" in info and "sec_per_pos" in info and "eta_s" in info


def test_chunk_heartbeat_no_checkpoint_dir_skips_json(tmp_path):
    clock = _FakeClock()
    prints, ticks = [], []
    cb = fit._chunk_heartbeat(i=0, n_total=1, total_fit_seconds=0.0, positions_completed=0,
                              items_timed=0, positions_remaining=10, positions_total=10,
                              checkpoint_dir=None, on_tick=ticks.append, min_interval=30.0,
                              _now=clock, _print=lambda *a, **k: prints.append(a[0]))
    cb(1, 5)
    assert len(prints) == 1 and len(ticks) == 1
    assert not (tmp_path / "progress.json").exists()


# --- fit_corpus integration: heartbeat fires at item completion + writes progress.json ----------

class _Item:
    def __init__(self, val, positions=(0, 1, 2)):
        self.ids = [val, val, val, val]
        self.positions = list(positions)
        self.on_policy = False
        self.stratum = "test"


class _Corpus:
    def __init__(self, items):
        self.items = items


def test_fit_corpus_heartbeat_fires_on_item_completion(monkeypatch, tmp_path):
    def _mock(model, ids, layers, **kw):
        v = float(ids[0])
        return {l: mx.full((2, 2), v, dtype=mx.float32) for l in layers}, len(ids)

    monkeypatch.setattr(fit, "fit_prompt", _mock)
    corpus = _Corpus([_Item(1.0), _Item(2.0)])
    ticks = []
    fit.fit_corpus(object(), corpus, source_layers=[3], adapter=object(), target_layer=7,
                   use_chain=False, checkpoint_dir=str(tmp_path), heartbeat=ticks.append)
    assert len(ticks) >= 2                                    # at least one per completed item
    assert (tmp_path / "progress.json").exists()
    info = json.loads((tmp_path / "progress.json").read_text())
    assert info["item"] == 2 and info["n_items"] == 2
