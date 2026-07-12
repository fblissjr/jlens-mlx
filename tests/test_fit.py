"""Unit tests for fit_corpus orchestration: checkpoint/resume + the progress callback.

These mock `fit_prompt` (no model / no real VJP) so they run fast on CPU alongside a live fit --
they test the ORCHESTRATION (accumulation, atomic checkpointing, resume-skips-done-items,
incompatible-checkpoint refusal, per-item progress), not the numerics (the chain/GDN numerics are
gated by scripts/check_*_synthetic.py). The checkpoint layer is what makes a killed fit recoverable
instead of losing hours -- so it's worth locking down.

Run:  uv run pytest tests/test_fit.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

# Orchestration tests (accumulation / checkpoint / resume / progress) are device-agnostic; run the
# tiny mx ops on the CPU backend so they pass in a sandboxed / headless / no-Metal session.
mx.set_default_device(mx.cpu)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jlens_mlx.fit as fit  # noqa: E402


class _Item:
    def __init__(self, val, positions=(0, 1, 2), on_policy=False):
        # ids[0] carries a distinct per-item value so we can verify the summed/averaged J.
        self.ids = [val, val, val, val]
        self.positions = list(positions)
        self.on_policy = on_policy
        self.stratum = "test"


class _Corpus:
    def __init__(self, items):
        self.items = items


def _mock_fit(returns_value_from_ids=True, raise_on_call=None):
    """A fit_prompt stand-in: item i (ids[0]=v) contributes `full((2,2), v)` per layer.
    Optionally raises RuntimeError on the Nth successful-eligible call (simulates a kill)."""
    calls = {"n": 0}

    def _f(model, ids, layers, **kw):
        if not kw.get("positions"):
            raise ValueError("no usable positions")  # exercised by the skip path
        calls["n"] += 1
        if raise_on_call is not None and calls["n"] == raise_on_call:
            raise RuntimeError("simulated kill mid-fit")
        v = float(ids[0])
        return {l: mx.full((2, 2), v, dtype=mx.float32) for l in layers}, len(ids)

    return _f, calls


def _J(corpus, layers, **kw):
    """Run fit_corpus with a fresh mock (use_chain=False so it routes to the patched fit_prompt)."""
    return fit.fit_corpus(object(), corpus, source_layers=layers, adapter=object(),
                          target_layer=7, use_chain=False, **kw)


def test_fit_corpus_averages_contributions(monkeypatch):
    f, _ = _mock_fit()
    monkeypatch.setattr(fit, "fit_prompt", f)
    corpus = _Corpus([_Item(2.0), _Item(4.0), _Item(6.0)])  # mean 4.0
    J, n = _J(corpus, [3, 5])
    assert n == 3
    for l in (3, 5):
        assert abs(float(J[l][0, 0].item()) - 4.0) < 1e-6


def test_fit_corpus_skips_unusable_items(monkeypatch):
    f, _ = _mock_fit()
    monkeypatch.setattr(fit, "fit_prompt", f)
    corpus = _Corpus([_Item(2.0), _Item(4.0, positions=()), _Item(6.0)])  # middle has no positions
    J, n = _J(corpus, [3])
    assert n == 2                                  # the empty-positions item skipped
    assert abs(float(J[3][0, 0].item()) - 4.0) < 1e-6  # mean of 2 and 6


def test_fit_corpus_checkpoint_resume_matches_clean(monkeypatch, tmp_path):
    layers = [3, 5]
    corpus = _Corpus([_Item(1.0), _Item(2.0), _Item(3.0), _Item(4.0)])  # clean mean 2.5

    # (a) clean run with checkpointing.
    f, _ = _mock_fit()
    monkeypatch.setattr(fit, "fit_prompt", f)
    J_clean, n_clean = _J(corpus, layers, checkpoint_dir=str(tmp_path / "clean"))
    assert n_clean == 4 and abs(float(J_clean[3][0, 0].item()) - 2.5) < 1e-6

    # (b) interrupted run: raise on the 3rd fit -> checkpoint holds items 0,1 (next_idx=2).
    ck = str(tmp_path / "resumed")
    f2, _ = _mock_fit(raise_on_call=3)
    monkeypatch.setattr(fit, "fit_prompt", f2)
    try:
        _J(corpus, layers, checkpoint_dir=ck)
        assert False, "should have raised the simulated kill"
    except RuntimeError:
        pass
    jsum, meta = fit._ckpt_load(ck)
    assert meta["next_idx"] == 2 and meta["n_done"] == 2      # 2 items checkpointed
    assert abs(float(jsum[3][0, 0].item()) - 3.0) < 1e-6      # sum of items 1.0 + 2.0

    # (c) resume: fresh mock, same checkpoint dir -> skips 0,1, fits 2,3, matches the clean mean.
    f3, calls3 = _mock_fit()
    monkeypatch.setattr(fit, "fit_prompt", f3)
    J_res, n_res = _J(corpus, layers, checkpoint_dir=ck, resume=True)
    assert calls3["n"] == 2                                   # ONLY items 2,3 re-fit (0,1 skipped)
    assert n_res == 4 and abs(float(J_res[3][0, 0].item()) - 2.5) < 1e-6
    for l in layers:
        assert abs(float(J_res[l][0, 0].item()) - float(J_clean[l][0, 0].item())) < 1e-6


def test_fit_corpus_checkpoint_incompatible_starts_fresh(monkeypatch, tmp_path):
    ck = str(tmp_path / "ck")
    f, _ = _mock_fit()
    monkeypatch.setattr(fit, "fit_prompt", f)
    _J(_Corpus([_Item(1.0), _Item(2.0)]), [3, 5], checkpoint_dir=ck)  # checkpoint for layers [3,5]

    # Re-run with DIFFERENT layers -> checkpoint must be refused (fresh), not silently wrong.
    events = []
    f2, calls2 = _mock_fit()
    monkeypatch.setattr(fit, "fit_prompt", f2)
    J, n = fit.fit_corpus(object(), _Corpus([_Item(1.0), _Item(2.0)]), source_layers=[3, 4],
                          adapter=object(), target_layer=7, use_chain=False, checkpoint_dir=ck,
                          progress=events.append)
    assert any(e.get("resumed") is False for e in events)     # announced the refusal
    assert calls2["n"] == 2 and n == 2                        # both items actually re-fit
    assert set(J) == {3, 4}


def test_fit_corpus_progress_callback(monkeypatch):
    f, _ = _mock_fit()
    monkeypatch.setattr(fit, "fit_prompt", f)
    corpus = _Corpus([_Item(1.0), _Item(2.0, positions=()), _Item(3.0)])
    events = []
    fit.fit_corpus(object(), corpus, source_layers=[3], adapter=object(), target_layer=7,
                   use_chain=False, progress=events.append)
    item_events = [e for e in events if "skipped" in e]
    assert len(item_events) == 3                              # one per item
    assert item_events[1]["skipped"] is True                 # the empty-positions item
    # Positions-weighted ETA (see fit._positions_eta): the FIRST item actually timed this run has
    # no rate yet ("eta pending"); the SECOND timed item (index 2 here -- item 1 was skipped) gets
    # a real rate-based eta.
    assert item_events[0]["skipped"] is False and item_events[0]["eta_secs"] is None
    assert item_events[2]["skipped"] is False and item_events[2]["eta_secs"] is not None
    assert item_events[0]["done"] == 1 and item_events[2]["done"] == 2
