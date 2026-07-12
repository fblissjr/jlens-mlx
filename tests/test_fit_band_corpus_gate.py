"""Unit tests for scripts/fit_band_corpus.py's corpus diversity gate (`_diversity_gate`).

scripts/ isn't a package (no __init__.py, matches the rest of the repo), so the script is loaded
via importlib.util -- its module-level imports (mlx.core, mlx_lm) are lightweight at import time
(no Metal/model touched until main() runs), so this is safe under CPU/sandboxed pytest.

Run:  uv run pytest tests/test_fit_band_corpus_gate.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "fit_band_corpus_under_test", ROOT / "scripts" / "fit_band_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DRIVER = _load_driver()


def test_diversity_gate_passes_when_both_low():
    hard, warn = DRIVER._diversity_gate(
        {"shared_fraction": 0.1, "on_policy": {"n_items": 4, "shared_fraction": 0.2}})
    assert hard == [] and warn == []


def test_diversity_gate_overall_hard_fail():
    hard, warn = DRIVER._diversity_gate(
        {"shared_fraction": 0.6, "on_policy": {"n_items": 0, "shared_fraction": 0.0}})
    assert hard == [("overall", 0.6)]
    # on_policy sub-dict has n_items == 0 -- not even considered.
    assert all(name != "on_policy" for name, _ in hard + warn)


def test_diversity_gate_overall_warn_only():
    hard, warn = DRIVER._diversity_gate(
        {"shared_fraction": 0.4, "on_policy": {"n_items": 4, "shared_fraction": 0.1}})
    assert hard == []
    assert warn == [("overall", 0.4)]


def test_diversity_gate_on_policy_hard_fails_even_when_overall_passes():
    # The exact scenario from the historical band-n12 corpus: overall 0.205 (well under 0.35, no
    # warning even) while the on_policy sub-arm is 0.535 boilerplate -- off-policy items diluted
    # the average and hid on-policy degeneracy. The gate must trip on the on_policy metric alone.
    diversity = {"shared_fraction": 0.205, "on_policy": {"n_items": 6, "shared_fraction": 0.535}}
    hard, warn = DRIVER._diversity_gate(diversity)
    assert hard == [("on_policy", 0.535)]
    assert warn == []


def test_diversity_gate_both_metrics_can_trip_hard_simultaneously():
    diversity = {"shared_fraction": 0.7, "on_policy": {"n_items": 4, "shared_fraction": 0.9}}
    hard, warn = DRIVER._diversity_gate(diversity)
    assert set(hard) == {("overall", 0.7), ("on_policy", 0.9)}


def test_diversity_gate_missing_on_policy_key_treated_as_absent():
    hard, warn = DRIVER._diversity_gate({"shared_fraction": 0.1})
    assert hard == [] and warn == []
