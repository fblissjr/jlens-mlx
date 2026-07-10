"""Verification: parity vs an oracle, the held-out fidelity gate, and lens diffing.

A lens is not saved unless the fidelity gate passes. See docs/DESIGN.md.
"""
from __future__ import annotations


def parity_vs_oracle(lens, oracle) -> dict:
    """MLX apply vs a genuine jlens.apply() oracle: per-layer cosine + top-k
    overlap. Port migrated_from_scratch/mlx_apply*.py (the V1/V2 gate)."""
    raise NotImplementedError("port migrated_from_scratch/mlx_apply.py")


def fidelity_gate(model, lens, held_out, *, min_topk_agreement: float = 0.5) -> dict:
    """Per-layer KL / top-k agreement between the lens readout and the TRUE logits,
    on HELD-OUT data from the target distribution. Returns per-layer scores; the
    caller refuses to save if below threshold. Never grade a lens on its fit corpus."""
    raise NotImplementedError("new -- docs/DESIGN.md 'Held-out fidelity gate'")


def diff(lens_a, lens_b, *, tokens=None) -> dict:
    """Diff two lenses' readouts (e.g. stock vs abliterated) along given token
    directions. For the abliteration study, the diff IS the finding."""
    raise NotImplementedError("new -- docs/DESIGN.md 'Lens diffing'")
