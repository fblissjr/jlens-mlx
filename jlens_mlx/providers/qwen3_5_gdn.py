"""qwen3_5 (Gated-DeltaNet hybrid) GDN speed accelerator — DEFERRED.

The direct-VJP baseline (jlens_mlx.fit) works on any arch but is slow through qwen3_5's 48
GDN linear-attention layers: MLX's fused GDN kernel has no VJP, so a direct end-to-end VJP
falls back to a ~22x-slower ops path. This module will hold the ~30-60x speedup PORTED (not
vendored) from WeZZard/jlens-qwen36 (Apache-2.0): a custom Metal GDN backward kernel + a
qwen3_5-aware tail runner (fa/ssm masks + is_linear dispatch). Each piece is re-verified vs
`mx.vjp` before use — jlens-qwen36's own closed-form RMSNorm seed had an rms**2/rms**3 bug.

Only needed when the baseline is too slow on the real 27B; small-model baselines need none.
Reach the text stack of Qwen3_5ForConditionalGeneration via .language_model.model.
"""
from __future__ import annotations


def make_qwen3_5_tail(adapter, start, end):
    """The GDN-aware analogue of fit.make_tail (blocks [start, end) with fa/ssm masks +
    is_linear dispatch). Deferred; see the plan's 'The modular fitter'."""
    raise NotImplementedError(
        "deferred GDN accelerator: port the qwen3_5 tail + Metal GDN backward from "
        "WeZZard/jlens-qwen36, verified vs mx.vjp"
    )
