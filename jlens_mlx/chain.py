"""Exact reverse-mode CHAIN fitter -- an unverified speedup for multi-layer fits.

┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ ⚠  UNVERIFIED ON GPU (drafted 2026-07-10). DO NOT trust these J's or wire this into the      │
│    default fit path until `verify_chain_matches_direct()` passes on a real model. This is a  │
│    deliberate "get it close, leave notes" draft, per owner request -- written to be correct  │
│    by construction, but NOT yet numerically confirmed against the direct-VJP baseline. The   │
│    whole project ethos is port/derive → VERIFY → trust (we caught an rms² bug in a reference  │
│    by exactly this discipline). Treat every J from here as suspect until the gate is green.   │
│    A later Claude: run the verification (below), then flip DEFAULT + delete this banner.      │
└───────────────────────────────────────────────────────────────────────────────────────────┘

WHY THIS EXISTS (the perf problem it solves)
--------------------------------------------
`fit.fit_prompt` (the trusted baseline) fits each source layer l with its OWN end-to-end VJP
through the tail blocks[l+1..target]. Cost = sum over source layers of (tail_length · D) block
backward-passes = O(n_source · avg_tail · D). For a DENSE band fit (layers 16..47 of a 64-layer
model) the deep layers have ~30-47-block tails, so this is hours→days even dim-batched.

THE INSIGHT (why the chain is exact here, not the decorrelated approximation)
----------------------------------------------------------------------------
Reverse-mode autodiff computes the gradient w.r.t. EVERY intermediate activation in one backward
pass. So a SINGLE backward sweep from acts[target] -- seeded with the same cotangent the direct
path uses -- yields d(acts[target])/d(acts[l]) for ALL l as it flows past each layer. Reading the
cotangent at acts[l] gives exactly the same J_l the direct VJP computes (same seed, same blocks,
same position averaging). Cost = O(n_blocks · D) block backward-passes -- a ~n_source× win for a
band fit, EXACT, no approximation.

This is NOT the naive "chain of per-layer averaged M_l" (`J_{l-1}=J_l·M_l` with position-averaged
factors). That approach loses per-position norm/gradient correlations and diverges 33-49% from the
truth (the reference had to do complex per-position analytic folding to avoid it). We avoid it
entirely: we never average before multiplying -- we carry the full [C,S,D] cotangent through every
block and only average at the readout, identically to the direct VJP.

WHAT MIGHT BE WRONG (verify these before trusting)
--------------------------------------------------
1. ⚠ GDN custom_function under a PER-BLOCK vjp. The direct path VJPs a whole tail; here we VJP one
   block at a time. The GDN custom VJP is registered on the op, so per-block vjp SHOULD fire it --
   but this is the single most likely thing to be subtly wrong (batching, state handling). The
   verification compares against the direct path which we DID verify, so a mismatch localizes here.
2. ⚠ Per-block mask dispatch must match the model's forward EXACTLY (fa vs ssm for qwen3_5; array
   vs "causal" for gemma vs gpt2). We reuse the same construction as fit.make_tail /
   qwen3_5_gdn.make_qwen3_5_tail; if those are right, this is too -- but confirm.
3. ⚠ The gdn_fit_patch context must wrap the WHOLE sweep (every block vjp), like the tail runner.
4. ⚠ Indexing: J_l is read from the cotangent at acts[l]; block l maps acts[l-1]→acts[l], so we
   vjp block l with primal acts[l-1]. Off-by-one here is the classic chain bug (the reference hit
   it). The identity check (J_target == I) + the direct-path parity are the guards.
5. ⚠ Precision/eval cadence: we eval the cotangent each block to bound the graph; confirm this
   doesn't change results vs the direct path (it shouldn't -- eval is a materialization, not a
   numerical change).

VERIFICATION PLAN (a later Claude: do this first)
-------------------------------------------------
`verify_chain_matches_direct(model, ids, sources, target)` fits the same layers both ways and
returns per-layer cosine + max-abs-err. Gate: cos > 0.99999 and rel err < 1e-4 on (a) the tiny
synthetic qwen3_5 (scripts/check_qwen3_5_synthetic.py's tiny_qwen3_5 -- exercises the GDN path)
AND (b) gpt2 (LayerNorm, the trusted baseline arch). Only then wire `fit_lens`/`fit_corpus` to
use `fit_prompt_chain` for multi-layer fits and drop this banner.
"""
from __future__ import annotations

from contextlib import nullcontext

import mlx.core as mx

from .capture import ModelAdapter, capture_residuals
from .fit import (
    _ARRAY_MASK_ARCHS, _GDN_TAIL_ARCHS, SKIP_FIRST_DEFAULT, _model_type, valid_positions,
)
from .providers.generic_vjp import CHUNK_SIZE_DEFAULT


def _resolve_target(ad: ModelAdapter, target_layer: int | None) -> int:
    n = ad.n_layers
    if target_layer is None:
        return n - 1
    t = int(target_layer)
    if t < 0:
        t += n
    if not (0 <= t < n):
        raise ValueError(f"target_layer {target_layer} out of range for {n} layers")
    return t


def _mask_fn(ad: ModelAdapter, h: mx.array):
    """A fn(block) -> mask, mirroring fit.make_tail / qwen3_5_gdn.make_qwen3_5_tail EXACTLY (see
    uncertainty #2). Masks depend only on the sequence length (h.shape[1]), so they are built once
    for the prompt and reused for every block and every chunk."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    mt = _model_type(ad)
    if mt in _GDN_TAIL_ARCHS:
        fa = create_attention_mask(h, cache=None)
        ssm = create_ssm_mask(h, cache=None)
        return lambda block: (ssm if getattr(block, "is_linear", False) else fa)
    array_mask = mt in _ARRAY_MASK_ARCHS
    m = create_attention_mask(h, cache=None, return_array=array_mask)
    return lambda block: m


def fit_prompt_chain(model, input_ids, source_layers, *, adapter: ModelAdapter | None = None,
                     target_layer: int | None = None, skip_first: int = SKIP_FIRST_DEFAULT,
                     positions=None, chunk_size: int = CHUNK_SIZE_DEFAULT):
    """UNVERIFIED. Fit `J_l` for ALL `source_layers` in one backward sweep (see module header).
    Same signature/semantics as `fit.fit_prompt` -- MUST return the same J's (that is the gate).
    Returns ({l: [D,D] float32}, seq_len)."""
    ad = adapter or ModelAdapter(model)
    target = _resolve_target(ad, target_layer)
    layers = sorted({int(l) for l in source_layers})
    for l in layers:
        if not (0 <= l <= target):
            raise ValueError(f"source layer {l} must be in [0, target={target}]")

    ids = list(input_ids)
    S = len(ids)
    if positions is not None:
        vp = sorted({p for p in positions if 0 <= p < S - 1})
        if not vp:
            raise ValueError("no valid positions after clamping to the sequence")
    else:
        vp = valid_positions(S, skip_first)
    valid = mx.array(vp)

    blocks = ad.layers
    min_src = min(layers)
    # Block l (for l in [min_src+1 .. target]) is VJP'd with primal acts[l-1]; so capture the block
    # OUTPUTS at indices [min_src .. target-1] (= those block inputs). J_target reads the seed only.
    cap = list(range(min_src, target))
    acts = capture_residuals(model, ids, cap, adapter=ad) if cap else {}
    # d_model from a captured act, else from the first block's input norm (single-layer identity case).
    D = (acts[min_src].shape[-1] if cap
         else blocks[min_src].input_layernorm.weight.shape[0])

    is_gdn = _model_type(ad) in _GDN_TAIL_ARCHS
    patch_cm = nullcontext()
    if is_gdn:                                               # ⚠ #3: wrap the whole sweep
        from .providers.qwen3_5_gdn import gdn_fit_patch
        patch_cm = gdn_fit_patch()

    eye = mx.eye(D, dtype=mx.float32)
    pos_mask = mx.zeros((S,), dtype=mx.float32).at[valid].add(1.0)
    # A representative h to build masks (batch dim irrelevant to mask construction).
    mask_for = _mask_fn(ad, mx.zeros((1, S, D), dtype=mx.float32))

    # Accumulate J rows per layer as we sweep, chunk by chunk over output dims.
    rows: dict[int, list[mx.array]] = {l: [] for l in layers}
    C = max(1, int(chunk_size))
    with patch_cm:
        for lo in range(0, D, C):
            dims = list(range(lo, min(lo + C, D)))
            c = len(dims)
            onehots = eye[mx.array(dims)]                       # [c, D]
            cot = pos_mask[None, :, None] * onehots[:, None, :]  # [c, S, D] at acts[target]
            for l in range(target, min_src - 1, -1):
                if l in rows:
                    # J_l rows for these output dims = avg over valid SOURCE positions of the
                    # cotangent now sitting at acts[l] (identical readout to jacobian_via_vjp).
                    rows[l].append(cot[:, valid, :].astype(mx.float32).mean(axis=1))  # [c, D]
                if l > min_src:
                    h_in = mx.repeat(acts[l - 1][None], c, axis=0)  # [c, S, D] independent copies
                    mask = mask_for(blocks[l])
                    block = blocks[l]
                    _, g = mx.vjp(lambda h: block(h, mask, cache=None), [h_in], [cot])
                    cot = g[0]                                      # [c, S, D] at acts[l-1]
                    mx.eval(cot)                                    # ⚠ #5: bound the graph

    out = {l: mx.concatenate(rows[l], axis=0) for l in layers}     # each [D, D], row = output dim
    return out, S


def verify_chain_matches_direct(model, input_ids, source_layers, *, adapter: ModelAdapter | None = None,
                                target_layer: int | None = None, skip_first: int = SKIP_FIRST_DEFAULT,
                                chunk_size: int = CHUNK_SIZE_DEFAULT) -> dict:
    """THE GATE. Fit the same layers via the trusted direct VJP and via this chain, and compare.
    Returns {l: {"cos": float, "max_abs_err": float, "rel_err": float}} + {"pass": bool}. A later
    Claude runs this on tiny_qwen3_5 (GDN path) AND gpt2 (LayerNorm) before trusting/wiring chain.

    Gate: cos > 0.99999 and rel_err < 1e-4 for every layer (they are the SAME estimator, so any
    real gap is a bug -- most likely uncertainty #1 or #4 in the module header)."""
    import numpy as np
    from .fit import fit_prompt
    ad = adapter or ModelAdapter(model)
    direct, _ = fit_prompt(model, input_ids, source_layers, adapter=ad, target_layer=target_layer,
                           skip_first=skip_first, chunk_size=chunk_size)
    chained, _ = fit_prompt_chain(model, input_ids, source_layers, adapter=ad,
                                  target_layer=target_layer, skip_first=skip_first,
                                  chunk_size=chunk_size)
    report: dict = {}
    ok = True
    for l in sorted(direct):
        a = np.asarray(direct[l], dtype=np.float64).ravel()
        b = np.asarray(chained[l], dtype=np.float64).ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        mae = float(np.abs(a - b).max())
        rel = float(mae / (np.abs(a).max() + 1e-12))
        report[l] = {"cos": cos, "max_abs_err": mae, "rel_err": rel}
        ok = ok and cos > 0.99999 and rel < 1e-4
    report["pass"] = ok
    return report
