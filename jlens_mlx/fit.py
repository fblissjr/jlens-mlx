"""Direct end-to-end Jacobian-lens fitting — Anthropic jacobian-lens design, ported to MLX.

For each source layer l, `J_l = d(acts[target])/d(acts[l])`: the exact end-to-end Jacobian
from layer l's residual output to the target block's residual output, via `mx.vjp` of the
"tail" (blocks l+1..target). Averaged over corpus prompts. There is NO chain of per-layer
factors and NO closed-form norm seed — the final norm stays OUTSIDE J and is applied as the
real module at decode (jlens_mlx.lens.JSpaceLens.apply, mirroring the heylook server). This
mirrors anthropics/jacobian-lens fitting.py and is correct-by-construction.

The tail runner (make_tail) is the only arch-specific piece. It builds the causal mask the
model's own forward uses: the "causal" STRING for SDPA-path archs (gpt2/llama), or an ARRAY
mask for archs whose attention reads `mask.dtype` (gemma*, which use a manual softcapped score
path). qwen3_5 (GDN hybrid) dispatches to providers/qwen3_5_gdn.make_qwen3_5_tail (per-layer
fa/ssm masks + the differentiable GDN recurrence with the ported Metal backward).
"""
from __future__ import annotations

import mlx.core as mx

from .capture import ModelAdapter, capture_residuals
from .providers.generic_vjp import CHUNK_SIZE_DEFAULT, jacobian_via_vjp

#: Leading positions excluded from the average (attention sinks). Anthropic uses 16 for
#: 128-token prompts; scale down for short prompts.
SKIP_FIRST_DEFAULT = 16

#: Architectures whose attention reads `mask.dtype` (a manual score path, e.g. softcapped
#: attention) and so require an ARRAY causal mask, not the "causal" string that SDPA-path
#: models accept. gpt2/llama stay on the string mask, so their verified parity is untouched.
_ARRAY_MASK_ARCHS = {"gemma2", "gemma3", "gemma3_text", "gemma"}

#: Gated-DeltaNet hybrid architectures (fa/ssm mask dispatch + a GDN recurrence with
#: no fused-kernel VJP) -- routed to providers/qwen3_5_gdn.make_qwen3_5_tail.
_GDN_TAIL_ARCHS = {"qwen3_5", "qwen3_5_text", "qwen3_5_moe"}


def valid_positions(seq_len: int, skip_first: int = SKIP_FIRST_DEFAULT) -> list[int]:
    """Positions [skip_first, seq_len-1): skip attention sinks + the last position (no
    next-token target). Clamps to a single position on short prompts. Requires seq_len >= 2."""
    if seq_len < 2:
        raise ValueError(f"prompt too short (seq_len={seq_len}); need >= 2 tokens to fit")
    lo = skip_first if seq_len - 1 > skip_first else seq_len - 2
    return list(range(lo, seq_len - 1)) or [seq_len - 2]


def _model_type(adapter: ModelAdapter) -> str:
    for obj in (adapter.model, getattr(adapter.model, "args", None),
                getattr(adapter.model, "config", None), adapter.inner):
        mt = getattr(obj, "model_type", None)
        if isinstance(mt, str):
            return mt
    return ""


def make_tail(adapter: ModelAdapter, start: int, end: int):
    """fn(h[1,S,D]) -> [1,S,D] running decoder blocks [start, end) with the model's own
    causal mask type (array for gemma*, "causal" string for gpt2/llama). For start >= end
    (l == target) it is the identity, so J = I. qwen3_5/GDN gets its own runner."""
    from mlx_lm.models.base import create_attention_mask

    if _model_type(adapter) in _GDN_TAIL_ARCHS:
        from .providers.qwen3_5_gdn import make_qwen3_5_tail
        return make_qwen3_5_tail(adapter, start, end)

    blocks = adapter.layers
    array_mask = _model_type(adapter) in _ARRAY_MASK_ARCHS

    def tail(h: mx.array) -> mx.array:
        mask = create_attention_mask(h, cache=None, return_array=array_mask)
        for i in range(start, end):
            h = blocks[i](h, mask, cache=None)
        return h

    return tail


def fit_prompt(model, input_ids, source_layers, *, adapter: ModelAdapter | None = None,
               target_layer: int | None = None, skip_first: int = SKIP_FIRST_DEFAULT,
               chunk_size: int = CHUNK_SIZE_DEFAULT):
    """Per-prompt `J_l = d(acts[target])/d(acts[l])` via mx.vjp of blocks[l+1..target].
    `chunk_size` is the output-dim batch per VJP (see providers.generic_vjp).
    Returns ({l: [D,D]}, seq_len)."""
    ad = adapter or ModelAdapter(model)
    n = ad.n_layers
    if target_layer is None:
        target = n - 1
    else:
        target = int(target_layer)
        if target < 0:
            target += n
        if not (0 <= target < n):
            raise ValueError(f"target_layer {target_layer} out of range for {n} layers")
    layers = [int(l) for l in source_layers]
    for l in layers:
        if not (0 <= l <= target):
            raise ValueError(f"source layer {l} must be in [0, target={target}]")

    ids = list(input_ids)
    valid = mx.array(valid_positions(len(ids), skip_first))
    acts = capture_residuals(model, ids, layers, adapter=ad)  # only the source layers
    out = {}
    for l in layers:
        tail = make_tail(ad, l + 1, target + 1)  # blocks l+1..target; l == target -> identity
        out[l] = jacobian_via_vjp(tail, acts[l][None], valid, chunk_size=chunk_size)
    return out, len(ids)


def fit_lens(model, prompts, *, source_layers, tokenize, adapter: ModelAdapter | None = None,
             target_layer: int | None = None, skip_first: int = SKIP_FIRST_DEFAULT,
             chunk_size: int = CHUNK_SIZE_DEFAULT):
    """Average `J_l` over `prompts` (a running sum divided once). `tokenize(prompt) -> list[int]`.
    `chunk_size` is the output-dim batch per VJP (see providers.generic_vjp).
    Returns ({l: [D,D]}, n_prompts). Save with jlens_mlx.lens.save."""
    ad = adapter or ModelAdapter(model)
    layers = sorted(int(l) for l in source_layers)
    acc: dict[int, mx.array] | None = None
    n = 0
    for p in prompts:
        per, _ = fit_prompt(model, tokenize(p), layers, adapter=ad,
                            target_layer=target_layer, skip_first=skip_first,
                            chunk_size=chunk_size)
        acc = dict(per) if acc is None else {l: acc[l] + per[l] for l in layers}
        mx.eval(list(acc.values()))
        n += 1
    if not n:
        raise ValueError("fit_lens: no prompts provided")
    return {l: acc[l] / n for l in layers}, n
