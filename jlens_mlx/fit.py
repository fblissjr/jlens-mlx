"""Direct end-to-end Jacobian-lens fitting — Anthropic jacobian-lens design, ported to MLX.

For each source layer l, `J_l = d(acts[target])/d(acts[l])`: the exact end-to-end Jacobian
from layer l's residual output to the target block's residual output, via `mx.vjp` of the
"tail" (blocks l+1..target). Averaged over corpus prompts. There is NO chain of per-layer
factors and NO closed-form norm seed — the final norm stays OUTSIDE J and is applied as the
real module at decode (jlens_mlx.lens.JSpaceLens.apply, mirroring the heylook server). This
mirrors anthropics/jacobian-lens fitting.py and is correct-by-construction.

The tail runner (make_tail) is the only arch-specific piece; the default handles
gpt2/llama-style blocks `block(h, mask, cache)`. qwen3_5 (GDN hybrid) needs its own runner
(fa/ssm masks + is_linear dispatch) — the deferred accelerator (providers/qwen3_5_gdn).
"""
from __future__ import annotations

import mlx.core as mx

from .capture import ModelAdapter, capture_residuals
from .providers.generic_vjp import jacobian_via_vjp

#: Leading positions excluded from the average (attention sinks). Anthropic uses 16 for
#: 128-token prompts; scale down for short prompts.
SKIP_FIRST_DEFAULT = 16


def valid_positions(seq_len: int, skip_first: int = SKIP_FIRST_DEFAULT) -> list[int]:
    """Positions [skip_first, seq_len-1): skip attention sinks + the last position (no
    next-token target). Mirrors Anthropic's valid_position_mask; clamps on short prompts
    so at least one position remains."""
    lo = skip_first if seq_len - 1 > skip_first else max(0, seq_len - 2)
    return list(range(lo, seq_len - 1)) or [max(0, seq_len - 2)]


def make_tail(adapter: ModelAdapter, start: int, end: int):
    """fn(h[1,S,D]) -> [1,S,D] running decoder blocks [start, end).

    Default runner for gpt2/llama-style blocks `block(h, mask, cache)`. For start >= end
    (l == target) it is the identity, so J = I. qwen3_5/GDN needs its own runner."""
    from mlx_lm.models.base import create_attention_mask

    blocks = adapter.layers

    def tail(h: mx.array) -> mx.array:
        mask = create_attention_mask(h, cache=None)
        for i in range(start, end):
            h = blocks[i](h, mask, cache=None)
        return h

    return tail


def fit_prompt(model, input_ids, source_layers, *, adapter: ModelAdapter | None = None,
               target_layer: int | None = None, skip_first: int = SKIP_FIRST_DEFAULT):
    """Per-prompt `J_l = d(acts[target])/d(acts[l])` via mx.vjp of blocks[l+1..target].
    Returns ({l: [D,D]}, seq_len)."""
    ad = adapter or ModelAdapter(model)
    n = ad.n_layers
    target = (n - 1) if target_layer is None else (int(target_layer) % n)
    ids = list(input_ids)
    S = len(ids)
    acts = capture_residuals(model, ids, list(range(n)), adapter=ad)  # {l: [S, D]}
    valid = mx.array(valid_positions(S, skip_first))
    out = {}
    for l in source_layers:
        tail = make_tail(ad, int(l) + 1, target + 1)       # blocks l+1..target; l==target -> identity
        out[int(l)] = jacobian_via_vjp(tail, acts[int(l)][None], valid)
    return out, S


def fit_lens(model, prompts, *, source_layers, tokenize, adapter: ModelAdapter | None = None,
             target_layer: int | None = None, skip_first: int = SKIP_FIRST_DEFAULT):
    """Average `J_l` over `prompts` (a running mean). `tokenize(prompt) -> list[int]`.
    Returns ({l: [D,D]}, n_prompts). Save with jlens_mlx.lens.save."""
    ad = adapter or ModelAdapter(model)
    layers = sorted(int(l) for l in source_layers)
    acc: dict[int, mx.array] | None = None
    n = 0
    for p in prompts:
        per, _ = fit_prompt(model, tokenize(p), layers, adapter=ad,
                            target_layer=target_layer, skip_first=skip_first)
        acc = per if acc is None else {l: acc[l] + per[l] for l in layers}
        mx.eval(list(acc.values()))
        n += 1
    return {l: acc[l] / n for l in layers}, n
