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
               chunk_size: int = CHUNK_SIZE_DEFAULT, positions=None):
    """Per-prompt `J_l = d(acts[target])/d(acts[l])` via mx.vjp of blocks[l+1..target].
    `chunk_size` is the output-dim batch per VJP (see providers.generic_vjp). `positions`
    overrides the default valid_positions mask (e.g. a corpus role-aware mask); positions
    are clamped to the sequence and the last index is dropped (no next-token target).
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
    if positions is not None:
        vp = sorted({p for p in positions if 0 <= p < len(ids) - 1})
        if not vp:
            raise ValueError("no valid positions after clamping to the sequence")
        valid = mx.array(vp)
    else:
        valid = mx.array(valid_positions(len(ids), skip_first))
    acts = capture_residuals(model, ids, layers, adapter=ad)  # only the source layers
    out = {}
    # NOTE: this fits each layer with its OWN full-tail VJP -- O(n_source * avg_tail) block passes,
    # the cost wall for a dense band fit. `jlens_mlx.chain.fit_prompt_chain` is a drop-in that fits
    # ALL layers in one backward sweep (O(n_blocks)) and is VERIFIED EQUAL to this on qwen3_5 (GDN)
    # + gpt2 (LayerNorm) -- fit_corpus/fit_lens use it by default (`use_chain=True`). This direct
    # path remains the reference the chain is gated against (scripts/check_chain_vs_direct.py) and
    # the fallback for un-gated arches (e.g. gemma array-mask). Keep it correct + simple.
    for l in layers:
        tail = make_tail(ad, l + 1, target + 1)  # blocks l+1..target; l == target -> identity
        out[l] = jacobian_via_vjp(tail, acts[l][None], valid, chunk_size=chunk_size)
    return out, len(ids)


def _fit_one(use_chain: bool):
    """Pick the per-prompt fitter: the O(n_blocks) chain (verified equal on qwen3_5+gpt2) or the
    O(n_source*avg_tail) direct baseline. Lazy import avoids a chain<->fit import cycle."""
    if use_chain:
        from .chain import fit_prompt_chain
        return fit_prompt_chain
    return fit_prompt


def fit_lens(model, prompts, *, source_layers, tokenize, adapter: ModelAdapter | None = None,
             target_layer: int | None = None, skip_first: int = SKIP_FIRST_DEFAULT,
             chunk_size: int = CHUNK_SIZE_DEFAULT, use_chain: bool = True):
    """Average `J_l` over `prompts` (a running sum divided once). `tokenize(prompt) -> list[int]`.
    `chunk_size` is the output-dim batch per VJP (see providers.generic_vjp). `use_chain` picks the
    one-sweep chain fitter (default; verified == direct on qwen3_5/gpt2) vs the per-layer direct VJP.
    Returns ({l: [D,D]}, n_prompts). Save with jlens_mlx.lens.save."""
    ad = adapter or ModelAdapter(model)
    layers = sorted(int(l) for l in source_layers)
    _fit = _fit_one(use_chain)
    acc: dict[int, mx.array] | None = None
    n = 0
    for p in prompts:
        per, _ = _fit(model, tokenize(p), layers, adapter=ad,
                      target_layer=target_layer, skip_first=skip_first,
                      chunk_size=chunk_size)
        acc = dict(per) if acc is None else {l: acc[l] + per[l] for l in layers}
        mx.eval(list(acc.values()))
        n += 1
    if not n:
        raise ValueError("fit_lens: no prompts provided")
    return {l: acc[l] / n for l in layers}, n


def _ckpt_paths(checkpoint_dir):
    from pathlib import Path
    d = Path(checkpoint_dir)
    return d, d / "jsum.safetensors", d / "ckpt.json"


def _ckpt_save(checkpoint_dir, jsum: dict, meta: dict) -> None:
    """Atomically persist the running J_sum (safetensors) + meta (json). Written after each item
    so a killed fit resumes instead of losing everything (the session/harness reaps background
    children on teardown -- see the ops gotcha in the memory)."""
    import json
    import os
    d, sf, js = _ckpt_paths(checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    # mx.save_safetensors APPENDS ".safetensors" if absent -> the tmp name MUST already end in it,
    # else the written file name won't match what we os.replace (a real bug the tests caught).
    tmp_sf = str(d / f".jsum.{os.getpid()}.tmp.safetensors")
    mx.save_safetensors(tmp_sf, {str(l): v.astype(mx.float32) for l, v in jsum.items()})
    os.replace(tmp_sf, sf)                     # atomic swap into place
    tmp_js = str(js) + f".tmp.{os.getpid()}"
    with open(tmp_js, "w") as f:
        json.dump(meta, f)
    os.replace(tmp_js, js)


def _ckpt_load(checkpoint_dir):
    """Return (jsum {l: array}, meta) or (None, None) if absent."""
    import json
    _, sf, js = _ckpt_paths(checkpoint_dir)
    if not (sf.exists() and js.exists()):
        return None, None
    meta = json.loads(js.read_text())
    arrays = mx.load(str(sf))
    jsum = {int(k): v for k, v in arrays.items()}
    return jsum, meta


def fit_corpus(model, corpus, *, source_layers, adapter: ModelAdapter | None = None,
               target_layer: int | None = None, chunk_size: int = CHUNK_SIZE_DEFAULT,
               progress=None, use_chain: bool = True, checkpoint_dir=None, resume: bool = True):
    """Average `J_l` over a materialized `Corpus` (jlens_mlx.corpus), using each item's own
    role-aware position mask (assistant/think span for on-policy prompts, content span for
    human-text). Returns ({l: [D,D]}, n_items). The corpus provenance should be stamped onto
    the lens sidecar. Items whose positions all fall outside their sequence are skipped.

    `use_chain` (default) fits all layers per item in one backward sweep (jlens_mlx.chain, verified
    == direct on qwen3_5/gpt2) -- the big win for a dense band fit; set False for the direct baseline
    (e.g. an un-gated arch like gemma array-mask).

    `checkpoint_dir` (recommended for long fits): after EACH item, atomically save the running J_sum
    + progress. A killed fit resumes from there (skips completed items) when re-run with the SAME
    corpus + `source_layers` + `target_layer` (the caller must pass the same materialized corpus --
    serialize it alongside; see corpus.Corpus.to_json). Resume is refused (fresh start) if those
    don't match. This is the robust alternative to detaching the process.

    `progress(info)` is called after each item (long deep-band fits are otherwise silent) with
    a dict: {i, n_total, done, skipped, seq_len, n_pos, on_policy, secs, elapsed, eta_secs}.
    `secs` is that item's fit time; `eta_secs` extrapolates from the mean so far."""
    import time
    ad = adapter or ModelAdapter(model)
    layers = sorted(int(l) for l in source_layers)
    target = (ad.n_layers - 1) if target_layer is None else int(target_layer)
    _fit = _fit_one(use_chain)
    n_total = len(corpus.items)

    acc: dict[int, mx.array] | None = None
    n = 0            # items that contributed to J_sum (the divisor)
    start_idx = 0    # next item index to process
    if checkpoint_dir and resume:
        jsum, meta = _ckpt_load(checkpoint_dir)
        if jsum is not None and meta is not None and meta.get("layers") == layers \
                and meta.get("target") == target and meta.get("n_total") == n_total \
                and meta.get("next_idx", 0) < n_total:
            acc, n, start_idx = jsum, int(meta["n_done"]), int(meta["next_idx"])
            if progress:
                progress({"resumed": True, "next_idx": start_idx, "done": n, "n_total": n_total})
        elif jsum is not None:
            if progress:
                progress({"resumed": False, "reason": "checkpoint incompatible -- starting fresh",
                          "n_total": n_total})

    t_start = time.perf_counter()
    for i, item in enumerate(corpus.items):
        if i < start_idx:
            continue  # already done in a prior (killed) run
        t0 = time.perf_counter()
        try:
            per, _ = _fit(model, item.ids, layers, adapter=ad, target_layer=target_layer,
                          chunk_size=chunk_size, positions=item.positions)
        except ValueError:
            if checkpoint_dir and acc is not None:
                _ckpt_save(checkpoint_dir, acc, {"next_idx": i + 1, "n_done": n, "layers": layers,
                                                 "target": target, "chunk_size": chunk_size,
                                                 "use_chain": use_chain, "n_total": n_total})
            if progress:
                progress({"i": i, "n_total": n_total, "done": n, "skipped": True,
                          "seq_len": len(item.ids), "n_pos": len(item.positions),
                          "on_policy": item.on_policy, "secs": 0.0,
                          "elapsed": time.perf_counter() - t_start, "eta_secs": None})
            continue  # no usable positions for this item
        acc = dict(per) if acc is None else {l: acc[l] + per[l] for l in layers}
        mx.eval(list(acc.values()))
        n += 1
        if checkpoint_dir:
            _ckpt_save(checkpoint_dir, acc, {"next_idx": i + 1, "n_done": n, "layers": layers,
                                             "target": target, "chunk_size": chunk_size,
                                             "use_chain": use_chain, "n_total": n_total})
        if progress:
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / (i - start_idx + 1)) * (n_total - i - 1)
            progress({"i": i, "n_total": n_total, "done": n, "skipped": False,
                      "seq_len": len(item.ids), "n_pos": len(item.positions),
                      "on_policy": item.on_policy, "secs": time.perf_counter() - t0,
                      "elapsed": elapsed, "eta_secs": eta})
    if not n:
        raise ValueError("fit_corpus: no items produced a usable fit")
    return {l: acc[l] / n for l in layers}, n
