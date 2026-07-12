"""VJP primitive for direct end-to-end Jacobian fitting.

Ports the estimator from anthropics/jacobian-lens (fitting.py::jacobian_for_prompt) to
MLX. For a function `fn` (the "tail" from a source layer's residual to the target block's
residual), compute the position-averaged, future-summed Jacobian. Correct by construction
(autograd); no closed form, so no rms**2/rms**3 class of bug.

The Jacobian has D output rows, each the gradient of one output dim. Computing them
one-at-a-time reruns `fn`'s forward D times (D=5120 on the 27B) -- the fit bottleneck.
`jacobian_via_vjp` instead **batches C rows through `fn`'s native batch axis**: C independent
copies of the primal `h` (`[C, S, D]`), each with a cotangent hot at a different output dim.
One `mx.vjp` returns all C gradients at once (`grad[c]` is d/d(copy c), independent because
`fn` has no cross-batch interaction). This turns D forward+backward passes into ceil(D/C)
batched passes -- same total FLOPs, but batched matmuls instead of batch-1 ops (a large GPU
utilization win) and no vmap over the GDN `custom_function` (which has no vmap rule; the
batch axis is one the GDN kernel already handles -- see check_qwen3_5_synthetic.py's B=2 arm).
"""
from __future__ import annotations

import mlx.core as mx

#: Output-dim rows per batched VJP. Trades memory (C copies of the tail's activations) for
#: throughput. 128 is a safe default for the 27B (d_model 5120, short prompts); raise it when
#: memory allows, drop toward 1 for very deep tails or long sequences. chunk_size=1 reproduces
#: the original one-VJP-per-dim path exactly.
CHUNK_SIZE_DEFAULT = 128


def jacobian_via_vjp(fn, h, valid, *, chunk_size: int = CHUNK_SIZE_DEFAULT, progress=None):
    """End-to-end Jacobian of `fn` at `h`, Anthropic's estimator (dim-batched).

        J[i, j] = mean_{p in valid} sum_{p' in valid} d(fn(h)[p', i]) / d(h[p, j])

    Args:
        fn: h[1, S, D] -> [1, S, D] -- the tail (decoder blocks from a source layer to
            the target). Must process the batch axis independently (all decoder tails do;
            attention is within-sequence). For an empty tail (l == target) fn is the
            identity and J = I.
        h:  [1, S, D] source-layer residual (the linearization point).
        valid: 1-D mx.array of position indices -- summed over as targets, averaged over
            as sources (the valid_position mask: skip attention sinks + the last position).
        chunk_size: output-dim rows batched through one mx.vjp (see CHUNK_SIZE_DEFAULT).
        progress: optional `fn(done_chunks, total_chunks)` called after each chunk completes
            (1-indexed `done_chunks`, final call has `done_chunks == total_chunks`). Purely an
            observability hook -- default None reproduces the exact prior behavior/timing.

    Returns:
        [D, D] float32. Row i = output dim i; transport is `residual @ J.T` (i.e. J @ h).
    """
    S, D = h.shape[1], h.shape[-1]
    C = max(1, int(chunk_size))
    total_chunks = -(-D // C)  # ceil(D / C)
    pos_mask = mx.zeros((S,), dtype=h.dtype).at[valid].add(1.0)  # 1 at valid positions
    eye = mx.eye(D, dtype=h.dtype)                               # rows = output-dim one-hots
    rows = []
    done_chunks = 0
    for lo in range(0, D, C):
        dims = list(range(lo, min(lo + C, D)))
        c = len(dims)
        h_rep = mx.repeat(h, c, axis=0)                         # [c, S, D] independent copies
        # cot[b] hot at (valid positions, output dim dims[b]).
        onehots = eye[mx.array(dims)]                           # [c, D]
        cot = pos_mask[None, :, None] * onehots[:, None, :]     # [c, S, D]
        _, g = mx.vjp(fn, [h_rep], [cot])
        grad = g[0]                                             # [c, S, D]
        # Row b = avg over valid source positions of the gradient for copy b.
        chunk_rows = grad[:, valid, :].astype(mx.float32).mean(axis=1)  # [c, D]
        rows.append(chunk_rows)
        mx.eval(rows[-1])                                       # bound graph/memory growth
        done_chunks += 1
        if progress is not None:
            progress(done_chunks, total_chunks)
    return mx.concatenate(rows, axis=0)                         # [D, D], row d = output dim
