"""VJP primitive for direct end-to-end Jacobian fitting.

Ports the estimator from anthropics/jacobian-lens (fitting.py::jacobian_for_prompt) to
MLX. For a function `fn` (the "tail" from a source layer's residual to the target block's
residual), compute the position-averaged, future-summed Jacobian via one `mx.vjp` per
output dimension. Correct by construction (autograd); no closed form, so no rms**2/rms**3
class of bug.
"""
from __future__ import annotations

import mlx.core as mx


def jacobian_via_vjp(fn, h, valid):
    """End-to-end Jacobian of `fn` at `h`, Anthropic's estimator.

        J[i, j] = mean_{p in valid} sum_{p' in valid} d(fn(h)[p', i]) / d(h[p, j])

    Args:
        fn: h[1, S, D] -> [1, S, D] -- the tail (decoder blocks from a source layer to
            the target). For an empty tail (l == target) fn is the identity and J = I.
        h:  [1, S, D] source-layer residual (the linearization point).
        valid: 1-D mx.array of position indices -- summed over as targets, averaged over
            as sources (the valid_position mask: skip attention sinks + the last position).

    Returns:
        [D, D] float32. Row i = output dim i; transport is `residual @ J.T` (i.e. J @ h).

    One `mx.vjp` per output dim (no dim-batching yet -- a later speedup).
    """
    S, D = h.shape[1], h.shape[-1]
    pos_mask = mx.zeros((S,), dtype=h.dtype).at[valid].add(1.0)  # 1 at valid positions
    rows = []
    for d in range(D):
        onehot = mx.zeros((D,), dtype=h.dtype).at[d].add(1.0)
        cot = pos_mask[None, :, None] * onehot[None, None, :]    # [1, S, D], hot at (valid, d)
        _, g = mx.vjp(fn, [h], [cot])
        grad = g[0][0]                                           # [S, D]
        rows.append(grad[valid].astype(mx.float32).mean(axis=0))  # avg over valid source pos
        if (d & 127) == 0:
            mx.eval(rows)                                        # bound graph/memory growth
    return mx.stack(rows)                                        # [D, D], row d = output dim
