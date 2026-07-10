"""Atomic verification of the closed-form RMSNorm Jacobian (the chain SEED).

The fit chains J_{l-1} = J_l @ M_l seeded by J_L = d(final_norm)/d(acts). For
RMSNorm models (qwen3_5 heretic, gemma) that seed is computed in closed form
(vendor/jlens_qwen36/analytic.py::rms_norm_jacobian) instead of a 5120-VJP pass.
This proves that closed form matches mx.vjp autograd -- both the single-position
exact Jacobian AND the position-averaging convention (a real source of bugs).

No model download; runs in seconds. Run under an env with mlx:
    uv run python scripts/check_rmsnorm_seed.py
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np


# jlens-qwen36's analytic.py::rms_norm_jacobian, reproduced to demonstrate the bug: its
# rank-1 term is over rms**2. The correct derivative (rms_norm_jacobian_derived below) is
# over rms**3, verified against mx.vjp. We do NOT use this closed form -- the fitter follows
# Anthropic (autograd, norm outside J), so there is no closed-form seed to get wrong.
def rms_norm_jacobian(x, weight, eps=1e-6, valid_mask=None):
    S, D = x.shape
    xf = x.astype(mx.float32)
    ms = (xf * xf).mean(axis=-1, keepdims=True) + eps
    rms = mx.sqrt(ms)
    w = weight.astype(mx.float32)
    mask = mx.ones((S,), dtype=mx.float32) if valid_mask is None else valid_mask.astype(mx.float32)
    n_valid = float(mask.sum().tolist())
    weighted_inv_rms = mask / rms[:, 0]
    term1_diag = w * (weighted_inv_rms.sum() / n_valid)
    x_normed = xf / rms
    wx_normed = w * x_normed
    m = mask[:, None]
    term2 = mx.matmul((m * wx_normed).T, x_normed) / (n_valid * D)
    return -term2 + mx.diag(term1_diag)


def rms_norm_jacobian_derived(x, weight, eps=1e-6, valid_mask=None):
    """Hand-derived: dy_i/dx_j = w_i(d_ij/rms - x_i x_j/(D*rms**3)), pos-averaged.
    Same shape as the vendored fn but the rank-1 term is over rms**3, not rms**2."""
    S, D = x.shape
    xf = x.astype(mx.float32)
    w = weight.astype(mx.float32)
    rms = mx.sqrt((xf * xf).mean(axis=-1, keepdims=True) + eps)  # [S,1]
    mask = mx.ones((S,), dtype=mx.float32) if valid_mask is None else valid_mask.astype(mx.float32)
    n = float(mask.sum().tolist())
    diag = w * ((mask / rms[:, 0]).sum() / n)
    x3 = xf / (rms ** 1.5)                       # split rms**3 as rms**1.5 each side
    rank1 = mx.matmul((mask[:, None] * (w * x3)).T, x3) / (n * D)
    return -rank1 + mx.diag(diag)


def rmsnorm_fwd(x1d, w, eps):
    ms = (x1d * x1d).mean() + eps
    return x1d * mx.rsqrt(ms) * w


def dense_jacobian_via_vjp(x1d, w, eps):
    """[D, D] true Jacobian dy_i/dx_j via D one-hot VJPs (row i = output i)."""
    D = x1d.shape[0]
    rows = []
    for i in range(D):
        cot = mx.zeros((D,)).at[i].add(1.0)
        _, g = mx.vjp(lambda z: rmsnorm_fwd(z, w, eps), [x1d], [cot])
        rows.append(np.array(g[0]))
    return np.stack(rows)  # [D, D]


def main() -> int:
    mx.random.seed(0)
    D, eps = 8, 1e-5
    w = mx.random.normal((D,)) * 0.5 + 1.0

    # --- Check 1: single position, exact Jacobian ---
    x1 = mx.random.normal((D,))
    ref1 = dense_jacobian_via_vjp(x1, w, eps)
    vend1 = float(np.abs(np.array(rms_norm_jacobian(x1[None, :], w, eps, mx.ones((1,)))) - ref1).max())
    der1 = float(np.abs(np.array(rms_norm_jacobian_derived(x1[None, :], w, eps, mx.ones((1,)))) - ref1).max())
    print(f"single-position : vendored(rms^2) err={vend1:.2e}   derived(rms^3) err={der1:.2e}")

    # --- Check 2: position-averaging (skip pos 0, average 1..S-1) ---
    S = 5
    xs = mx.random.normal((S, D))
    mask = mx.array([0.0, 1.0, 1.0, 1.0, 1.0])
    valid = [1, 2, 3, 4]
    ref2 = np.mean([dense_jacobian_via_vjp(xs[s], w, eps) for s in valid], axis=0)
    vend2 = float(np.abs(np.array(rms_norm_jacobian(xs, w, eps, mask)) - ref2).max())
    der2 = float(np.abs(np.array(rms_norm_jacobian_derived(xs, w, eps, mask)) - ref2).max())
    print(f"pos-averaged    : vendored(rms^2) err={vend2:.2e}   derived(rms^3) err={der2:.2e}")

    ok = der1 < 1e-4 and der2 < 1e-4
    print(f"\nDERIVED (rms^3) SEED {'PASS' if ok else 'FAIL'} | vendored closed-form "
          f"{'MATCHES' if vend1 < 1e-4 else 'DIVERGES from'} mx.vjp")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
