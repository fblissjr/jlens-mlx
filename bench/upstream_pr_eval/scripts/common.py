"""Shared helpers for the PR evaluation scripts."""
import importlib.util
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# This file lives at bench/upstream_pr_eval/scripts/common.py -- walk up to
# the jlens-mlx repo root so the harness has no hardcoded user path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
JLENS_GDN = _REPO_ROOT / "jlens_mlx" / "providers" / "qwen3_5_gdn.py"

# Realistic Qwen3.5-27B GDN dims (from modelzoo config.json):
# linear_num_key_heads=16, linear_num_value_heads=48,
# linear_key_head_dim=128, linear_value_head_dim=128
REAL = dict(Hk=16, Hv=48, Dk=128, Dv=128)


def load_jlens_gdn():
    """Import our Metal backward kernel module directly from file (read-only,
    bypasses jlens_mlx package __init__)."""
    spec = importlib.util.spec_from_file_location("qwen3_5_gdn_ro", str(JLENS_GDN))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_inputs(B, T, Hk, Hv, Dk, Dv, dtype=mx.float32, seed=0):
    """Random GDN inputs. q/k L2-normalized along Dk (delta-rule stability,
    matches model behavior); g via compute_g on random a/A_log/dt_bias so the
    decay values are realistic (in (0,1), often near 1)."""
    mx.random.seed(seed)
    q = mx.random.normal((B, T, Hk, Dk))
    k = mx.random.normal((B, T, Hk, Dk))
    q = q / mx.linalg.norm(q, axis=-1, keepdims=True)
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    v = 0.5 * mx.random.normal((B, T, Hv, Dv))
    a = mx.random.normal((B, T, Hv))
    b = mx.random.normal((B, T, Hv))
    A_log = mx.random.uniform(low=0.0, high=2.0, shape=(Hv,))
    dt_bias = 0.5 * mx.random.normal((Hv,))
    g = mx.exp(-mx.exp(A_log) * nn.softplus(a + dt_bias))  # [B,T,Hv], fp32
    beta = mx.sigmoid(b)
    dy = mx.random.normal((B, T, Hv, Dv))
    out = dict(q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
               g=g, beta=beta, dy=dy)
    for key in ("q", "k", "v", "g", "beta", "dy"):
        out[key] = out[key].astype(dtype)
    mx.eval(*out.values())
    return out


def stats(test, ref):
    """(max_abs, rel = max_abs/max|ref|, cosine) in fp32."""
    t = test.astype(mx.float32)
    r = ref.astype(mx.float32)
    d = mx.abs(t - r).max().item()
    m = mx.abs(r).max().item()
    num = (t * r).sum()
    den = mx.sqrt((t * t).sum()) * mx.sqrt((r * r).sum()) + 1e-30
    return d, d / (m + 1e-30), (num / den).item()


def fmt(name, s):
    return f"{name:<14s} max_abs={s[0]:.3e}  rel={s[1]:.3e}  cos={s[2]:.9f}"


def timed(fn, warmup=1, iters=2):
    """Returns (mean_seconds, peak_gb). fn must mx.eval its outputs."""
    for _ in range(warmup):
        fn()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = (time.perf_counter() - t0) / iters
    return dt, mx.get_peak_memory() / 2**30
