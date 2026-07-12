# PORTED (not vendored) from WeZZard/jlens-qwen36 (Apache-2.0):
# jlens_qwen/{custom_gdn_vjp,custom_gdn_patch,patch_gdn}.py -- the Metal GDN backward
# kernel (its "v4" layout, incl. the real dg/dbeta gate gradients) and the fit-time
# GDN dispatch. Re-verified vs mx.vjp here (scripts/check_qwen3_5_synthetic.py) per
# the project rule: port + verify, never vendor + trust. Adaptations from the
# reference are listed in the docstring below. See NOTICE.
"""qwen3_5 (Gated-DeltaNet hybrid) tail + GDN speed accelerator.

The direct-VJP fitter (jlens_mlx.fit) needs two things the generic tail can't give
qwen3_5:

1. **A qwen3_5-aware tail** -- the arch dispatches per-layer masks: the "causal"
   string (SDPA) for the full-attention layers, ``create_ssm_mask`` (None when
   uncached) for the 48 Gated-DeltaNet layers, chosen by ``layer.is_linear`` --
   mirroring ``Qwen3_5TextModel.__call__``.
2. **A differentiable (and fast) GDN recurrence.** MLX's fused GDN kernel
   (``gated_delta_kernel``) has no VJP, and the differentiable ops fallback
   (``gated_delta_ops``) is a per-timestep Python loop -- slow forward AND its
   autodiff backward retains per-step state (memory blow-up at 27B scale). The
   ported Metal backward kernel re-runs the forward in registers and does the
   reverse scan on-GPU, so the fused kernel stays on the forward and the backward
   is one kernel launch per layer.

Adaptations from the jlens-qwen36 reference (each covered by the synthetic gate):

- ``gated_delta_update`` is swapped via a **context manager** (both module refs:
  ``gated_delta`` and ``qwen3_5``'s from-import), not a permanent class patch of
  ``GatedDeltaNet.__call__``. The forward math is byte-identical to stock (same
  fused kernel); only the recurrence's VJP is customized -- so there is no
  replicated forward to drift out of sync with mlx-lm.
- The atomic accumulation buffers are kernel **outputs** with ``init_value=0``
  (the reference passed pre-zeroed inputs and mutated them through casts).
- dg/dbeta (decay/write-gate gradients) are ALWAYS returned (reference kernel v4
  behavior) -- the direct end-to-end Jacobian should include the x -> a/b -> gate
  paths (the reference measured them at ~5-8% of ||M||).
- Threadgroup shared arrays are sized by the ``n_per_t`` template constant
  (reference hardcoded the Dk=192 case).
- NOT ported: ``analytic_attn/analytic_layer`` (per-layer analytic assembly) and
  the chain-multiply fit -- artifacts of the chain design this project pivoted
  away from -- and the ``mx.checkpoint`` wrap (only the ops fallback needs it,
  and that path is for small models / no-Metal verification only).

Fit-path semantics: the patched update returns ``stop_gradient(state_in)`` as the
output state on the kernel path. That is NOT the true post-sequence state -- fine
for fitting, where every call is uncached (``cache=None``) and the returned state
is discarded. Do not run cached generation inside ``gdn_fit_patch()``.

The served 27B: 64 layers, ``full_attention_interval=4`` -> 48 GDN + 16 FA,
Hk=16 / Hv=48 (GQA repeat 3), Dk=Dv=128 (n_per_t=4). Reach its text stack via
``.language_model.model`` (capture.ModelAdapter already does).
"""
from __future__ import annotations

from contextlib import contextmanager

import mlx.core as mx

#: Compile-time bound on sequence length in the backward kernel (per-thread
#: replay buffers are sized [MAX_T][n_per_t]). Anthropic-style fits use
#: 128-token prompts, which this exactly covers; longer prompts fall back to ops.
MAX_T = 128

#: Bound on Dk/32 (per-thread state slots + threadgroup memory). Real model: 4.
_MAX_N_PER_T = 6

#: Test hook: set False to force the differentiable ops fallback everywhere
#: (the mx.vjp ground-truth arm of the synthetic check).
KERNEL_ENABLED = True

_BACKWARD_KERNEL = None
_PATCH_DEPTH = 0
_SAVED_REFS: list = []


def _get_backward_kernel():
    """Lazily build + cache the Metal backward kernel (None when Metal absent).

    Grid (32, Dv, B*Hv); threadgroup (32, 4, 1): 32 dk-threads (one simdgroup,
    simd_sum reduces over Dk) x 4 dv-threads. Phase 1 replays the forward,
    storing per-t PRE-decay states (so dg needs no division by a saturated
    gate) and deltas; phase 2 runs the reverse scan, reducing the 4 dv-threads
    through threadgroup memory and accumulating across dv-threadgroups via
    atomics into per-Hv buffers (Python reduces Hv -> Hk).
    """
    global _BACKWARD_KERNEL
    if _BACKWARD_KERNEL is not None:
        return _BACKWARD_KERNEL
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        auto dk_idx = thread_position_in_threadgroup.x;        // 0..31
        auto dv_idx = thread_position_in_grid.y;               // 0..Dv-1
        auto tg_dv = thread_position_in_threadgroup.y;         // 0..3
        constexpr int n_per_t = Dk / 32;
        constexpr int MAX_T = 128;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        // v, dy: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto dy_ = dy + b_idx * T * Hv * Dv + hv_idx * Dv;
        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;
        // state_in: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          state[i] = i_state[n_per_t * dk_idx + i];
        }

        // Phase 1: forward replay, storing per-t s_pre (PRE-decay state) and
        // delta. s_pre (not s_dec) is stored so dg = <ds_dec, s_pre> needs no
        // division by g_t (which underflows for saturated gates -> 0/0).
        float s_pre_store[MAX_T][n_per_t];
        float delta_store[MAX_T];

        for (int t = 0; t < T; ++t) {
          float g_t = g_[hv_idx];
          for (int i = 0; i < n_per_t; ++i) {
            s_pre_store[t][i] = state[i];
            state[i] *= g_t;
          }
          float kv_mem_partial = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            kv_mem_partial += state[i] * k_[n_per_t * dk_idx + i];
          }
          float kv_mem = simd_sum(kv_mem_partial);
          float beta_t = beta_[hv_idx];
          float delta = (v_[dv_idx] - kv_mem) * beta_t;
          delta_store[t] = delta;
          for (int i = 0; i < n_per_t; ++i) {
            state[i] += k_[n_per_t * dk_idx + i] * delta;
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          dy_ += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        // Phase 2: reverse scan.
        float s_bar[n_per_t];
        for (int i = 0; i < n_per_t; ++i) s_bar[i] = 0.0f;

        q_ -= Hk * Dk;
        k_ -= Hk * Dk;
        v_ -= Hv * Dv;
        dy_ -= Hv * Dv;
        g_ -= Hv;
        beta_ -= Hv;

        // Shared memory reduces the 4 dv-threads' dq/dk partials (4x fewer
        // atomics) plus per-dv dg/dbeta scalars.
        threadgroup float dq_shared[32 * n_per_t * 4];
        threadgroup float dk_shared[32 * n_per_t * 4];
        threadgroup float dg_shared[4];
        threadgroup float db_shared[4];

        for (int t = T - 1; t >= 0; --t) {
          float g_t = g_[hv_idx];
          float beta_t = beta_[hv_idx];
          float dy_val = dy_[dv_idx];

          float d_delta_partial = 0.0f;
          float kv_mem_partial = 0.0f;
          float ds_post[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            ds_post[i] = dy_val * q_[s_idx] + s_bar[i];
            d_delta_partial += ds_post[i] * k_[s_idx];
            kv_mem_partial += (s_pre_store[t][i] * g_t) * k_[s_idx];
          }
          float d_delta = simd_sum(d_delta_partial);
          float kv_mem = simd_sum(kv_mem_partial);
          float d_kv_mem = -d_delta * beta_t;

          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            float s_dec_val = s_pre_store[t][i] * g_t;
            float s_post_val = s_dec_val + k_[s_idx] * delta_store[t];
            dq_shared[(dk_idx * n_per_t + i) * 4 + tg_dv] = dy_val * s_post_val;
            dk_shared[(dk_idx * n_per_t + i) * 4 + tg_dv] =
                ds_post[i] * delta_store[t] + d_kv_mem * s_dec_val;
          }

          // dv: one writer per (b, t, hv, dv) slot; add == store on a
          // zero-initialized atomic output.
          if (thread_index_in_simdgroup == 0) {
            atomic_fetch_add_explicit(
                &dv_out[(b_idx * T * Hv * Dv) + (t * Hv * Dv) + (hv_idx * Dv) + dv_idx],
                d_delta * beta_t, memory_order_relaxed);
          }

          // s_bar = ds_dec * g, and the dg partial dg = <ds_dec, s_pre>
          // (this thread: its dk slots, its dv).
          float dg_partial = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            float ds_dec = ds_post[i] + d_kv_mem * k_[s_idx];
            dg_partial += ds_dec * s_pre_store[t][i];
            s_bar[i] = ds_dec * g_t;
          }
          float dg_dv = simd_sum(dg_partial);  // summed over dk; one per dv
          if (thread_index_in_simdgroup == 0) {
            dg_shared[tg_dv] = dg_dv;
            // dbeta = d_delta * (v - kv_mem); d_delta already dk-summed.
            db_shared[tg_dv] = d_delta * (v_[dv_idx] - kv_mem);
          }

          // Reduce the 4 dv-threads' partials, then one atomic add per slot.
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (tg_dv == 0) {
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              float dq_sum = 0.0f, dk_sum = 0.0f;
              for (int d = 0; d < 4; ++d) {
                dq_sum += dq_shared[(dk_idx * n_per_t + i) * 4 + d];
                dk_sum += dk_shared[(dk_idx * n_per_t + i) * 4 + d];
              }
              auto off = (b_idx * T * Hv * Dk) + (t * Hv * Dk) + (hv_idx * Dk) + s_idx;
              atomic_fetch_add_explicit(&dq_hv_buf[off], dq_sum, memory_order_relaxed);
              atomic_fetch_add_explicit(&dk_hv_buf[off], dk_sum, memory_order_relaxed);
            }
            if (dk_idx == 0) {
              float dg_sum = 0.0f, db_sum = 0.0f;
              for (int d = 0; d < 4; ++d) {
                dg_sum += dg_shared[d];
                db_sum += db_shared[d];
              }
              auto goff = (b_idx * T * Hv) + (t * Hv) + hv_idx;
              atomic_fetch_add_explicit(&dg_hv_buf[goff], dg_sum, memory_order_relaxed);
              atomic_fetch_add_explicit(&db_hv_buf[goff], db_sum, memory_order_relaxed);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          q_ -= Hk * Dk;
          k_ -= Hk * Dk;
          v_ -= Hv * Dv;
          dy_ -= Hv * Dv;
          g_ -= Hv;
          beta_ -= Hv;
        }
    """
    _BACKWARD_KERNEL = mx.fast.metal_kernel(
        name="jlens_gdn_backward",
        input_names=["q", "k", "v", "dy", "g", "beta", "state_in", "T"],
        output_names=["dv_out", "dq_hv_buf", "dk_hv_buf", "dg_hv_buf", "db_hv_buf"],
        atomic_outputs=True,
        source=source,
    )
    return _BACKWARD_KERNEL


def gdn_kernel_vjp(q, k, v, g, beta, state, dy):
    """(dq, dk, dv, dg, dbeta) via the Metal backward kernel (fp32 internally).

    q, k: [B, T, Hk, Dk]. v, dy: [B, T, Hv, Dv]. g, beta: [B, T, Hv] (scalar
    gating). state: [B, Hv, Dv, Dk]. Caller guarantees eligibility
    (see _kernel_eligible). Returns grads in the inputs' dtypes; dq/dk are
    reduced from Hv back to Hk (sum over the GQA repeats).
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    rf = Hv // Hk

    outs = _get_backward_kernel()(
        inputs=[a.astype(mx.float32) for a in (q, k, v, dy, g, beta, state)] + [T],
        template=[("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dk), (B, T, Hv, Dk),
                       (B, T, Hv), (B, T, Hv)],
        output_dtypes=[mx.float32] * 5,
        init_value=0,
    )
    dv_out, dq_hv, dk_hv, dg_hv, db_hv = outs

    if rf > 1:
        dq = dq_hv.reshape(B, T, Hk, rf, Dk).sum(axis=-2)
        dk = dk_hv.reshape(B, T, Hk, rf, Dk).sum(axis=-2)
    else:
        dq, dk = dq_hv, dk_hv
    return (dq.astype(q.dtype), dk.astype(k.dtype), dv_out.astype(v.dtype),
            dg_hv.astype(g.dtype), db_hv.astype(beta.dtype))


@mx.custom_function
def _gdn_fit_recurrence(q, k, v, g, beta, state):
    """Forward = the stock fused Metal kernel; VJP = the ported backward kernel.

    Post-projection inputs only (g/beta are computed OUTSIDE, so autograd chains
    dg/dbeta back through compute_g/sigmoid to the layer input). Returns y.
    """
    from mlx_lm.models.gated_delta import gated_delta_kernel
    y, _ = gated_delta_kernel(q, k, v, g, beta, state, mask=None)
    return y


@_gdn_fit_recurrence.vjp
def _gdn_fit_recurrence_vjp(primals, cotangent, output):
    q, k, v, g, beta, state = primals
    dq, dk, dv, dg, dbeta = gdn_kernel_vjp(q, k, v, g, beta, state, cotangent)
    # state is a zeros constant on the fit path -- no gradient flows through it.
    return dq, dk, dv, dg, dbeta, mx.zeros_like(state)


def _kernel_eligible(T, Hk, Hv, Dk, Dv, g, mask) -> bool:
    return (KERNEL_ENABLED
            and mask is None
            and g.ndim == 3                       # scalar gating (qwen3_5)
            and mx.metal.is_available() and mx.default_device() == mx.gpu
            and T <= MAX_T
            and Dk % 32 == 0 and Dk // 32 <= _MAX_N_PER_T
            and Dv % 4 == 0
            and Hv % Hk == 0)


def _fit_gated_delta_update(q, k, v, a, b, A_log, dt_bias,
                            state=None, mask=None, use_kernel=True,
                            **_future_kwargs):
    """Drop-in for ``gated_delta_update`` on the fit path (see gdn_fit_patch).

    Kernel-eligible: stock fused forward + custom Metal VJP. Otherwise: the
    stock differentiable ops loop. The returned state is only meaningful on the
    ops path; kernel path returns stop_gradient(state_in) -- callers under the
    fit patch are uncached (cache=None) and never read it.

    ``**_future_kwargs`` absorbs keywords upstream later adds to the real
    signature (e.g. the ``training=`` flag an open mlx-lm PR passes
    unconditionally at every qwen3_5 call site), so a pin bump can't
    TypeError mid-fit. The fit path is differentiable-by-construction, so
    such flags are safely ignored -- but warn once so drift is visible.
    """
    if _future_kwargs and not getattr(_fit_gated_delta_update, "_warned", False):
        _fit_gated_delta_update._warned = True  # type: ignore[attr-defined]
        print(f"gdn_fit_patch: ignoring unknown gated_delta_update kwargs "
              f"{sorted(_future_kwargs)} (upstream signature drift -- verify fit parity)",
              flush=True)
    from mlx_lm.models.gated_delta import compute_g, gated_delta_ops

    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if _kernel_eligible(T, Hk, Hv, Dk, Dv, g, mask):
        y = _gdn_fit_recurrence(q, k, v, g, beta, state)
        return y, mx.stop_gradient(state)
    return gated_delta_ops(q, k, v, g, beta, state, mask)


@contextmanager
def gdn_fit_patch():
    """Swap ``gated_delta_update`` for the differentiable fit version.

    Patches BOTH references (``mlx_lm.models.gated_delta`` and ``qwen3_5``'s
    from-import copy). Reentrant; restores the originals on exit. Only tail
    runners should hold this open -- plain forwards (capture, generation) want
    the stock path.
    """
    import mlx_lm.models.gated_delta as _gd
    import mlx_lm.models.qwen3_5 as _q35

    global _PATCH_DEPTH
    if _PATCH_DEPTH == 0:
        _SAVED_REFS[:] = [_gd.gated_delta_update, _q35.gated_delta_update]
        _gd.gated_delta_update = _fit_gated_delta_update
        _q35.gated_delta_update = _fit_gated_delta_update
    _PATCH_DEPTH += 1
    try:
        yield
    finally:
        _PATCH_DEPTH -= 1
        if _PATCH_DEPTH == 0:
            _gd.gated_delta_update, _q35.gated_delta_update = _SAVED_REFS
            _SAVED_REFS.clear()


def make_qwen3_5_tail(adapter, start: int, end: int):
    """The GDN-aware analogue of fit.make_tail: blocks [start, end) with the
    per-layer fa/ssm mask dispatch (``layer.is_linear``), run under
    ``gdn_fit_patch`` so the GDN recurrence is differentiable (and fast).
    Mirrors ``Qwen3_5TextModel.__call__``'s mask construction exactly."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    blocks = adapter.layers

    def tail(h: mx.array) -> mx.array:
        with gdn_fit_patch():
            fa_mask = create_attention_mask(h, cache=None)
            ssm_mask = create_ssm_mask(h, cache=None)
            for i in range(start, end):
                layer = blocks[i]
                h = layer(h, ssm_mask if layer.is_linear else fa_mask, cache=None)
        return h

    return tail
