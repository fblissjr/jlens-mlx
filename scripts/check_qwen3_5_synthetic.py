"""Synthetic qwen3_5 gate -- verifies the ported GDN tail + Metal backward vs mx.vjp.

No download: builds a TINY random-weight qwen3_5 (GDN hybrid, GQA repeat 3 like the real
27B) and checks the port at three grains, per the port-then-verify rule (the rms**2 lesson):

  [1] KERNEL: the ported Metal backward (dq/dk/dv/dg/dbeta) == mx.vjp through the stock
      differentiable ``gated_delta_ops`` on raw GDN tensors (incl. B=2 and GQA rf=3).
  [2] FORWARD PARITY: the patched tail's forward == the same blocks run stock (the patch
      must change the VJP only, never the forward).
  [3] FIT PARITY: fit_prompt J (kernel path) == fit_prompt J with KERNEL_ENABLED=False
      (pure mx.vjp through the stock ops loop -- the autodiff ground truth), plus
      J_target == I and finite J. Also asserts make_tail actually dispatches to the
      qwen3_5 tail.

Random weights, so this is a correctness/plumbing gate, NOT a real-weights fidelity check
(that is the 27B own-fit's held-out gate). Metal-gated: [1]+[3] compare the kernel path,
which needs a GPU; without Metal the script degrades to ops-only and says so.

Run:  uv run python scripts/check_qwen3_5_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_ops
from mlx_lm.models.qwen3_5 import Model, ModelArgs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jlens_mlx.providers.qwen3_5_gdn as gdn  # noqa: E402
from jlens_mlx.capture import ModelAdapter  # noqa: E402
from jlens_mlx.fit import _model_type, fit_prompt, make_tail  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def tiny_qwen3_5() -> Model:
    """8 layers, full_attention_interval=4 -> layers 3,7 are full-attention, 6 are GDN.
    GDN dims mirror the real 27B's structure (GQA rf=3, Dk a multiple of 32)."""
    text_config = dict(
        model_type="qwen3_5_text", hidden_size=64, num_hidden_layers=8,
        intermediate_size=128, num_attention_heads=4, head_dim=16,
        num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=256,
        linear_num_value_heads=6, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, full_attention_interval=4,
        tie_word_embeddings=False,
    )
    m = Model(ModelArgs(model_type="qwen3_5", text_config=text_config))
    m.eval()
    mx.eval(m.parameters())
    return m


def check_kernel_vs_vjp() -> None:
    """[1] the Metal backward vs mx.vjp through stock gated_delta_ops.

    q/k are rms-normalized + inv_scale'd exactly as GatedDeltaNet does before the
    recurrence (raw randn q/k make the recurrence explode over T steps, so an
    absolute-error gate is meaningless there). Graded on RELATIVE max error +
    cosine per gradient -- both arms are fp32 with different summation orders.
    """
    print("[1] Metal backward kernel vs mx.vjp(gated_delta_ops)")
    for label, (B, T, Hk, Dk, Hv, Dv) in {
        "rf=3, n_per_t=2":          (1, 7, 2, 64, 6, 8),
        "rf=3, n_per_t=1, B=2":     (2, 5, 2, 32, 6, 8),
        "real-ratio (n_per_t=4)":   (1, 16, 2, 128, 6, 128),
        "real-ratio, T=128":        (1, 128, 2, 128, 6, 128),
    }.items():
        inv_scale = Dk ** -0.5
        q = (inv_scale ** 2) * mx.fast.rms_norm(mx.random.normal((B, T, Hk, Dk)), None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(mx.random.normal((B, T, Hk, Dk)), None, 1e-6)
        v = mx.random.normal((B, T, Hv, Dv))
        g = mx.random.uniform(shape=(B, T, Hv))       # decay gate in (0,1)
        beta = mx.random.uniform(shape=(B, T, Hv))    # write gate in (0,1)
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        dy = mx.random.normal((B, T, Hv, Dv))

        def fwd(q, k, v, g, beta):
            y, _ = gated_delta_ops(q, k, v, g, beta, state, None)
            return y

        _, refs = mx.vjp(fwd, [q, k, v, g, beta], [dy])
        got = gdn.gdn_kernel_vjp(q, k, v, g, beta, state, dy)
        rels, coss = {}, {}
        for n, r, o in zip(("dq", "dk", "dv", "dg", "dbeta"), refs, got):
            a = np.asarray(r, dtype=np.float64).ravel()
            b = np.asarray(o, dtype=np.float64).ravel()
            rels[n] = float(np.abs(a - b).max() / (np.abs(a).max() + 1e-12))
            coss[n] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        ok = max(rels.values()) < 5e-4 and min(coss.values()) > 0.999999
        check(f"kernel [{label}]", ok,
              " ".join(f"{n}:rel={rels[n]:.1e},cos={coss[n]:.6f}" for n in rels))


def _stock_tail(blocks, start, end):
    """The same block loop the qwen3_5 tail runs, but fully stock (no patch):
    the fused-kernel forward. Forward-parity reference for [2]."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    def tail(h):
        fa = create_attention_mask(h, cache=None)
        ssm = create_ssm_mask(h, cache=None)
        for i in range(start, end):
            layer = blocks[i]
            h = layer(h, ssm if layer.is_linear else fa, cache=None)
        return h

    return tail


def check_forward_parity(model, ad) -> None:
    """[2] the patched tail must not change the forward."""
    print("[2] forward parity: patched tail vs stock blocks")
    S, D = 12, ad.layers[0].input_layernorm.weight.shape[0]
    h = mx.random.normal((1, S, D)) * 0.1
    tail = make_tail(ad, 0, ad.n_layers)
    y_patched = tail(h)
    y_stock = _stock_tail(ad.layers, 0, ad.n_layers)(h)
    err = float(mx.abs(y_patched - y_stock).max().item())
    scale = float(mx.abs(y_stock).max().item())
    check("forward parity", err <= 1e-5 * max(scale, 1.0),
          f"max_abs_err={err:.2e} (|y|max={scale:.2e})")


def check_fit_parity(model, ad) -> None:
    """[3] J via the kernel path vs J via pure mx.vjp (ops), + structure checks."""
    print("[3] fit parity: kernel-VJP J vs pure-autodiff J")
    tail_fn = make_tail(ad, 0, 1)
    check("make_tail dispatch",
          tail_fn.__qualname__.startswith("make_qwen3_5_tail"),
          f"model_type={_model_type(ad)!r} tail={tail_fn.__qualname__}")

    n = ad.n_layers
    target = n - 1
    ids = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
    # Source layers: one whose tail crosses GDN + FA layers, plus the identity.
    sources = [2, target]

    assert gdn.KERNEL_ENABLED and mx.metal.is_available()
    J_kernel, _ = fit_prompt(model, ids, sources, adapter=ad,
                             target_layer=target, skip_first=2)
    gdn.KERNEL_ENABLED = False
    try:
        J_ops, _ = fit_prompt(model, ids, sources, adapter=ad,
                              target_layer=target, skip_first=2)
    finally:
        gdn.KERNEL_ENABLED = True

    D = J_kernel[target].shape[0]
    ident = float(mx.abs(J_kernel[target] - mx.eye(D)).max().item())
    check("J_target == I", ident < 1e-4, f"max|J-I|={ident:.2e}")

    a = np.asarray(J_kernel[2], dtype=np.float64).ravel()
    b = np.asarray(J_ops[2], dtype=np.float64).ravel()
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    check("J finite", finite, f"|J|max kernel={np.abs(a).max():.2e} ops={np.abs(b).max():.2e}")
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    rel = float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))
    check("J kernel==ops", cos > 0.9999 and rel < 5e-3,
          f"cos={cos:.6f} max_rel_err={rel:.2e}")


def check_dim_batching(model, ad) -> None:
    """[4] dim-batched J (chunk_size>1) == one-at-a-time J (chunk_size=1).

    Same estimator, just batched through fn's native batch axis -- must match to
    fp32 round-off. Guards the GDN custom_function working under a batched primal.
    """
    print("[4] dim-batching: chunked J == chunk_size=1 J")
    n = ad.n_layers
    target = n - 1
    ids = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
    src = 2
    J1, _ = fit_prompt(model, ids, [src], adapter=ad, target_layer=target,
                       skip_first=2, chunk_size=1)
    Jc, _ = fit_prompt(model, ids, [src], adapter=ad, target_layer=target,
                       skip_first=2, chunk_size=7)  # odd chunk -> exercises the ragged tail
    a = np.asarray(J1[src], dtype=np.float64)
    b = np.asarray(Jc[src], dtype=np.float64)
    err = float(np.abs(a - b).max() / (np.abs(a).max() + 1e-12))
    check("batched == unbatched", err < 1e-5, f"max_rel_err={err:.2e}")


def main() -> int:
    mx.random.seed(0)
    metal = mx.metal.is_available()
    print(f"metal={metal}")
    if not metal:
        print("NO METAL: kernel-path checks impossible; ops fallback is stock code. "
              "Run this gate on Apple silicon before trusting the kernel.")
        return 1

    check_kernel_vs_vjp()

    model = tiny_qwen3_5()
    ad = ModelAdapter(model)
    n_gdn = sum(1 for l in ad.layers if l.is_linear)
    print(f"tiny qwen3_5: model_type={_model_type(ad)!r} n_layers={ad.n_layers} "
          f"(GDN={n_gdn}, FA={ad.n_layers - n_gdn})")
    check_forward_parity(model, ad)
    check_fit_parity(model, ad)
    check_dim_batching(model, ad)

    print(f"\nQWEN3_5 GDN GATE {'PASS' if not FAILURES else 'FAIL: ' + ', '.join(FAILURES)}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
