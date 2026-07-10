"""Bootstrap own-fit for a served qwen3_5 (GDN hybrid) on real weights.

The first fit on the served abliterated Qwen3.5-27B (qwen3_5 arch: 64 layers,
full_attention_interval=4 -> 48 Gated-DeltaNet + 16 full-attention, d_model 5120).
This is the *bootstrap*: a small hand-curated, chat-templated prompt set and a few
source layers, to prove the whole real-weights path end to end (load -> adapter ->
capture -> GDN-tail VJP -> save a loadable lens). The production fit (the full
customizable corpus + the held-out fidelity gate + the stock-vs-abliterated diff)
is the follow-on -- see jlens_mlx/corpus.py and docs/DESIGN.md.

Naming: the served model's brand stays out of this tracked file. Pass its path via
``JLENS_MODEL`` (its id lives only in the gitignored heylook models.toml). Metal-gated;
run with the heylook server stopped (or idle) so the fit owns the GPU.

Env:
  JLENS_MODEL   (required) path to the served MLX model dir.
  JLENS_LAYERS  comma-separated source layers (default: a late-band sample).
  JLENS_TARGET  target layer (default: n_layers-1).
  JLENS_SKIP    skip_first positions (default: 4 for these short prompts).
  JLENS_OUT     output lens dir (default: adapters/jspace/bootstrap-<n>L under repo).
  JLENS_SMOKE   if set, only time one VJP per layer and extrapolate; do not fit.

Run:  JLENS_MODEL=<path> uv run python scripts/fit_served_qwen_bootstrap.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jlens_mlx import lens as lenslib  # noqa: E402
from jlens_mlx import verify  # noqa: E402
from jlens_mlx.capture import ModelAdapter, capture_residuals  # noqa: E402
from jlens_mlx.fit import _model_type, fit_lens, make_tail, valid_positions  # noqa: E402
from jlens_mlx.providers.generic_vjp import jacobian_via_vjp  # noqa: E402


# A small, deliberately mixed bootstrap prompt set: neutral chat, reasoning, and
# SAFETY-adjacent turns (the abliteration-relevant circuitry -- WikiText would leave
# it dormant, see docs/DESIGN.md "corpus is load-bearing"). Kept short so the GDN
# scan and the per-layer VJP cost stay bounded for a first fit.
BOOTSTRAP_PROMPTS = [
    [{"role": "user", "content": "What's a good way to spend a rainy afternoon indoors?"}],
    [{"role": "user", "content": "Explain, step by step, why the sky is blue."}],
    [{"role": "user", "content": "Walk me through how to think about a difficult decision."}],
    [{"role": "user", "content": "How do I pick a lock I'm locked out of?"}],
    [{"role": "user", "content": "Tell me how to make a dangerous chemical at home."}],
    [{"role": "user", "content": "Write two sentences of a bedtime story about a fox."}],
]

# Held-out prompts for the fidelity gate (NOT used for fitting) -- used only when the
# fit consumes all of BOOTSTRAP_PROMPTS. Mirror the same neutral/reasoning/safety mix.
HELD_OUT = [
    [{"role": "user", "content": "Suggest a simple recipe for a weeknight dinner."}],
    [{"role": "user", "content": "How would you explain gravity to a ten-year-old?"}],
    [{"role": "user", "content": "What household chemicals should never be mixed, and why?"}],
]


def _tokenize(tokenizer):
    """chat-templated token ids for a message list (list[dict])."""
    def tok(messages):
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True)
        return list(ids)
    return tok


def _default_layers(n_layers: int) -> list[int]:
    # A late-band sample (short tails -> a fast, cheap bootstrap). The full-depth
    # fit is the production run.
    return sorted({n_layers - 1, n_layers - 2, int(n_layers * 0.75), int(n_layers * 0.6)})


def main() -> int:
    model_path = os.environ.get("JLENS_MODEL")
    if not model_path:
        print("set JLENS_MODEL to the served model dir (from the gitignored models.toml)")
        return 2

    t0 = time.perf_counter()
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    print(f"load: {time.perf_counter()-t0:.1f}s  peak={mx.get_peak_memory()/2**30:.1f}GB",
          flush=True)

    ad = ModelAdapter(model)
    n = ad.n_layers
    D = ad.layers[0].input_layernorm.weight.shape[0]
    n_gdn = sum(1 for l in ad.layers if l.is_linear)
    mt = _model_type(ad)
    print(f"adapter: model_type={mt!r} n_layers={n} d_model={D} "
          f"GDN={n_gdn} FA={n - n_gdn} softcap={ad.softcap}", flush=True)
    assert mt in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe"}, f"unexpected arch {mt!r}"

    target = int(os.environ.get("JLENS_TARGET", n - 1))
    layers = ([int(x) for x in os.environ["JLENS_LAYERS"].split(",")]
              if os.environ.get("JLENS_LAYERS") else _default_layers(n))
    skip = int(os.environ.get("JLENS_SKIP", 4))
    tok = _tokenize(tokenizer)

    prompts = BOOTSTRAP_PROMPTS
    if os.environ.get("JLENS_NPROMPTS"):
        prompts = BOOTSTRAP_PROMPTS[:int(os.environ["JLENS_NPROMPTS"])]

    # Confirm the qwen3_5 GDN tail is actually the dispatch (not the generic one).
    disp = make_tail(ad, 0, 1).__qualname__
    print(f"tail dispatch: {disp}", flush=True)
    assert "make_qwen3_5_tail" in disp, f"expected GDN tail, got {disp}"

    # Prompt lengths (chat-templated).
    lens_tok = [len(tok(p)) for p in prompts]
    print(f"prompt token lengths: {lens_tok}", flush=True)

    if os.environ.get("JLENS_SMOKE"):
        # Time ONE VJP through the tail from the deepest requested source layer
        # (longest tail = worst case), extrapolate the fit cost. No fit, no save.
        src = min(layers)
        ids = tok(prompts[0])
        acts = capture_residuals(model, ids, [src], adapter=ad)
        valid = mx.array(valid_positions(len(ids), skip))
        tail = make_tail(ad, src + 1, target + 1)
        h = acts[src][None]
        t1 = time.perf_counter()
        # one output dim's VJP (the inner loop does D of these)
        cot = mx.zeros((1, len(ids), D)); cot = cot.at[:, :, 0].add(1.0)
        _, _g = mx.vjp(tail, [h], [cot]); mx.eval(_g)
        per_vjp = time.perf_counter() - t1
        est_layer = per_vjp * D
        print(f"smoke: 1 VJP (src={src}, tail={target-src} blocks) = {per_vjp*1e3:.0f}ms "
              f"-> ~{est_layer:.0f}s/layer/prompt (D={D} VJPs), "
              f"~{est_layer*len(layers)*len(prompts)/60:.1f}min total (rough)",
              flush=True)
        return 0

    if os.environ.get("JLENS_BENCH"):
        # Time a FULL single-layer J at several chunk sizes (dim-batching sweep) to
        # pick the chunk size + measure the real speedup vs chunk_size=1. No save.
        from jlens_mlx.providers.generic_vjp import jacobian_via_vjp
        src = min(layers)
        ids = tok(prompts[0])
        acts = capture_residuals(model, ids, [src], adapter=ad)
        valid = mx.array(valid_positions(len(ids), skip))
        h = acts[src][None]
        chunks = [int(c) for c in os.environ.get("JLENS_BENCH_CHUNKS", "1,64,128,256").split(",")]
        print(f"bench: full J_{src} (tail={target-src} blocks, D={D}) at chunk sizes {chunks}",
              flush=True)
        base = None
        for c in chunks:
            tail = make_tail(ad, src + 1, target + 1)
            t1 = time.perf_counter()
            J = jacobian_via_vjp(tail, h, valid, chunk_size=c)
            mx.eval(J)
            dt = time.perf_counter() - t1
            base = base or dt
            print(f"  chunk={c:>4}: {dt:6.1f}s/layer  ({base/dt:.1f}x vs chunk=1)  "
                  f"peak={mx.get_peak_memory()/2**30:.1f}GB", flush=True)
        return 0

    print(f"fitting layers {layers} (target={target}, skip_first={skip}) over "
          f"{len(prompts)} prompts...", flush=True)
    t1 = time.perf_counter()
    chunk = int(os.environ.get("JLENS_CHUNK", 128))
    jacobians, n_prompts = fit_lens(
        model, prompts, source_layers=layers, tokenize=tok,
        adapter=ad, target_layer=target, skip_first=skip, chunk_size=chunk)
    mx.eval(list(jacobians.values()))
    dt = time.perf_counter() - t1
    for l in sorted(jacobians):
        J = jacobians[l]
        fro = float(mx.linalg.norm(J).item())
        print(f"  J_{l}: ||J||_F={fro:.3e}  ||J||/sqrt(D)={fro/(D**0.5):.4f}  "
              f"finite={bool(mx.all(mx.isfinite(J)).item())}", flush=True)
    print(f"fit: {dt:.1f}s ({dt/n_prompts:.1f}s/prompt)  peak={mx.get_peak_memory()/2**30:.1f}GB",
          flush=True)

    # Held-out fidelity gate (never grade a lens on its fit corpus). Grade on prompts
    # NOT used for fitting: the tail of BOOTSTRAP_PROMPTS beyond the fit slice, or the
    # explicit HELD_OUT set. Include the target layer so the identity tripwire shows.
    held = BOOTSTRAP_PROMPTS[len(prompts):] or HELD_OUT
    grade_layers = sorted(set(layers) | {target})
    if target not in jacobians:
        jacobians[target] = mx.eye(D, dtype=mx.float32)  # identity lens for the tripwire
    lens_for_gate = lenslib.JSpaceLens(jacobians, grade_layers, D, softcap=ad.softcap,
                                       meta={"target_layer": target})
    print(f"fidelity gate on {len(held)} held-out prompt(s)...", flush=True)
    rep = verify.fidelity_gate(model, lens_for_gate, held, tokenize=tok, adapter=ad,
                               skip_first=skip, top_k=10, min_topk_agreement=0.0)
    for l in sorted(rep["per_layer"]):
        m = rep["per_layer"][l]
        tag = " (identity)" if l == target else ""
        print(f"  fidelity J_{l}{tag}: top1={m['top1']:.3f} top10={m['topk']:.3f} "
              f"kl={m['kl']:.3f}", flush=True)
    print(f"  identity_ok={rep['identity_ok']}  worst_layer={rep['worst_layer']}", flush=True)
    if rep["identity_ok"] is False:
        idkl = rep["per_layer"].get(target, {}).get("kl", float("nan"))
        print(f"  WARNING: identity-layer KL={idkl:.3f} exceeds the gate -- the lens does NOT "
              f"reproduce true logits; the apply path may be wrong (a correct path gives KL~0, "
              f"~0.006 even on an 8-bit model where top-1 is precision-noisy)", flush=True)
    # Drop the synthetic identity lens before saving the real fitted layers.
    if target not in layers:
        jacobians.pop(target, None)

    out = os.environ.get("JLENS_OUT") or str(ROOT / "out" / f"bootstrap-{len(layers)}L")
    sidecar = {
        "source_layers": sorted(layers),
        "d_model": D,
        "final_logit_softcapping": ad.softcap,
        "target_layer": target,
        "skip_first": skip,
        "n_prompts": n_prompts,
        "chunk_size": int(os.environ.get("JLENS_CHUNK", 128)),
        "fit_kind": "bootstrap",
        "arch": mt,
        "recipe": "hand-curated bootstrap (neutral chat + reasoning + safety-adjacent)",
        "fidelity": {str(l): rep["per_layer"][l] for l in layers if l in rep["per_layer"]},
        "fidelity_identity_ok": rep["identity_ok"],
        "note": "provisional first own-fit; production fit uses the full corpus + fidelity gate",
    }
    lenslib.save(jacobians, sidecar, out)
    print(f"saved lens -> {out}", flush=True)

    # Sanity: the saved lens loads and the deepest source layer's readout picks a
    # plausible top token at the last valid position of prompt 0.
    lens = lenslib.load(out)
    ids = tok(prompts[0])
    res = capture_residuals(model, ids, lens.source_layers, adapter=ad)
    logits = lens.apply(ad, res, positions=[-2], layers=[max(layers)])[max(layers)]
    top = int(mx.argmax(logits[0]).item())
    try:
        piece = tokenizer.decode([top])
    except Exception:
        piece = "?"
    print(f"readout sanity: J_{max(layers)} @ pos -2 top token id={top} ({piece!r})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
