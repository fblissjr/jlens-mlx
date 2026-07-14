"""Is the abliteration EDIT DIRECTION refusal-token-shaped, per layer?

Motivation: the deep-band diff (L48-59) surfaces refusal vocab at L48-54 (o_proj/attention tent)
then FADES to structural noise at L55-59 (down_proj/MLP tent) -- despite those layers being the MOST
legible. Open question: is the deep edit's DIRECTION not refusal-shaped, or does a refusal-shaped
deep direction just fail to propagate to the readout?

This answers it in WEIGHT SPACE, no Metal/generation. Heretic's per-layer edit is low-rank
(`W_B = W_A - weight * v v^T W_A`, norm-preserving SVD to rank 3), so the dominant LEFT singular
vector of the weight delta `W_B - W_A` IS the residual-space direction the edit moved (== the
per-layer refusal direction r_l, up to sign). We recover it by power iteration, then logit-lens
decode it (RMSNorm + lm_head) to top tokens. Per layer we pick the most-edited residual-writing
matrix (d_out == d_model): o_proj/out_proj where attention is edited (shallow), an expert down_proj
where MLP is edited (deep).

  shallow (o_proj tent ~L46) should decode to refusal tokens; the test is whether DEEP (down_proj
  tent ~L58) does too. If deep decodes to structural/non-refusal tokens, the deep direction itself
  isn't refusal-shaped (explains the readout-invisibility); if it decodes to refusal, the edit is
  refusal-shaped but doesn't propagate (a mechanism/propagation story instead).

Env: JLENS_MODEL_A (stock 8-bit dir), JLENS_MODEL_B (abliterated 8-bit dir),
JLENS_LAYERS (comma list; default a shallow+deep contrast), JLENS_TOPK (12), JLENS_TOK (tokenizer
dir; default = MODEL_A). CPU-only linear algebra; no GPU generation.
"""
from __future__ import annotations

import glob
import os

import mlx.core as mx
from transformers import AutoTokenizer

GROUP_SIZE, BITS = 64, 8


def load_lazy(d: str) -> dict:
    w: dict = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        w.update(mx.load(f))
    return w


def dequant(w: dict, base: str) -> mx.array:
    return mx.dequantize(
        w[base + ".weight"], w[base + ".scales"], w[base + ".biases"],
        group_size=GROUP_SIZE, bits=BITS,
    ).astype(mx.float32)


def top_left_singular(delta: mx.array, iters: int = 40) -> mx.array:
    """Top left singular vector of `delta` [d_out, d_in] via deterministic power iteration on
    delta @ delta^T. The edit is ~rank-3 with a dominant top component, so this converges fast."""
    u = mx.ones((delta.shape[0],), dtype=mx.float32)
    u = u / mx.linalg.norm(u)
    for _ in range(iters):
        u = delta @ (delta.T @ u)
        u = u / (mx.linalg.norm(u) + 1e-12)
        mx.eval(u)
    return u


def logit_lens(u: mx.array, lm_head: mx.array, norm_w: mx.array, tok, k: int) -> tuple[list, list]:
    """Decode a residual-space direction to tokens: RMSNorm(u) * norm_w, then lm_head. Sign of a
    singular vector is arbitrary, so return BOTH poles (+u surfaced-more / -u surfaced-less)."""
    x = u / mx.sqrt(mx.mean(u * u) + 1e-6) * norm_w
    logits = lm_head @ x  # [vocab]
    order = mx.argsort(logits)
    top = [tok.decode([int(i)]) for i in order[-k:].tolist()][::-1]
    bot = [tok.decode([int(i)]) for i in order[:k].tolist()]
    return top, bot


def main() -> int:
    da, db = os.environ.get("JLENS_MODEL_A"), os.environ.get("JLENS_MODEL_B")
    if not (da and db):
        print("set JLENS_MODEL_A (stock 8-bit) and JLENS_MODEL_B (abliterated 8-bit)")
        return 2
    k = int(os.environ.get("JLENS_TOPK", 12))
    tok = AutoTokenizer.from_pretrained(os.environ.get("JLENS_TOK", da), trust_remote_code=True)

    A, B = load_lazy(da), load_lazy(db)
    lm_head = dequant(A, "language_model.lm_head")           # [vocab, d_model], readout shared A/B
    d_model = lm_head.shape[1]
    norm_w = A["language_model.model.norm.weight"].astype(mx.float32)
    print(f"A={os.path.basename(da.rstrip('/'))}  B={os.path.basename(db.rstrip('/'))}  "
          f"d_model={d_model} vocab={lm_head.shape[0]}\n", flush=True)

    layers = ([int(x) for x in os.environ["JLENS_LAYERS"].split(",") if x.strip()]
              if os.environ.get("JLENS_LAYERS") else [44, 46, 48, 51, 54, 55, 57, 58])

    for L in layers:
        pfx = f"language_model.model.layers.{L}."
        # residual-writing, quantized bases only (have .scales) with d_out == d_model.
        bases = sorted({k2[len(pfx):].rsplit(".", 1)[0]
                        for k2 in A if k2.startswith(pfx) and k2.endswith(".scales")})
        best_base, best_rel, best_delta = None, -1.0, None
        for b in bases:
            wa = dequant(A, pfx + b)
            if wa.shape[0] != d_model:      # skip input matrices (q/k/v/gate/up): wrong output dim
                continue
            wb = dequant(B, pfx + b)
            delta = wb - wa
            rel = float(mx.linalg.norm(delta) / (mx.linalg.norm(wa) + 1e-12))
            if rel > best_rel:
                best_base, best_rel, best_delta = b, rel, delta
        if best_delta is None:
            print(f"L{L}: no residual-writing matrix found", flush=True)
            continue
        u = top_left_singular(best_delta)
        top, bot = logit_lens(u, lm_head, norm_w, tok, k)
        print(f"L{L}  edited={best_base}  rel_delta={best_rel:.4f}", flush=True)
        print(f"   +dir: {' '.join(repr(t) for t in top)}", flush=True)
        print(f"   -dir: {' '.join(repr(t) for t in bot)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
