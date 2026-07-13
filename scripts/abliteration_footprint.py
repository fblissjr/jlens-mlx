"""Abliteration weight footprint: where did coder3101's Heretic edit actually land?

Dequantizes the two matched 8-bit MLX builds (A = stock, B = abliterated) layer by
layer and reports the RELATIVE Frobenius weight delta ||W_B - W_A|| / ||W_A|| per
language-model layer and per projection type. Abliteration orthogonalizes the
residual-writing matrices against the refusal direction, so the edit is a small,
CONCENTRATED delta -- the per-layer SHAPE is the footprint. Also checks the vision
tower for bit-identity (abliteration is a text-refusal edit; it should not touch it).

Caveat: both builds are independently 8-bit quantized, so a flat component of the LM
delta is quant asymmetry, not abliteration. Read the SHAPE across layers (quant noise
is ~uniform; abliteration is peaked), not the absolute floor. The vision tower gives
the LM-only-ness check, not the LM quant floor (vision is stored unquantized).

Env: JLENS_MODEL_A (stock 8-bit dir, req), JLENS_MODEL_B (abliterated 8-bit dir, req).
CPU-only (no Metal needed); reads local dirs, writes nothing.
"""
from __future__ import annotations

import glob
import math
import os
import re

import mlx.core as mx

mx.set_default_device(mx.cpu)

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
    )


def main() -> int:
    da = os.environ.get("JLENS_MODEL_A")
    db = os.environ.get("JLENS_MODEL_B")
    if not (da and db):
        print("set JLENS_MODEL_A (stock 8-bit dir) and JLENS_MODEL_B (abliterated 8-bit dir)")
        return 2

    A, B = load_lazy(da), load_lazy(db)
    print(f"A(stock)={os.path.basename(da.rstrip('/'))}  B(ablit)={os.path.basename(db.rstrip('/'))}", flush=True)

    layers = sorted({
        int(m.group(1))
        for k in A
        if (m := re.match(r"language_model\.model\.layers\.(\d+)\.", k))
    })

    # --- vision tower: bit-identity check (abliteration should be LM-only) ---
    vkeys = [k for k in A if k.startswith("vision_tower.") and k.endswith(".weight")]
    vmax = 0.0
    for k in vkeys[:60]:  # sample -- enough to catch any nonzero
        if k in B:
            d = float(mx.max(mx.abs(B[k].astype(mx.float32) - A[k].astype(mx.float32))))
            vmax = max(vmax, d)
    print(f"\nvision_tower: max |B-A| over {min(len(vkeys),60)} weight tensors = {vmax:.3e}  "
          f"({'BIT-IDENTICAL -> abliteration is LM-only' if vmax == 0.0 else 'DIFFERS'})", flush=True)

    # --- the READOUT (outside J): is it shared A/B? ---
    # If lm_head / final norm are untouched, the diff is a PURE transport difference and
    # the identity tripwire holds identically for both lenses. embed_tokens is the input side.
    print("\nreadout / input tensors (relative delta):", flush=True)
    for base, quant in [
        ("language_model.lm_head", True),
        ("language_model.model.embed_tokens", True),
        ("language_model.model.norm", False),
    ]:
        if quant and base + ".scales" in A:
            wa, wb = dequant(A, base), dequant(B, base)
        elif base + ".weight" in A:
            wa, wb = A[base + ".weight"].astype(mx.float32), B[base + ".weight"].astype(mx.float32)
        else:
            print(f"  {base:<34s} (absent)", flush=True)
            continue
        d2 = float(mx.sum((wb - wa) ** 2))
        w2 = float(mx.sum(wa * wa))
        print(f"  {base:<34s} {math.sqrt(d2/(w2+1e-12)):.5f}", flush=True)

    # --- per-layer LM weight delta ---
    print("\nper-layer relative weight delta ||B-A|| / ||A||  (dequantized 8-bit):", flush=True)
    rows = []
    proj_accum: dict[str, list[float]] = {}
    for L in layers:
        pfx = f"language_model.model.layers.{L}."
        bases = sorted({
            k[len(pfx):].rsplit(".", 1)[0]
            for k in A if k.startswith(pfx) and k.endswith(".scales")
        })
        td2 = tw2 = 0.0
        per = {}
        for b in bases:
            wa, wb = dequant(A, pfx + b), dequant(B, pfx + b)
            d = wb - wa
            d2 = float(mx.sum(d * d))
            w2 = float(mx.sum(wa * wa))
            td2 += d2
            tw2 += w2
            rel = math.sqrt(d2 / (w2 + 1e-12))
            per[b] = rel
            proj_accum.setdefault(b, []).append(rel)
        rel_layer = math.sqrt(td2 / (tw2 + 1e-12))
        rows.append((L, rel_layer, per))

    mxrel = max(r[1] for r in rows)
    for L, rel, per in rows:
        bar = "#" * int(round(rel / mxrel * 40))
        print(f"  L{L:<2d} {rel:.5f} {bar}", flush=True)

    # --- ranked footprint ---
    print("\ntop-12 layers by relative delta (the abliteration footprint):", flush=True)
    for L, rel, per in sorted(rows, key=lambda r: -r[1])[:12]:
        hot = " ".join(f"{k}={v:.4f}" for k, v in sorted(per.items(), key=lambda kv: -kv[1])[:3])
        print(f"  L{L:<2d} rel={rel:.5f}   {hot}", flush=True)

    # --- per projection-type mean (where does abliteration concentrate?) ---
    print("\nmean relative delta by projection type:", flush=True)
    for b, vals in sorted(proj_accum.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {b:<28s} {sum(vals)/len(vals):.5f}  (n={len(vals)})", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
