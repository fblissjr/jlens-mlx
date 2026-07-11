"""Qualitative readout: apply a fitted lens to a prompt and print the top 'silent' tokens per
source layer at the answer-onset. The fidelity gate (agreement with FINAL logits) can't judge a
Jacobian lens at early layers -- the lens is meant to DIVERGE from the output there. This eyeballs
whether the band readout is meaningful disposition (sensible tokens) or noise.

JLENS_MODEL + JLENS_LENS (a lens dir with lens.safetensors + sidecar). JLENS_PROMPT optional.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jlens_mlx import lens as lenslib  # noqa: E402
from jlens_mlx.capture import ModelAdapter, capture_residuals  # noqa: E402


def main() -> int:
    model, tokenizer = load(os.environ["JLENS_MODEL"])
    ad = ModelAdapter(model)
    lens = lenslib.load(os.environ["JLENS_LENS"])
    prompt = os.environ.get("JLENS_PROMPT", "The capital of France is")
    ids = list(tokenizer.encode(prompt))
    res = capture_residuals(model, ids, lens.source_layers, adapter=ad)
    # Read at the last few positions (answer onset region).
    for pos in (-1,):
        out = lens.apply(ad, res, positions=[pos], layers=lens.source_layers)
        print(f"\nprompt={prompt!r}  pos={pos}")
        for l in lens.source_layers:
            v = out[l][0]
            top = mx.argsort(-v)[:8]
            toks = [tokenizer.decode([int(t)]) for t in top]
            print(f"  L{l}: " + " | ".join(repr(t) for t in toks))
    # For contrast, the TRUE next-token top-8 (what the identity layer would read).
    true = ad.logits(mx.array([ids]))[0][-1]
    tt = [tokenizer.decode([int(t)]) for t in mx.argsort(-true)[:8]]
    print(f"\n  TRUE next-token top-8: " + " | ".join(repr(t) for t in tt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
