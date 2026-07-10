"""Baseline fitter verification on gpt2 (direct end-to-end mx.vjp, Anthropic design).

Fits two late source layers on short prompts and verifies the fitter + apply + capture
path end-to-end:
  [1] J_target (l == target) == I  — exercises the fit plumbing exactly (empty tail).
  [2] apply(J_target, acts[target]) == the model's real logits (cos ~1) — the whole
      capture -> fit -> transport -> real-norm-unembed path.
  [3] a non-trivial fitted layer produces a finite, sensible readout + ||J||/sqrt(d).

The oracle lens is NOT a gate here (it is corpus-dependent). The rigorous cross-check
(torch jlens vs this MLX fit on the same corpus) is a future V-gate.

Run under an env with mlx-lm:  uv run python scripts/fit_gpt2_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map
from mlx_lm import load

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jlens_mlx import lens as lenslib  # noqa: E402
from jlens_mlx.capture import ModelAdapter, capture_residuals  # noqa: E402
from jlens_mlx.fit import fit_lens, valid_positions  # noqa: E402


def cos(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    model, tok = load("openai-community/gpt2")
    model.update(tree_map(lambda p: p.astype(mx.float32) if isinstance(p, mx.array) else p,
                          model.parameters()))
    mx.eval(model.parameters())
    ad = ModelAdapter(model)
    n = ad.n_layers
    target = n - 1
    tokenize = lambda s: list(tok.encode(s))  # noqa: E731
    prompts = ["The Eiffel Tower is in the city of", "The capital of Japan is"]

    src = [target - 1, target]  # a non-trivial layer + the identity target
    print(f"fitting layers {src} (target={target}) on {len(prompts)} prompts ...")
    J, nprompts = fit_lens(model, prompts, source_layers=src, tokenize=tokenize, skip_first=1)
    D = J[target].shape[0]

    # [1] identity: J_target == I
    ident_err = float(mx.max(mx.abs(J[target] - mx.eye(D))).item())
    print(f"[1] J_{target} == I         : max|J-I| = {ident_err:.2e}")

    # [2] apply(J_target) vs the model's real logits (cos ~1 over valid positions)
    ids = tokenize(prompts[0])
    acts_t = capture_residuals(model, ids, [target], adapter=ad)
    jl_t = lenslib.JSpaceLens({target: J[target]}, [target], D, softcap=ad.softcap)
    valid = valid_positions(len(ids), 1)
    applied = jl_t.apply(ad, acts_t, positions=valid, layers=[target])[target]  # [len(valid), vocab]
    real = ad.logits(mx.array([ids]))[0]                                        # [S, vocab]
    parity = min(cos(np.asarray(applied)[i], np.asarray(real)[valid[i]]) for i in range(len(valid)))
    print(f"[2] apply(J_{target}) vs real logits: min cos over valid pos = {parity:.5f}")

    # [3] non-trivial layer: norm + finite + top-5 readout at the last content position
    l = target - 1
    jnorm = float(mx.sqrt(mx.sum(J[l] * J[l])).item()) / (D ** 0.5)
    jl_l = lenslib.JSpaceLens({l: J[l]}, [l], D, softcap=ad.softcap)
    acts_l = capture_residuals(model, ids, [l], adapter=ad)
    logit = jl_l.apply(ad, acts_l, positions=[valid[-1]], layers=[l])[l][0]     # [vocab]
    finite = bool(mx.all(mx.isfinite(logit)).item())
    top = np.asarray(mx.argpartition(-logit, 5)[:5])
    toks = [repr(tok.decode([int(t)])) for t in top]
    print(f"[3] J_{l}: ||J||/sqrt(d)={jnorm:.3f} finite={finite} top5@last={toks}")

    ok = ident_err < 1e-3 and parity > 0.999 and finite
    print(f"\nBASELINE FIT {'PASS' if ok else 'FAIL'} ({nprompts} prompts, direct end-to-end VJP)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
