"""Fit/apply capture-parity gate: cache-less vs fresh-cache residual capture.

The fit side (jlens_mlx.capture.capture_residuals) runs the inner forward
CACHE-LESS; the apply side (the server's jspace capture twin) runs it with a
FRESH per-layer cache (its served hybrid path requires one). Both are
causal-from-scratch and SHOULD produce identical block-output residuals -- but
until this gate, that equivalence was asserted in comments, never measured.
This is the foundation of served-model lens correctness: the lens is fit on
side-A residuals and applied to side-B residuals.

Run it BEFORE any corpus fit (it is the opening step of a refit session).

Requires the server package importable (run under the server repo's venv) and:
  JLENS_MODEL   -- model directory (absolute path)
  JLENS_PARITY_LAYERS (optional) -- comma list; default: sentinels + band edges

Exit 0 = parity holds (max rel err < 1e-5 on every checked layer).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

REL_TOL = 1e-5


def rel_err(a: mx.array, b: mx.array) -> float:
    num = mx.max(mx.abs(a - b)).item()
    den = max(mx.max(mx.abs(b)).item(), 1e-12)
    return num / den


def main() -> int:
    model_path = os.environ.get("JLENS_MODEL")
    if not model_path:
        print("set JLENS_MODEL to the served model dir (absolute path)")
        return 2
    model_path = os.path.abspath(model_path)

    from mlx_lm import load

    from jlens_mlx.capture import ModelAdapter as FitAdapter
    from jlens_mlx.capture import capture_residuals as fit_capture
    try:
        from heylook_llm.jspace.capture import ModelAdapter as ApplyAdapter
        from heylook_llm.jspace.capture import capture_residuals as apply_capture
    except ImportError:
        print("server jspace package not importable -- run under the server repo's venv")
        return 2

    model, tokenizer = load(model_path)

    prompt = "The Eiffel Tower is in the city of"
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=True, enable_thinking=False)

    fit_ad = FitAdapter(model)
    apply_ad = ApplyAdapter(model)
    n = fit_ad.n_layers
    env_layers = os.environ.get("JLENS_PARITY_LAYERS")
    if env_layers:
        layers = sorted({int(x) for x in env_layers.split(",")})
    else:
        # band edges + early/late sentinels, clipped to depth
        layers = sorted({0, 1, 8, 16, 24, 31, 40, 47, n - 2} & set(range(n))
                        | {l for l in (16, 47) if l < n})

    print(f"model={os.path.basename(model_path)} n_layers={n} "
          f"seq_len={len(ids)} layers={layers}", flush=True)

    fit_res = fit_capture(model, ids, layers, adapter=fit_ad)
    apply_res = apply_capture(model, ids, layers, adapter=apply_ad)

    worst = 0.0
    failed = []
    for l in layers:
        r = rel_err(fit_res[l], apply_res[l])
        worst = max(worst, r)
        status = "OK " if r < REL_TOL else "FAIL"
        if r >= REL_TOL:
            failed.append(l)
        print(f"  L{l:>3}  rel_err={r:.3e}  {status}", flush=True)

    if failed:
        print(f"CAPTURE PARITY FAIL: layers {failed} exceed rel {REL_TOL:g} "
              f"(worst {worst:.3e}). The fit-side and apply-side forwards "
              f"DIVERGE -- do not fit until understood.")
        return 1
    print(f"CAPTURE PARITY OK: {len(layers)} layers, worst rel_err {worst:.3e} "
          f"< {REL_TOL:g} (cache-less fit capture == fresh-cache apply capture)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
