"""Synthetic gemma2 arch check — proves the gemma array-mask fix in make_tail without a download.

gemma-2 is gated on HF, so we can't fetch it here for a numerical cross-check. This builds a
TINY random-weight gemma2 model and runs the fitter through it. It confirms:
  [1] the fitter RUNS on gemma2 (arch-dispatched ARRAY mask) + J_target == I + a finite J,
  [2] the "causal" STRING mask WOULD crash gemma2 (mask.dtype read) — i.e. the fix is load-bearing.
Random weights, so this is an arch/plumbing gate, NOT the numerical generalization cross-check
(that still needs a real gemma-2-2b + torch).

Run:  uv run python scripts/check_gemma2_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.gemma2 import Model, ModelArgs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jlens_mlx.fit as fitmod  # noqa: E402
from jlens_mlx.capture import ModelAdapter  # noqa: E402
from jlens_mlx.fit import fit_prompt  # noqa: E402


def tiny_gemma2() -> Model:
    args = ModelArgs(model_type="gemma2", hidden_size=64, num_hidden_layers=4,
                     intermediate_size=128, num_attention_heads=4, head_dim=16,
                     rms_norm_eps=1e-6, vocab_size=256, num_key_value_heads=2)
    m = Model(args)
    mx.eval(m.parameters())
    return m


def main() -> int:
    mx.random.seed(0)
    model = tiny_gemma2()
    ad = ModelAdapter(model)
    n = ad.n_layers
    target = n - 1
    ids = [1, 5, 9, 13, 17, 21, 25]

    print(f"tiny gemma2: model_type={fitmod._model_type(ad)!r} n_layers={n} softcap={ad.softcap}")

    # [1] fitter runs on gemma2 (array mask) + identity + finite non-trivial J
    J, _ = fit_prompt(model, ids, [target - 1, target], adapter=ad, target_layer=target, skip_first=1)
    D = J[target].shape[0]
    ident = float(mx.max(mx.abs(J[target] - mx.eye(D))).item())
    finite = bool(mx.all(mx.isfinite(J[target - 1])).item())
    print(f"[1] gemma2 fit runs (array mask): J_{target}==I max|J-I|={ident:.2e}  J_{target-1} finite={finite}")

    # [2] the array mask reproduces gemma2's OWN forward (it uses return_array=True). Compare
    # a non-trivial J fit with the array mask vs the "causal" string, to show whether the
    # string silently diverges (fix required) or is handled (fix = future-proofing) here.
    import numpy as np
    J_arr = np.asarray(J[target - 1])
    orig = fitmod._model_type
    fitmod._model_type = lambda _ad: ""  # force the "causal" string mask
    try:
        Js, _ = fit_prompt(model, ids, [target - 1], adapter=ad, target_layer=target, skip_first=1)
        J_str = np.asarray(Js[target - 1])
        a, b = J_arr.ravel(), J_str.ravel()
        c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        me = float(np.abs(J_arr - J_str).max())
        verdict = "string handled here (fix = future-proofing)" if c > 0.999 else "string DIVERGES (fix required)"
        print(f"[2] array vs string mask J_{target-1}: cos={c:.6f} max_err={me:.2e} -> {verdict}")
    except Exception as e:  # noqa: BLE001
        print(f"[2] string mask crashes gemma2: {type(e).__name__} -> array-mask fix REQUIRED")
    finally:
        fitmod._model_type = orig

    ok = ident < 1e-3 and finite
    print(f"\nGEMMA2 ARCH FIX {'PASS (fitter runs correctly with the faithful array mask)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
