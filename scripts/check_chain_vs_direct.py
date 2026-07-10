"""THE GATE for the UNVERIFIED chain speedup (jlens_mlx/chain.py). Run this on a FREE GPU.

Fits the same layers via the trusted direct VJP and via the one-sweep chain, and checks they
match. Two arches, because the chain's #1 risk is the GDN custom_function under a per-block vjp:
  - tiny synthetic qwen3_5 (GDN hybrid path) -- the discriminating case,
  - gpt2 (LayerNorm, download) -- the plain-arch baseline, if TORCH/net available (optional; set
    CHAIN_GPT2=1 to include it).

PASS = cos > 0.99999 and rel_err < 1e-4 for every layer (same estimator, so any gap is a bug --
see chain.py's "WHAT MIGHT BE WRONG"). On PASS, a later Claude may wire fit_lens/fit_corpus to
fit_prompt_chain for multi-layer fits and delete chain.py's UNVERIFIED banner.

Run (heylook venv, from the heylook dir):  uv run python <jlens-mlx>/scripts/check_chain_vs_direct.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from jlens_mlx.capture import ModelAdapter  # noqa: E402
from jlens_mlx.chain import verify_chain_matches_direct  # noqa: E402

FAIL = []


def _report(name, rep):
    ok = rep.pop("pass")
    print(f"[{name}] {'PASS' if ok else 'FAIL'}")
    for l, m in sorted((k, v) for k, v in rep.items() if isinstance(k, int)):
        print(f"    J_{l}: cos={m['cos']:.6f} rel_err={m['rel_err']:.2e} max_abs={m['max_abs_err']:.2e}")
    if not ok:
        FAIL.append(name)


def main() -> int:
    if not mx.metal.is_available():
        print("no Metal -- run on Apple silicon"); return 1

    # (a) tiny synthetic qwen3_5 -- exercises the GDN per-block vjp (the #1 risk).
    from check_qwen3_5_synthetic import tiny_qwen3_5
    mx.random.seed(0)
    m = tiny_qwen3_5()
    ad = ModelAdapter(m)
    ids = list(range(2, 24))
    rep = verify_chain_matches_direct(m, ids, [2, 5, ad.n_layers - 1], adapter=ad,
                                      target_layer=ad.n_layers - 1, skip_first=2)
    _report("qwen3_5-synth (GDN)", rep)

    # (b) gpt2 -- plain LayerNorm arch (optional; needs a download).
    if os.environ.get("CHAIN_GPT2"):
        from mlx_lm import load
        gm, _ = load("gpt2")
        gad = ModelAdapter(gm)
        rep2 = verify_chain_matches_direct(gm, list(range(2, 24)), [3, 6, gad.n_layers - 1],
                                           adapter=gad, target_layer=gad.n_layers - 1, skip_first=2)
        _report("gpt2 (LayerNorm)", rep2)

    print(f"\nCHAIN GATE {'PASS' if not FAIL else 'FAIL: ' + ', '.join(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
