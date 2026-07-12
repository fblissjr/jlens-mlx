"""gpt2 apply-parity gate (V1) for jlens-mlx.

Proves the ported apply path (jlens_mlx.capture + jlens_mlx.lens) reproduces the
genuine jlens.apply() oracle on gpt2-small, re-establishing the heylook V1 gate in
this repo. This is the foundation slice: the fit driver (next) is verified by
checking that a freshly-fit lens' apply matches the same oracle.

Run under an env that has mlx-lm (e.g. the heylook venv):

    uv run python scripts/check_gpt2_parity.py

Gate: worst lens cosine > 0.99 AND min top-5 overlap >= 4/5 across all oracle
prompts x all fitted source layers.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map
from mlx_lm import load

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jlens_mlx.capture import ModelAdapter, capture_residuals  # noqa: E402
from jlens_mlx.lens import JSpaceLens  # noqa: E402

FIX = ROOT / "tests" / "golden"


def cos(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def top5(v) -> set:
    return set(np.asarray(v).argsort()[-5:][::-1].tolist())


def main() -> int:
    # This is the gpt2 gate. The gemma22b oracles (V2, vocab 256000) are for the
    # gemma lens and belong to a separate check -- skip them here, loudly.
    all_oracles = sorted(glob.glob(str(FIX / "oracle_*.npz")))
    oracles = [o for o in all_oracles if "gemma" not in Path(o).name]
    skipped = [Path(o).name for o in all_oracles if "gemma" in Path(o).name]
    if skipped:
        print(f"(skipping non-gpt2 oracles: {', '.join(skipped)})")
    if not oracles:
        print(f"no gpt2 oracle_*.npz under {FIX} -- nothing to check", file=sys.stderr)
        return 2

    model, _ = load("openai-community/gpt2")
    # Force fp32 to isolate port-correctness from dtype noise (matches the V0 oracle).
    model.update(tree_map(lambda p: p.astype(mx.float32) if isinstance(p, mx.array) else p,
                          model.parameters()))
    mx.eval(model.parameters())

    ad = ModelAdapter(model)
    lens = JSpaceLens.from_files(FIX / "lens_gpt2.safetensors", FIX / "lens_gpt2.sidecar.json")
    src = lens.source_layers
    print(f"adapter: n_layers={ad.n_layers} softcap={ad.softcap} | lens: {lens}")
    print(f"{'prompt':12} {'layer':>5} {'lens_cos':>9} {'top5':>6}")

    worst, min_ov = 1.0, 5
    for npz in oracles:
        slug = Path(npz).name[len("oracle_"):-len(".npz")]
        d = np.load(npz)
        ids = d["input_ids"].tolist()
        resids = capture_residuals(model, ids, src, adapter=ad)
        out = lens.apply(ad, resids, positions=[-1], layers=src)  # {l: [1, vocab]}
        for l in src:
            lens_last = np.asarray(out[l][0])
            lc = cos(lens_last, d[f"lens_last_{l}"])
            ov = len(top5(lens_last) & set(d[f"lens_topk_{l}"][:5].tolist()))
            worst, min_ov = min(worst, lc), min(min_ov, ov)
            flag = "" if (lc > 0.99 and ov >= 4) else "  <-- MISS"
            print(f"{slug:12} {l:>5} {lc:9.5f} {ov:>4}/5{flag}")

    ok = worst > 0.99 and min_ov >= 4
    print(f"\nWORST lens_cos={worst:.5f} min_top5_overlap={min_ov}/5")
    print("V1", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
