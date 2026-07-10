"""MLX side of the fitter cross-check: fit gpt2 on the SAME ids and compare J vs torch.

Reads scripts/_xcheck_torch.npz (produced by xcheck_fit_torch.py) and fits our direct-VJP
fitter on the exact same token ids, then compares J per layer. gpt2 mlx == HF numerically
(V1 apply parity cos 1.0), so a correct fitter should match to ~fp precision.

Run in the heylook venv:  uv run python scripts/xcheck_fit_mlx.py
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

from jlens_mlx.capture import ModelAdapter  # noqa: E402
from jlens_mlx.fit import fit_prompt  # noqa: E402

NPZ = Path(__file__).resolve().parent / "_xcheck_torch.npz"


def cos(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    if not NPZ.exists():
        print(f"missing {NPZ}; run xcheck_fit_torch.py first", file=sys.stderr)
        return 2
    d = np.load(NPZ)
    src = [int(x) for x in d["src"]]
    skip_first = int(d["skip_first"])
    target = int(d["target"])
    n_prompts = sum(1 for k in d.files if k.startswith("ids_"))

    model_id = str(d["model"]) if "model" in d.files else "openai-community/gpt2"
    model, _ = load(model_id)
    model.update(tree_map(lambda p: p.astype(mx.float32) if isinstance(p, mx.array) else p,
                          model.parameters()))
    mx.eval(model.parameters())
    ad = ModelAdapter(model)

    acc = None
    for i in range(n_prompts):
        ids = d[f"ids_{i}"].tolist()
        per, _ = fit_prompt(model, ids, src, adapter=ad, target_layer=target, skip_first=skip_first)
        acc = per if acc is None else {l: acc[l] + per[l] for l in src}
        mx.eval(list(acc.values()))
    jmlx = {l: acc[l] / n_prompts for l in src}

    print(f"cross-check gpt2 fit: layers {src} target={target} skip_first={skip_first} "
          f"n_prompts={n_prompts}")
    worst = 1.0
    for l in src:
        jt = d[f"J_{l}"]
        jm = np.asarray(jmlx[l])
        c = cos(jt, jm)
        ae = float(np.abs(jt - jm).max())
        rel = float(np.linalg.norm(jt - jm) / (np.linalg.norm(jt) + 1e-12))
        worst = min(worst, c)
        print(f"  J_{l}: cos={c:.6f}  max_abs_err={ae:.2e}  rel_frob={rel:.2e}")
    ok = worst > 0.999
    print(f"\nFITTER CROSS-CHECK {'PASS' if ok else 'FAIL'}  (torch jlens vs MLX, worst cos={worst:.6f})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
