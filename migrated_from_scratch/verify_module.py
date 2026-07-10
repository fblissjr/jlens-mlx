# End-to-end check: the REAL heylook_llm.jspace module vs the torch oracle
# fixtures (V1 gpt2 + V2 gemma-2-2b). Proves the committed module -- not the
# ad-hoc spike scripts -- reproduces cos~1.0 parity.
#   uv run python coderef/jspace_scratch/verify_module.py
import os, glob
import numpy as np
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from heylook_llm.jspace import JSpaceLens, ModelAdapter, capture_residuals

OUT = os.path.dirname(os.path.abspath(__file__))

def cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def top5(v): return set(np.asarray(v).argsort()[-5:][::-1].tolist())

CASES = [("gpt2", "openai-community/gpt2"), ("gemma22b", "google/gemma-2-2b")]
overall = True
for prefix, hf in CASES:
    lens = JSpaceLens.from_files(os.path.join(OUT, f"lens_{prefix}.safetensors"),
                                 os.path.join(OUT, f"lens_{prefix}.sidecar.json"))
    model, _ = load(hf)
    model.update(tree_map(lambda p: p.astype(mx.float32) if isinstance(p, mx.array) else p,
                          model.parameters()))
    mx.eval(model.parameters())
    ad = ModelAdapter(model)
    worst = 1.0; min_ov = 5
    for npz in sorted(glob.glob(os.path.join(OUT, f"oracle_{prefix}_*.npz"))):
        d = np.load(npz); ids = d["input_ids"].tolist()
        res = capture_residuals(model, ids, lens.source_layers, adapter=ad)
        lens_logits = lens.apply(ad, res, positions=[-1])
        for l in lens.source_layers:
            v = np.asarray(lens_logits[l][0])
            lc = cos(v, d[f"lens_last_{l}"])
            ov = len(top5(v) & set(d[f"lens_topk_{l}"][:5].tolist()))
            worst = min(worst, lc); min_ov = min(min_ov, ov)
    ok = worst > 0.99 and min_ov >= 4
    overall &= ok
    print(f"[{prefix:9}] softcap={ad.softcap} layers={len(lens.source_layers)} "
          f"worst_cos={worst:.5f} min_top5={min_ov}/5 -> {'PASS' if ok else 'FAIL'}")
print("MODULE E2E", "PASS" if overall else "FAIL")
