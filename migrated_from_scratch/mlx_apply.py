# V1: MLX apply + residual capture, compared against the V0 torch oracle.
# Runs in the PROJECT venv (mlx-lm). Reuses oracle input_ids (no tokenizer drift).
#   uv run python coderef/jspace_scratch/mlx_apply.py
import os, json, glob
import numpy as np
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

OUT = os.path.dirname(os.path.abspath(__file__))
side = json.load(open(os.path.join(OUT, "lens_gpt2.sidecar.json")))
SRC = side["source_layers"]
J = mx.load(os.path.join(OUT, "lens_gpt2.safetensors"))   # {str(l): [d,d] fp32}

model, _ = load("openai-community/gpt2")
# Force fp32 everywhere for a clean port-correctness check (isolate from dtype noise).
model.update(tree_map(lambda p: p.astype(mx.float32) if isinstance(p, mx.array) else p,
                      model.parameters()))
mx.eval(model.parameters())
gm = model.model   # GPT2Model

def forward_capture(ids):
    inputs = mx.array([ids])
    L = inputs.shape[1]
    h = gm.wte(inputs) + gm.wpe(mx.arange(L))
    mask = create_attention_mask(h, None)
    resids = {}
    for i, layer in enumerate(gm.h):
        h = layer(h, mask, cache=None)
        resids[i] = h[0]                       # [L, d], drop batch
    return resids                              # includes final block (n_layers-1)

def unembed(x):                                # gpt2: no softcap
    return gm.wte.as_linear(gm.ln_f(x))

def cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def top5(v): return set(np.asarray(v).argsort()[-5:][::-1].tolist())

print(f"{'prompt':10} {'layer':>5} {'resid_cos':>10} {'lens_cos':>9} {'top5_overlap':>12}")
worst_resid = worst_lens = 1.0; min_overlap = 5
for npz in sorted(glob.glob(os.path.join(OUT, "oracle_*.npz"))):
    slug = os.path.basename(npz)[len("oracle_"):-len(".npz")]
    d = np.load(npz)
    ids = d["input_ids"].tolist()
    last = len(ids) - 1
    resids = forward_capture(ids)              # keys 0..n_layers-1
    final = resids[len(gm.h) - 1]              # last block output -> model logits
    mlx_model_last = np.asarray(unembed(final)[last])
    mo = cos(mlx_model_last, d["model_last"]); mov = len(top5(mlx_model_last) & top5(d["model_last"]))
    print(f"{slug:10} {'MODEL':>5} {'':>10} {mo:9.5f} {mov:>10}/5")
    for l in SRC:
        rc = cos(np.asarray(resids[l]), d[f"resid_{l}"].astype(np.float32))
        transported = resids[l].astype(mx.float32) @ J[str(l)].T
        lens_last = np.asarray(unembed(transported)[last])
        lc = cos(lens_last, d[f"lens_last_{l}"])
        ov = len(top5(lens_last) & set(d[f"lens_topk_{l}"][:5].tolist()))
        worst_resid = min(worst_resid, rc); worst_lens = min(worst_lens, lc)
        min_overlap = min(min_overlap, ov)
        flag = "" if (lc > 0.99 and ov >= 4) else "  <-- MISS"
        print(f"{slug:10} {l:>5} {rc:10.5f} {lc:9.5f} {ov:>10}/5{flag}")

print(f"\nWORST resid_cos={worst_resid:.5f} lens_cos={worst_lens:.5f} min_top5_overlap={min_overlap}/5")
print("V1", "PASS" if (worst_lens > 0.99 and min_overlap >= 4) else "FAIL")
