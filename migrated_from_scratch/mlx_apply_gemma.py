# V2: MLX apply for gemma (RMSNorm + sqrt(d) embed + final_logit_softcapping)
# vs the torch oracle. Project venv. Reuses oracle input_ids.
#   uv run python coderef/jspace_scratch/mlx_apply_gemma.py
import os, json, glob
import numpy as np
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

OUT = os.path.dirname(os.path.abspath(__file__))
PREFIX = "gemma22b"
side = json.load(open(os.path.join(OUT, f"lens_{PREFIX}.sidecar.json")))
SRC = side["source_layers"]
CAP = side["final_logit_softcapping"]      # 30.0
HF = side["hf_model_name"]
J = mx.load(os.path.join(OUT, f"lens_{PREFIX}.safetensors"))

model, _ = load(HF)
model.update(tree_map(lambda p: p.astype(mx.float32) if isinstance(p, mx.array) else p,
                      model.parameters()))
mx.eval(model.parameters())
gm = model.model                            # GemmaModel

def forward_capture(ids):
    inputs = mx.array([ids])
    h = gm.embed_tokens(inputs) * (gm.args.hidden_size ** 0.5)     # gemma sqrt(d) scaling
    mask = create_attention_mask(h, None, return_array=True)
    resids = {}
    for i, layer in enumerate(gm.layers):
        h = layer(h, mask, None)
        resids[i] = h[0]
    return resids

def unembed(x):
    logits = gm.embed_tokens.as_linear(gm.norm(x))                 # RMSNorm + tied head
    return mx.tanh(logits / CAP) * CAP if CAP else logits          # final_logit_softcapping

def cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def top5(v): return set(np.asarray(v).argsort()[-5:][::-1].tolist())

print(f"model={HF} softcap={CAP} src_layers={SRC[0]}..{SRC[-1]} ({len(SRC)})")
print(f"{'prompt':10} {'layer':>5} {'resid_cos':>10} {'lens_cos':>9} {'top5':>6}")
worst_resid = worst_lens = 1.0; min_overlap = 5
for npz in sorted(glob.glob(os.path.join(OUT, f"oracle_{PREFIX}_*.npz"))):
    slug = os.path.basename(npz).split("_")[-1][:-4]
    d = np.load(npz); ids = d["input_ids"].tolist(); last = len(ids) - 1
    resids = forward_capture(ids)
    final = resids[len(gm.layers) - 1]
    mlx_model_last = np.asarray(unembed(final)[last])
    mo = cos(mlx_model_last, d["model_last"]); mov = len(top5(mlx_model_last) & top5(d["model_last"]))
    print(f"{slug:10} {'MODEL':>5} {'':>10} {mo:9.5f} {mov:>4}/5")
    for l in SRC:
        rc = cos(np.asarray(resids[l]), d[f"resid_{l}"].astype(np.float32))
        lens_last = np.asarray(unembed(resids[l].astype(mx.float32) @ J[str(l)].T)[last])
        lc = cos(lens_last, d[f"lens_last_{l}"])
        ov = len(top5(lens_last) & set(d[f"lens_topk_{l}"][:5].tolist()))
        worst_resid = min(worst_resid, rc); worst_lens = min(worst_lens, lc); min_overlap = min(min_overlap, ov)
        if not (lc > 0.99 and ov >= 4):
            print(f"{slug:10} {l:>5} {rc:10.5f} {lc:9.5f} {ov:>4}/5  <-- MISS")
print(f"\nWORST resid_cos={worst_resid:.5f} lens_cos={worst_lens:.5f} min_top5_overlap={min_overlap}/5")
print("V2", "PASS" if (worst_lens > 0.99 and min_overlap >= 4) else "FAIL")
