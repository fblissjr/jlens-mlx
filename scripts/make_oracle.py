# V0/V2 golden oracle + lens converter (runs in throwaway torch venv).
# Uses the genuine jlens (a user-provided jacobian-lens clone) for
# authoritative lens_logits. Parameterized by env:
#   HF_MODEL  (default openai-community/gpt2)
#   LENS_PT   (path to the lens .pt; else resolved under $HF_HOME/hub)
#   PREFIX    (default gpt2)      -> lens_<PREFIX>.safetensors, oracle_<PREFIX>_<slug>.npz
#   JSPACE_OUT (output dir)
# Regenerates the golden fixtures consumed by scripts/check_gpt2_parity.py (tests/golden/).
import os, glob, json
import numpy as np
import torch
import transformers
import jlens
from jlens.hooks import ActivationRecorder
from safetensors.torch import save_file

HF = os.environ.get("HF_MODEL", "openai-community/gpt2")
LENS = os.environ.get("LENS_PT") or glob.glob(os.path.join(
    os.environ["HF_HOME"],   # set HF_HOME to your local HF cache, or pass LENS_PT
    "hub/models--neuronpedia--jacobian-lens/snapshots/*/"
    "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"))[0]
PREFIX = os.environ.get("PREFIX", "gpt2")
OUT = os.environ.get("JSPACE_OUT") or os.path.join(os.environ["TMPDIR"], "jspace_oracle")
os.makedirs(OUT, exist_ok=True)

tok = transformers.AutoTokenizer.from_pretrained(HF)
hf = transformers.AutoModelForCausalLM.from_pretrained(
    HF, dtype=torch.float32, attn_implementation="eager").eval()   # eager: softcap-safe
model = jlens.from_hf(hf, tok)                      # force_bos=True by default
lens = jlens.JacobianLens.load(LENS)
print(f"LENS: {lens} | n_layers={model.n_layers} | softcap={model._logit_softcap}")

# ---- converter: J[l] -> safetensors (fp32) + sidecar ----
Jt = {str(l): lens.jacobians[l].contiguous().float() for l in lens.source_layers}
save_file(Jt, os.path.join(OUT, f"lens_{PREFIX}.safetensors"))
json.dump({"prefix": PREFIX, "hf_model_name": HF,
           "source_layers": lens.source_layers, "d_model": lens.d_model,
           "n_prompts": lens.n_prompts, "final_logit_softcapping": model._logit_softcap,
           "apply": "unembed(residual @ J[l].T); unembed = softcap(lm_head(final_norm(x)))"},
          open(os.path.join(OUT, f"lens_{PREFIX}.sidecar.json"), "w"), indent=2)
print("saved lens ->", f"lens_{PREFIX}.safetensors", "| source_layers:", lens.source_layers)

PROMPTS = [
    ("multihop", "Fact: The capital of Japan is Tokyo.\n"
                 "Fact: The currency used in the country shaped like a boot is"),
    ("eiffel", "The Eiffel Tower is located in the city of"),
]

for slug, prompt in PROMPTS:
    lens_logits, model_logits, input_ids = lens.apply(model, prompt)   # all positions
    ids = input_ids[0].tolist()
    last = len(ids) - 1
    with ActivationRecorder(model.layers, at=lens.source_layers) as rec:
        model.forward(input_ids)
        resids = {l: rec.activations[l][0].detach().float().cpu().numpy()
                  for l in lens.source_layers}
    blob = {"input_ids": np.array(ids, dtype=np.int64)}
    topk_readable = {}
    for l in lens.source_layers:
        ll = lens_logits[l]                       # [seq, vocab]
        blob[f"resid_{l}"] = resids[l].astype(np.float16)
        blob[f"lens_last_{l}"] = ll[last].numpy().astype(np.float32)
        tk = torch.topk(ll[last], 10).indices.tolist()
        blob[f"lens_topk_{l}"] = np.array(tk, dtype=np.int64)
        topk_readable[l] = [tok.decode([t]) for t in tk]
    blob["model_last"] = model_logits[last].numpy().astype(np.float32)
    mtk = torch.topk(model_logits[last], 10).indices.tolist()
    np.savez(os.path.join(OUT, f"oracle_{PREFIX}_{slug}.npz"), **blob)
    json.dump({"slug": slug, "prompt": prompt, "input_ids": ids,
               "tokens": [tok.decode([i]) for i in ids], "last_pos": last,
               "source_layers": lens.source_layers, "lens_topk_last": topk_readable,
               "model_topk_last": [tok.decode([t]) for t in mtk]},
              open(os.path.join(OUT, f"oracle_{PREFIX}_{slug}.json"), "w"), indent=2)
    peek = lens.source_layers[-1]
    print(f"[{slug}] ntok={len(ids)} model_top5={[tok.decode([t]) for t in mtk[:5]]}")
    print(f"    lens L{peek} top5={topk_readable[peek][:5]}")
print("DONE ->", OUT)
