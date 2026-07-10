# Convert a jlens .pt (JacobianLens) to mx-safetensors + sidecar. Torch env,
# NO model load. Args: <lens.pt> <out_dir> <prefix> [hf_name] [softcap]
import json, os, sys
import torch
import jlens
from safetensors.torch import save_file

LENS, OUT, PREFIX = sys.argv[1], sys.argv[2], sys.argv[3]
HF = sys.argv[4] if len(sys.argv) > 4 else ""
SOFTCAP = float(sys.argv[5]) if len(sys.argv) > 5 else None

lens = jlens.JacobianLens.load(LENS)
Jt = {str(l): lens.jacobians[l].contiguous().float() for l in lens.source_layers}
save_file(Jt, os.path.join(OUT, f"lens_{PREFIX}.safetensors"))
json.dump({"prefix": PREFIX, "hf_model_name": HF, "source_layers": lens.source_layers,
           "d_model": lens.d_model, "n_prompts": lens.n_prompts,
           "final_logit_softcapping": SOFTCAP,
           "apply": "unembed(residual @ J[l].T)"},
          open(os.path.join(OUT, f"lens_{PREFIX}.sidecar.json"), "w"), indent=2)
print(f"saved lens_{PREFIX}: d_model={lens.d_model} n_prompts={lens.n_prompts} "
      f"layers={len(lens.source_layers)} [{lens.source_layers[0]}..{lens.source_layers[-1]}]")
