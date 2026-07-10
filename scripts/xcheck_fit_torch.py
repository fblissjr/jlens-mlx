"""Torch side of the fitter cross-check: fit gpt2 with Anthropic's `jlens` and dump J + ids.

The rigorous gate the "correct by construction" baseline needs: does our MLX fitter produce
the same J as the reference torch fitter on the same corpus? This script runs Anthropic's
fit() and writes scripts/_xcheck_torch.npz; xcheck_fit_mlx.py fits on the SAME token ids and
compares.

Run in a throwaway torch env (JLENS_SRC = path to the anthropics/jacobian-lens clone):
    JLENS_SRC=<path-to-your-jacobian-lens-clone> \
    uv run --with torch --with transformers --with numpy --with safetensors \
           --with huggingface_hub python scripts/xcheck_fit_torch.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ["JLENS_SRC"])  # the jacobian-lens clone (kept out of this file)
from jlens.fitting import fit  # noqa: E402
from jlens.hf import from_hf  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

OUT = Path(__file__).resolve().parent / "_xcheck_torch.npz"
MODEL = os.environ.get("MODEL", "openai-community/gpt2")  # id or local snapshot path
PROMPTS = ["The Eiffel Tower is in the city of", "The capital of Japan is"]
SRC = [int(x) for x in os.environ.get("SRC", "9,10").split(",")]
SKIP_FIRST = int(os.environ.get("SKIP_FIRST", "1"))
FORCE_BOS = os.environ.get("FORCE_BOS", "0") == "1"  # gemma needs BOS; gpt2 does not
MAX_SEQ = int(os.environ.get("MAX_SEQ", "64"))


def main() -> None:
    torch.manual_seed(0)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval().to("cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = from_hf(hf, tok, force_bos=FORCE_BOS, compile=False)
    lens = fit(model, PROMPTS, source_layers=SRC, target_layer=None,
               skip_first=SKIP_FIRST, max_seq_len=MAX_SEQ, dim_batch=8, checkpoint_path=None)

    data = {f"J_{l}": lens.jacobians[l].detach().cpu().float().numpy() for l in SRC}
    for i, p in enumerate(PROMPTS):
        data[f"ids_{i}"] = model.encode(p, max_length=MAX_SEQ)[0].cpu().numpy().astype(np.int64)
    data["src"] = np.array(SRC)
    data["skip_first"] = np.array(SKIP_FIRST)
    data["target"] = np.array(model.n_layers - 1)
    data["model"] = np.array(MODEL)
    np.savez(OUT, **data)

    print(f"torch fit done -> {OUT}")
    for l in SRC:
        j = data[f"J_{l}"]
        print(f"  J_{l} shape={j.shape}  ||J||/sqrt(d)={np.linalg.norm(j) / (j.shape[0] ** 0.5):.3f}")


if __name__ == "__main__":
    main()
