# Task 11: validate the served gemma-4-26b-a4b MoE (8-bit VLM) end to end.
# Loads the model exactly as the server does (mlx_vlm.load), resolves it with
# the REAL ModelAdapter, captures late-band residuals through the 128-expert
# MoE routing, applies the solarkyle MoE lens, and checks semantic sanity +
# a model-logits cross-check. No torch oracle (26B is too heavy) -- the MoE
# capture point is already confirmed by mlx_vlm's own code (it matches HF's
# Gemma4TextDecoderLayer hidden_states, the lens's fit convention).
#   uv run python migrated_from_scratch/validate_moe.py
import os
import numpy as np
import mlx.core as mx
from mlx_vlm.utils import load as vlm_load
from heylook_llm.jspace import JSpaceLens, ModelAdapter, capture_residuals
from heylook_llm.jspace.features import band_layers

OUT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.expanduser("gemma-4-26b-a4b-it-8bit-mlx")

lens = JSpaceLens.from_files(os.path.join(OUT, "lens_gemma4moe.safetensors"),
                             os.path.join(OUT, "lens_gemma4moe.sidecar.json"))
print(f"lens: {lens}")
model, processor = vlm_load(MODEL_PATH)
tok = getattr(processor, "tokenizer", processor)
ad = ModelAdapter(model)
print(f"adapter: inner={type(ad.inner).__name__} n_layers={ad.n_layers} softcap={ad.softcap}")

band = band_layers(ad.n_layers, lens.source_layers)
late = band[-5:]
print(f"band=L{band[0]}..L{band[-1]} ({len(band)}); reading late={late}")

def dec(t):
    return tok.decode([int(t)])

def top6(v):
    return [dec(t) for t in np.argsort(-np.asarray(v))[:6]]

BOS = getattr(tok, "bos_token_id", None)

def encode(p):
    ids = tok.encode(p)
    if BOS is not None and (not ids or ids[0] != BOS):   # gemma needs the BOS sink
        ids = [BOS] + ids
    return ids

print(f"bos_token_id={BOS}; sample encode('hi')={encode('hi')}")

PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "The chemical symbol for gold is",
    "The first President of the United States was named",
]
for p in PROMPTS:
    ids = encode(p)
    # Model's real next token (post-norm hidden -> head -> softcap).
    norm_h = ad.inner(mx.array([ids]))
    logits = ad.head(norm_h)
    if ad.softcap:
        logits = mx.tanh(logits / ad.softcap) * ad.softcap
    model_next = dec(mx.argmax(logits[0, -1]).item())
    # Workspace lens over the late band at the answer-onset position.
    res = capture_residuals(model, ids, late, adapter=ad)
    ll = lens.apply(ad, res, positions=[-1], layers=late)
    print(f"\nPROMPT {p!r}  -> model_next={model_next!r}")
    for l in late:
        print(f"  L{l:>2} workspace: {top6(ll[l][0])}")
print("\nDONE")
