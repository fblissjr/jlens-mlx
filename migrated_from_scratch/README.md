# jspace Phase 1 verifier harness (scratch, gitignored)

Last updated: 2026-07-09

Reproducible V0/V1 correctness spike for the Jacobian-lens (j-space) MLX port.
See `internal/research/jspace_integration_plan.md` for the full plan.

## Result

V1 PASS on gpt2-small (fp32): MLX apply == genuine `jlens.apply()` to cos 1.00000
across all 11 source layers, top-5 overlap 5/5, on both probe prompts. Residual
cosine 1.0 confirms the mlx-lm gpt2 forward is numerically identical to HF's, so the
block-output capture point is correct.

## Files

- `make_oracle.py` — runs in a THROWAWAY torch venv (torch + transformers + the
  user's `jacobian-lens` clone as editable `jlens`). Produces the golden
  oracle (`oracle_*.npz/.json`) via real `jlens.apply()`, and converts the lens
  `.pt` -> `lens_gpt2.safetensors` + sidecar.
- `mlx_apply.py` — runs in the PROJECT venv. Loads gpt2 via mlx-lm, reuses the
  oracle's `input_ids`, captures block-output residuals, applies
  `wte.as_linear(ln_f(h @ Jᵀ))`, and gates on cos>0.99 + top5>=4/5.
- `lens_gpt2.safetensors` / `.sidecar.json` — converted lens (`J[l]` fp32 + meta).
- `oracle_*.npz` — input_ids, per-layer residuals (fp16), last-position lens logits
  (fp32) + top-10 ids, model logits. `oracle_*.json` — human-readable top-k.

## Rerun

Oracle (throwaway venv already built under the real $TMPDIR/oracle-venv):
    JSPACE_OUT=<this dir> <oracle-venv>/bin/python make_oracle.py
MLX + V1 gate:
    uv run python migrated_from_scratch/mlx_apply.py

## What V1 does NOT yet cover (-> V2, on a gemma)

- gemma `final_logit_softcapping` unembed path (gpt2 has no softcap).
- gemma RMSNorm final norm + input-embedding sqrt(d) scaling (gpt2 uses LayerNorm).
- MoE block-output capture (gemma-4-26b-a4b routing).
- Quantization transfer: served MLX is 8-bit; lens fit in bf16. (gpt2 test was fp32↔fp32.)
- Lenses at a local neuronpedia lens store (full neuronpedia set, local) —
  use gemma-3-270m / gemma-3-1b next to validate softcap+RMSNorm cheaply before the MoE.
