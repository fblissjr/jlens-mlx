# Migration + port checklist

Last updated: 2026-07-12

> **PIVOT NOTE (2026-07-10):** the vendored seed was REMOVED (owner preference + a bug we
> caught in it). The fitter now ports from **Anthropic's `jacobian-lens`** (direct
> end-to-end `mx.vjp`, norm outside `J`), not from jlens-qwen36's chain. §2 below (which
> talks about modularizing a vendored seed) is superseded — the new step 2 is "port
> Anthropic's `fitting.py` autograd loop to `mx.vjp`". jlens-qwen36's GDN Metal kernel is an
> optional, later, *ported-not-vendored* speed accelerator. §1 (scratch relocation), §3–§5
> still stand. Reference clones live in a sibling `coderef/`, not in this repo.

How this repo gets from scaffold to "fits our own lens."

## 1. Relocated Phase-1 spike (`migrated_from_scratch/`) — cleanup executed 2026-07-12

The heylook server's gitignored `coderef/jspace_scratch/` verifier spike was moved here
verbatim (its data — `*.npz` / `*.safetensors` — is gitignored). Disposition:

| File | Disposition | Status |
|---|---|---|
| `oracle_eiffel.*`, `oracle_multihop.*`, `lens_gpt2.*` | gpt2 golden fixtures for the standing parity gate | **DONE** → `tests/golden/` (`.json`/`.sidecar.json` tracked, `.npz`/`.safetensors` gitignored); `scripts/check_gpt2_parity.py` reads from there now |
| `make_oracle.py`, `convert_lens.py` | the torch-venv regeneration path for those golden fixtures | **DONE** → `scripts/make_oracle.py`, `scripts/convert_lens.py` |
| `mlx_apply.py` | fully superseded by `scripts/check_gpt2_parity.py` | **DONE** — removed (content remains in git history) |
| `probe_thread.py`, `verify_endpoint.py` | test the running heylookitsanllm endpoint / MLX thread semantics, not this repo — import `heylook_llm.*` and cannot run here | **DONE** — removed; belong in the server repo as real server tests, not reproduced here (content remains in git history) |
| `validate_moe.py`, `verify_router.py`, `verify_module.py` | correction: an earlier pass of this table miscategorized these as "research verification → `tests/` here" — all three actually import `heylook_llm.*` (a different project) and cannot run in this repo | `verify_module.py` is foreign/superseded like `mlx_apply.py` — **DONE**, removed. `validate_moe.py` and `verify_router.py` have reference value — **DONE**, moved to `internal/reference/` (gitignored, local-only, untracked) |
| `mlx_apply_gemma.py`, `lens_gemma22b.*`, `oracle_gemma22b_*.*`, `README.md` | gemma2 real-weights parity gate | **ARCHIVED 2026-07-12** — owner decision: fully out of version control. Moved to `internal/archive/migrated_from_scratch/` (local-only, gitignored). Recoverable from git history; the gitignored binaries are regenerable via `scripts/make_oracle.py` + `scripts/convert_lens.py` (torch venv). A future gemma2 parity gate starts from that archive or regenerates |

After this pass the folder was DISSOLVED entirely (2026-07-12): everything graduated
into the repo's real layout (`tests/golden/`, `scripts/`), was removed (recoverable from
git history), or was archived locally under `internal/archive/migrated_from_scratch/`.

## 2. Modularize the vendored `qwen3_5` seed

`vendor/jlens_qwen36/` is copied verbatim (Apache-2.0). Wire it into the modular package
WITHOUT changing its numerics (the seed is validated — analytic branches vs `mx.vjp`,
kernel vs ops, golden gate). Keep provenance headers.

- [ ] `jlens_mlx/fit.py` — generic driver: chain `J_{l-1} = J_l @ M_l`, seed final-norm
      Jacobian, average over corpus. (Port from `vendor/.../fit_analytic.py` +
      `analytic.py`, stripped of qwen specifics.)
- [ ] `jlens_mlx/providers/generic_vjp.py` — universal baseline via `mx.vjp` /
      Rademacher probes (port `vendor/.../fit.py::per_layer_jacobian` +
      `probing.py`). NO custom kernel; works on any differentiable MLX model.
- [ ] `jlens_mlx/providers/qwen3_5_gdn.py` — the accelerator: analytic branch assembly
      (`analytic_layer.py`, `analytic_attn.py`) + the Metal GDN backward
      (`custom_gdn_vjp.py`, `gdn_backward.py`, `patch_gdn.py`, `custom_gdn_patch.py`).
      Register it in `providers/__init__.py` under `model_type == "qwen3_5"`.
- [ ] Reach the text stack of `Qwen3_5ForConditionalGeneration` via
      `.language_model.model` (multimodal/MTP wrapper) — same walk as the server's
      `capture.py`.

## 3. First own-fit (Metal-gated — a focused session on the Mac)

Target: `Qwen3.5-27B-abliterated-8bit-mlx` (confirmed `qwen3_5`: 64 layers,
`full_attention_interval=4` → 48 GDN + 16 full-attn, `d_model` 5120).

- [ ] A small **chat + safety** corpus recipe (chat-templated, assistant/think-span
      position mask, include would-have-refused prompts so the abliterated circuitry is
      live). NOT WikiText.
- [ ] Fit; **held-out fidelity gate** (per-layer KL / top-k vs true logits on held-out
      target-distribution data) — refuse to save if it fails.
- [ ] Convert → `lens.safetensors` + sidecar (recipe + model SHA + position policy as
      provenance); drop into the server's `adapters/jspace/Qwen3.5-27B-abliterated-8bit-mlx/`.

## 4. The first finding

- [ ] **Lens diff**: fit a stock-Qwen lens and a abliterated lens; diff their readouts along
      the refusal directions. The diff is the finding (what abliteration nulled vs rerouted).

## 5. Publish (only after step 3 succeeds)

- [ ] HuggingFace lens repo (LFS) for OUR fitted lenses. Do NOT republish the converted
      third-party lenses (gemma ones carry the Gemma Terms of Use).
- [ ] Once MLX-native lenses are on HF, the server's `scripts/jspace_convert_lens.py`
      collapses to a pure download (no torch on the server).
