# jlens-mlx

Last updated: 2026-07-10

Modular **Jacobian-lens (j-space)** fitting and research on Apple silicon (MLX).

A Jacobian lens is a per-layer linear map that reads, out of a transformer's residual
stream, which vocabulary tokens an activation is *disposed toward* — the model's silent
"workspace" (from Anthropic's 2026 *Verbalizable Representations Form a Global Workspace*
paper). This repo **fits** lenses; the `heylookitsanllm` server **applies** them.

## Why this repo exists

- **Fit our own lenses on Apple silicon** — no PyTorch/CUDA. The killer case: an
  abliterated ("heretic") instruct model has no pre-fit lens available, and a lens must
  be fit on the *edited* weights we actually serve to read what abliteration did.
- **Modular, not a single-arch trainer.** Fitting decomposes into a generic driver +
  a per-architecture Jacobian provider resolved by `model_type`. The generic
  VJP/probed provider works on *any* differentiable MLX model; analytic accelerators
  (e.g. the `qwen3_5` Gated-DeltaNet Metal kernel) are opt-in per-arch. See
  [`docs/DESIGN.md`](docs/DESIGN.md).
- **Corpus choice is load-bearing** (closest to quantization-calibration data). The
  fitting corpus, position mask, and provenance are first-class config, not hardcoded
  WikiText. See [`docs/DESIGN.md`](docs/DESIGN.md).

## Where this sits (three repos, one artifact contract)

| Repo | Role |
|---|---|
| **jlens-mlx** (here) | fit lenses (MLX); produce `lens.safetensors` + sidecar |
| **heylookitsanllm** (server) | *apply* the lens (`/v1/jspace/analyze`, the v3 `jspace` page); consumes the artifact |
| **jacobian-lens** fork (`fblissjr/jacobian-lens`) | the PyTorch reference (kept thin, rebaseable on Anthropic) |

Fitted lens weights (~0.5–3 GB) are **not** committed here — they go to a HuggingFace
lens repo (LFS), the same way the server downloads models. Small parity fixtures may be
committed under `tests/`.

## Status

**Scaffold + apply path GREEN (2026-07-10).** The ported apply path reproduces the reference
oracle on gpt2 (V1 parity, cos 1.00000; `scripts/check_gpt2_parity.py`). The fitter follows
**Anthropic's `jacobian-lens`** — direct end-to-end autograd (`mx.vjp` on MLX), the norm kept
*outside* `J` and applied as the real module at decode — see [`docs/DESIGN.md`](docs/DESIGN.md).

We do **not** vendor jlens-qwen36. We port specific pieces (verified vs `mx.vjp`, attributed
per-file); its only role is an *optional* GDN speed kernel for the 27B qwen. Reference clones
(`jacobian-lens`, `jlens-qwen36`, `jspace`) live outside this repo. Next work: [`MIGRATION.md`](MIGRATION.md).

## Layout

```
jlens_mlx/
  fit.py              # generic chain driver + JacobianProvider protocol + PROVIDER_REGISTRY
  providers/          # per-arch M_l = d(layer)/d(input): generic_vjp (universal), qwen3_5_gdn (accelerator)
  corpus.py           # Recipe + PositionMask + on-policy corpus builder (swappable, provenance-stamped)
  lens.py             # save/load safetensors + sidecar; transport + unembed (apply, mirrors the server)
  verify.py           # parity vs mx.vjp / oracle; held-out per-layer fidelity gate; lens diffing
migrated_from_scratch/# the heylook Phase-1 verifier spike, relocated here
scripts/              # verification gates (check_gpt2_parity, check_rmsnorm_seed, ...)
docs/DESIGN.md        # the fitter + corpus design
```

## Attribution

Built on two Apache-2.0 projects and one MIT design reference — see [`NOTICE`](NOTICE).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
