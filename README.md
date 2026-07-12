# jlens-mlx

Last updated: 2026-07-10

Modular **Jacobian-lens (j-space)** fitting and research on Apple silicon (MLX).

A Jacobian lens is a per-layer linear map that reads, out of a transformer's residual
stream, which vocabulary tokens an activation is *disposed toward* — the model's silent
"workspace" (from Anthropic's 2026 *Verbalizable Representations Form a Global Workspace*
paper). This repo **fits** lenses; the `heylookitsanllm` server **applies** them.

## Why this repo exists

- **Fit our own lenses on Apple silicon** — no PyTorch/CUDA. The killer case: an
  abliterated instruct model has no pre-fit lens available, and a lens must
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
The **`qwen3_5` GDN tail is ported + verified (2026-07-10)**: per-layer fa/ssm mask dispatch
plus the jlens-qwen36 Metal GDN backward as a custom VJP on the stock fused forward — gated
vs `mx.vjp` at every grain (`scripts/check_qwen3_5_synthetic.py`; kernel rel err ~3e-7,
whole-fit J cos 1.000000 vs the pure-autodiff path, ~8x faster on even a tiny model).

**Fitting pipeline complete (2026-07-10 PM).** On top of the above:
- **Cotangent dim-batching** (`providers/generic_vjp.py`) — batches the D output-dim rows through the
  tail's native batch axis; **2.4×**, verified == the one-at-a-time path (rel 2e-7).
- **Exact reverse-mode CHAIN fitter** (`chain.py`) — fits ALL source layers in ONE backward sweep
  (O(n_blocks) vs O(n_source·avg_tail)); **~20× on a dense band fit**, EXACT (verified == direct on
  qwen3_5 + gpt2, cos 1.000000; `scripts/check_chain_vs_direct.py`). The `fit_corpus`/`fit_lens`
  default (`use_chain=True`); direct path stays the golden reference + fallback. Caveat: the gemma
  array-mask branch is un-gated (`use_chain=False` there).
- **Corpus builder** (`corpus.py`) — streaming HF load + weighted strata + chat-template + role-aware
  position masks; on-policy generation is a separated GPU step. 21 CPU unit tests (`tests/test_corpus.py`).
- **Held-out fidelity gate + lens diff** (`verify.py`) — per-layer top-1/top-k/KL vs true logits with a
  KL-based identity tripwire (quantization-tolerant); two-lens diff for the abliterated-vs-stock finding.
- **First band-targeted own-fit** on the served abliterated Qwen3.5-27B (`scripts/fit_band_corpus.py`) —
  corpus → `fit_corpus` over the product band (layers 16–47) → gate → provenanced save. Timestamped
  progress + a positions-weighted ETA + a `progress.json` sidecar + a hang watchdog make a multi-hour
  run observable; for an overnight run wrap it in `scripts/fit_band_supervisor.sh`, which restarts on
  an unexpected exit (native crash) but stops on success/config-error/degenerate-corpus.

We do **not** vendor jlens-qwen36. We port specific pieces (verified vs `mx.vjp`, attributed
per-file); its only role is an *optional* GDN speed kernel for the 27B qwen. Reference clones
(`jacobian-lens`, `jlens-qwen36`, `jspace`) live outside this repo. Next work: [`MIGRATION.md`](MIGRATION.md).

## Layout

```
jlens_mlx/
  fit.py              # direct end-to-end VJP fitter (per-layer tails) + fit_lens/fit_corpus (chain-default)
  chain.py            # exact reverse-mode CHAIN fitter (one sweep, O(n_blocks)); VERIFIED == fit.py
  providers/          # arch-specific tail pieces: generic_vjp (dim-batched, universal), qwen3_5_gdn (GDN kernel)
  corpus.py           # Recipe + PositionMask + streaming on-policy corpus builder (provenance-stamped)
  lens.py             # save/load safetensors + sidecar; transport + unembed (apply, mirrors the server)
  verify.py           # held-out per-layer fidelity gate (KL identity tripwire) + lens diffing
tests/                # CPU unit tests (test_corpus.py) + golden/ gpt2 parity fixtures; run: uv run pytest tests/ -q
scripts/              # gates (check_{gpt2_parity,qwen3_5_synthetic,chain_vs_direct}, ...) + fit drivers
docs/DESIGN.md        # the fitter + corpus design
```

## Attribution

Built on two Apache-2.0 projects and one MIT design reference — see [`NOTICE`](NOTICE).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
