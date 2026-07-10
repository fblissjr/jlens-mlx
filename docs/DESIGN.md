# jlens-mlx design

Last updated: 2026-07-10

> **PIVOT NOTE (2026-07-10):** §1 below describes the original jlens-qwen36-style
> *chain* design (`J_{l-1} = J_l · M_l` + closed-form norm seed). We have since PIVOTED
> to **Anthropic's `jacobian-lens` design**: fit each `J_l` as a **direct end-to-end
> `mx.vjp`** (autograd, no chain), with the **norm kept OUTSIDE `J`** and applied as the
> real module at decode — which is correct-by-construction and designs away both the
> `rms²`-seed bug and the chain-indexing off-by-one we caught in jlens-qwen36. §2 (corpus)
> and §3 (vision) still stand. The authoritative statement is "The modular fitter" in the
> heylook `docs/jspace_integration_plan.md` Part 2; this file will be rewritten to match.
> The GDN Metal kernel is now an *optional* speed accelerator, not the core, and is not
> vendored.

The two design commitments that keep this from becoming a single-arch trainer with a
mediocre hardcoded corpus.

## 1. The modular fitter

Fitting a Jacobian lens decomposes into two parts. Only the second is arch-specific.

### Generic driver (arch-agnostic) — `jlens_mlx/fit.py`

Chain per-layer Jacobians through depth:

```
J_{L}   = closed-form final-norm Jacobian        (seed)
J_{l-1} = J_l @ M_l                               (M_l evaluated at layer l's INPUT acts)
```

averaged over corpus prompts/positions, saved as `lens.safetensors` (`J[l]` per source
layer) + a JSON sidecar. This is the apply path run backward; it does not know or care
what architecture produced `M_l`.

Chain-indexing invariant (the subtle bug both references hit): `J_l` transports
`acts[l]` = the residual AFTER layer `l`; `M_l` is evaluated at layer `l`'s *input*
`acts[l-1]`. Getting this wrong gives 33–49% rel error instead of 3–5%.

### Per-arch Jacobian provider — `jlens_mlx/providers/`

`M_l = d(layer_l)/d(input)` is the ONLY arch-specific piece. A registry keyed by
`model_type` resolves a provider, with a fallback ladder:

- **`generic_vjp`** — `mx.vjp` through the decoder layer with `D` one-hot cotangents, or
  `k` Rademacher probes (`M̂ = (1/k) Σ vᵢ(vᵢᵀM)`, unbiased) for a ~10× readout-grade
  speedup. Works on **any** differentiable MLX model. Slow, no custom kernel. **Default.**
- **analytic accelerator** (opt-in, per-arch) — closed-form/assembled Jacobians. For
  `qwen3_5` this is the analytic branch-Jacobian assembly + the custom Metal GDN backward
  kernel (MLX's fused GDN kernel has no VJP; the ops fallback is ~22× slower). ~30–60×
  faster than the exact VJP and, measured, *more* accurate.

```python
JacobianProvider = Protocol:
    def final_norm_jacobian(model) -> mx.array          # the chain seed
    def layer_jacobian(model, layer_idx, acts) -> mx.array   # M_l at this layer's input acts

PROVIDER_REGISTRY = {"qwen3_5": Qwen35GdnProvider, ...}   # default: GenericVjpProvider
```

**Coverage:** `qwen3_5` (Qwen3.5/3.6-27B, our served heretic) → accelerator. Everything
else → generic. Gemma-4 MoE stays on the heylook offline-torch convert path until it
earns an accelerator (its MoE attention/MLP need their own analytic derivation, or accept
the slow generic VJP).

## 2. Corpus is load-bearing (not a hardcoded WikiText)

Fitting a lens is closest to **quantization calibration**: you estimate a moment of the
activation distribution (`E[J(x)]` over corpus activations, like GPTQ's `E[xxᵀ]`) and
deploy it on a possibly-different distribution. Failure is smooth, not loud — you just
linearize in the wrong region. Two amplifiers:

- **Chained product through depth** → early-layer lenses inherit every downstream layer's
  mismatch and need the most data.
- **Circuit coverage** (the control-vector lesson) → a circuit the corpus never activates
  contributes ~0 to the averaged Jacobian, so the lens is structurally blind to it.

**Why WikiText is actively wrong for our case:** an abliterated model's refusal/safety
circuitry is dormant across a WikiText corpus (≈0 refusal-triggering content), so the lens
goes blind along exactly the directions abliteration edited — the directions we most want
to read. A WikiText lens reports "nothing unusual" by construction.

### First-class, swappable (`jlens_mlx/corpus.py`)

- **`Recipe`** — {prompt sources, mixing weights, prompt count, seed}, stamped onto the
  lens sidecar as provenance (recipe + model SHA + position policy). No lens without it.
- **Chat-templated by default** (render through the model's own template); keep one
  raw-prose control arm.
- **`PositionMask`** — average over assistant / think-span tokens; explicitly drop
  BOS/sink/role tokens (high-norm Jacobian outliers). NOT a hardcoded "skip first 4"
  (that heuristic is for raw-text BOS sinks; wrong under ChatML).
- **On-policy builder** — fit on the model's own sampled generations at generated-token
  positions, mixed ~50–70% with human-written diversity.

### Held-out fidelity gate (`jlens_mlx/verify.py`)

Per-layer KL / top-k agreement between the lens readout and true logits, on held-out data
**from the target distribution**. Refuse to save a lens that fails. Never grade a lens on
its own fitting corpus.

### Lens diffing

Compare two lenses (WikiText-fit vs chat-fit; stock vs heretic) as a first-class op. For
the abliteration study, **the diff is the finding.**

## 3. Vision (later)

Image tokens are projector outputs off the text-embedding manifold (larger norms,
different attention topology). A text-fit lens applied at image positions evaluates far
off-manifold. **Stratify** — a modality-conditioned image-position lens — rather than pool
(averaging two separated clusters linearizes around a midpoint faithful to neither).
Validate fidelity per-modality; image positions get their own mask.
