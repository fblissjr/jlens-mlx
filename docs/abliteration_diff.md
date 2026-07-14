# Abliteration diff — finding

Last updated: 2026-07-14

> **2026-07-14 — clean matched-pair update supersedes the benign-floor conclusion.**
> Everything below (§"The result" onward) is the ORIGINAL pair
> (`band-n14-stock` vs `band-n14-fixed`), fit on the mlx-community 8-bit builds of the
> stock model and the **coder3101** abliteration — a pair that differs in TWO ways at once
> (third-party converter AND abliteration recipe). We since built a **clean matched pair**
> (self-converted base + our own controlled abliteration, both via mlx-vlm 0.6.5, so the ONLY
> variable is the abliteration) and re-ran the per-prompt diff. It **reverses caveat 1**: on the
> clean pair the content floor HOLDS. Read the clean-pair section first; treat the old
> "benign floor falsified" as an artifact of the uncontrolled pair.

## Clean matched-pair result (2026-07-14) — magnitude vs content decouple

**Pair:** `A(stock)=out/pair-base`, `B(abliterated)=out/pair-heretic-t144`, substrate = the
self-converted base `Qwen3.5-27B-8bit-ours`. Both lenses fit on the **same** shared corpus,
band 16..47, target=63. The only A/B difference is the abliteration (both models 8-bit-converted
by the identical mlx-vlm 0.6.5 pipeline — the converter confound is eliminated, cf. caveat 2).
Abliteration recipe = our Trial 144: a **deep tent (L42–59)**, unlike coder3101's shallow L33/36.

**Pooled diff (`out/diff_clean_pair.log`, 1984 reads):** two readable clusters, both DEEPER than
the old pair, tracking the deep tent:
- **Safety-as-concept, L33–L40:** `Safety`, `安全风险` (safety-risk), `安全性` (safety),
  `安全问题` (safety-issue), `безопасность` (RU: security), `不承担任何` (assumes-no-liability).
- **Refusal-as-speech-act, L41–L47 (band edge, still RISING at L47):** `抱歉` (sorry), `Sorry`,
  `我不会` (I-won't), `违反了` (violated), `示威`/`protests`, plus loaded topics `China`/`Israel`/`Russia`.
- The largest raw `l2` (L16–L22, ~590–987) is **structural**, not safety: `幂`/`power`, `#`,
  `...**`, `<|box_end|>`. The safety-semantic shift is real but sits at moderate `l2` BELOW the
  shallow structural churn — different from coder3101, whose safety cluster sat shallower (L32–42).
- The refusal cluster climbing at L47 says its peak is **past the band** — Trial 144's tent centers
  L42–59. A deep-band fit (L48–59, both models, same corpus → `out/pair-*-deep`) is running to
  confirm the peak location; the 16..47 band structurally cannot see it.

**Per-prompt diff (`out/per_prompt_clean_pair.log`) — the decoupling, and the floor reversal:**
- **Magnitude (l2) is prompt-INDEPENDENT (flat).** Mean l2 over L32–42: benign:recipe **452.5**,
  safety:lock **457.4**, safety:chemicals **460.1**, benign:crypto-hist **404.5**. The
  worst-benign / best-safety margin is ~1% — magnitude does **not** discriminate. This is exactly
  what a static weight edit must do: its size can't depend on the prompt.
- **Content (top_up tokens) is prompt-CONDITIONAL.** At L34 the same edit surfaces:
  - safety:chemicals → `安全风险` `Safety` `安全性` `安全问题` `危险性` `安全隐患` (pure safety)
  - safety:lock → `不承担任何` `安全性` `安全风险` `Safety` `没有任何` (pure safety/refusal)
  - benign:recipe → `**:` `**,` `**` `Nothing` `nothing` (pure **formatting**, no safety)
  - benign:crypto-hist (security-adjacent) → mostly formatting, but `Safety` `安全风险` leak in at rank 6–8
- L37 repeats the pattern; by L40 it washes to noise for all (the cluster has moved to
  refusal-proper at L41–47).

**Headline (corrected, unified):** abliteration installs a **static directional transport edit**
whose **magnitude is prompt-independent** (fixed weight edit) but whose **token-space content is
prompt-conditional** — because a fixed direction read through prompt-dependent residuals surfaces
different tokens. It surfaces safety vocabulary precisely on safety prompts, formatting noise on a
truly benign prompt, and partial leakage on a benign-but-topic-adjacent prompt. **This reverses the
old "benign floor falsified":** on a properly controlled pair the content floor HOLDS. The earlier
falsification confounded converter + recipe and does not generalize.

---

## Original pair (coder3101 via mlx-community) — 2026-07-13

Last updated (this section): 2026-07-13

Diff-of-Jacobians (`scripts/diff_lenses.py`, `verify.diff` semantics) between two
own-fit band lenses, both fit on the **same corpus** (same token sequences, item 10
skipped in both), so the only variable is the model that produced the transport:

- **A (stock)** = `out/band-n14-stock`  ← lens fit on `Qwen3.5-27B-8bit-mlx`
- **B (abliterated)** = `out/band-n14-fixed`  ← lens fit on `Qwen3.5-27B-heretic-8bit-mlx`
- Shared band: layers 16..47 (32 layers). `B - A` = what abliteration's transport does
  *differently*, holding activations fixed.

Ran **both substrate directions** (the script flags substrate-dependence as a confound):
- `diff_ablit_vs_stock.txt` — stock model as substrate (neutral default)
- `diff_ablit_vs_stock_hereticsub.txt` — heretic model as substrate

Each report pools 1984 (prompt × position) reads over 4 held-out prompts (1 benign
recipe control + 3 safety-adjacent).

## The result (robust, substrate-independent)

The two substrate runs agree layer-for-layer. The finding is therefore a property of
the **lens pair** (the transport-geometry difference), not of which residuals are fed.

**Mid-to-late band (L32–L42) — the signal, clean and consistent:**

- `top_up` (abliterated surfaces MORE): `Safety`, ` safety`, ` unsafe`, ` Unsafe`,
  ` unethical`, ` dangerous`, `Cannot`, `Nothing`/` nothing`/`Impossible`,
  ` violations`, `Violation`, and the CJK equivalents `安全风险` (safety-risk),
  `安全教育` (safety-education), `安全隐患` (safety-hazard), `违反`/`违反了` (violate/-d),
  `不能使用` (cannot-use), `别无` (no-alternative).
- `top_down` (abliterated SUPPRESSES): geography/nationality — `China`, `cn`, `_cn`,
  `-China`, `中国站`, `Egypt`, `Portugal`, `Greece`, `Italy`, `Spain`, `Austria`;
  plus retrieval/UI verbs `查看`/`viewing`/`Viewing`/`fetching`/`-fetch`, `写入` (write).

**Early band (L16–L31):** largest raw `l2` (~520–625) but `top_up` is dominated by
illegible `*`-prefix subword junk (`*B`, `*S`, `(*`, `((*`) and CJK noise — discount
it as not-yet-legible band disagreement (matches the readout finding that the early
band isn't interpretable). Two real exceptions worth noting: **L16/L18 surface
`can`/`Can`/`可以` (can / able-to / may)** on the +side, and L20–L22 surface code-ish
tokens (`[method`, `[Int`, `<Response`).

## Interpretation (the headline, and why it's the opposite of the naive guess)

Naive expectation: abliteration *removes* refusal, so B−A should surface safety
**less**. It does the opposite — the abliterated transport surfaces safety/refusal
vocabulary **more** in the mid-late band.

CORRECTED 2026-07-13 (the weight footprint below, plus reading the Heretic source,
overturned an earlier tidy-but-wrong framing). Abliteration edits the **transport**,
NOT the readout. Heretic performs directional ablation: it orthogonalizes the
residual-**writing** matrices — every layer's `attn.o_proj` and `mlp.down_proj` — against
the refusal direction `r = mean(harmful) - mean(harmless)`. Those matrices are the tail
blocks, which sit **inside** J. The readout it leaves alone: `model.norm` is
**bit-identical** between the two models, `lm_head`/`embed` at the quant floor (see
`out/abliteration_footprint.txt`). So my first write-up ("abliteration edits the readout
outside J") was WRONG. The correct — and stronger — statement: the readout is shared A/B,
so the lens diff is a **pure transport difference**, which is exactly why the lens is the
right instrument.

What it reads out is the **direction of the weight edit itself**. Heretic removed the
refusal-behavior axis (harmful-vs-harmless means) from `o_proj`/`down_proj`, and that
removal has a fixed vocabulary-space signature: safety/refusal tokens. The tell: the
surfaced tokens (`unethical`/`illegal`/`harmful`/`violat…`) are almost verbatim Heretic's
own refusal-keyword list. So the diff correctly **characterizes what abliteration changed**
(a refusal-direction edit) and **where** (§ footprint) — but see the retraction below on
what it does NOT mean.

**RETRACTED 2026-07-13 (later, per the per-prompt test):** an earlier version of this
write-up said "the concept persists in the transport; the lens sees what the behavior
hides" — implying a *content-conditional* disposition (the model still "knows" a harmful
prompt is dangerous). The per-prompt benign-floor test **falsified** that (see caveat 1).
The effect is **prompt-independent**: it is the static weight-edit fingerprint, readable
on a benign recipe prompt as strongly as on a safety prompt. Correct statement: the lens
recovers *what* abliteration did and *where*, uniformly across inputs — not a
content-conditional internal state.

The suppressed geography cluster (China/Europe) + retrieval verbs is the
complementary half — worth a second look, but the safety axis is the load-bearing one.

## Weight footprint — independent cross-validation (2026-07-13)

`scripts/abliteration_footprint.py` dequantizes both 8-bit builds and reports the relative
weight delta `||W_B - W_A|| / ||W_A||` per layer — no lens involved (`out/abliteration_footprint.txt`):

- **Vision tower bit-identical** (max |B-A| = 0.0) → abliteration is LM-only. Confirms the
  vision path is untouched.
- **Readout untouched**: `model.norm` = 0.00000, `lm_head` 0.00375, `embed` 0.00266 (floor).
- **The edit is ~6x concentrated in the residual-writing matrices**: `mlp.down_proj` 0.0264,
  `self_attn.o_proj` 0.0241, `linear_attn.out_proj` 0.0226; every input matrix (q/k/v, gate,
  up) at the ~0.004 floor. Textbook directional ablation, confirmed against the Heretic source.
- **Per-layer shape = Heretic's converged tent schedule**: flat L0-13 → ramp L14-24 →
  plateau L25-50, **peaking L33 (0.039) / L36 (0.034)** → taper L51-63.
- **THE CROSS-VALIDATION**: the weight-edit peak (L33/36) sits inside the transport-diff safety
  cluster (L32-42, peaking L33-37). Two measurements sharing no machinery — weight-space edit
  magnitude vs transport-space readout shift — agree on **where** abliteration lives. That is
  what turns the finding from correlational to mechanistic.

## Caveats (do not overclaim)

1. **Benign floor — FALSIFIED on the old pair, then REVERSED on the clean pair (2026-07-14).**
   On the ORIGINAL (uncontrolled) pair, `scripts/per_prompt_diff.py` showed the benign recipe
   lighting up L32-42 as strongly as safety prompts (mean l2 596 vs 524-571) with refusal vocab in
   its own top_up → read as "no benign floor, effect is prompt-independent." **The clean matched pair
   overturns this** (see the clean-pair section up top): there, l2 magnitude is still flat across
   prompts (~450 everywhere), BUT the benign recipe's top_up is pure **formatting** (`**`/`Nothing`),
   not safety vocab, while safety prompts surface safety vocab. So the correct decomposition is
   **magnitude prompt-independent, content prompt-conditional** — and the content floor HOLDS. The old
   falsification confounded converter + recipe (the old pair differs in both) and does not generalize.
   The `l2`-magnitude reading of "prompt-independent" survives; the content reading of it does not.
2. **Quant-converter match — MEASURED, CLOSED (2026-07-13).** We self-converted the base
   `Qwen/Qwen3.5-27B` to 8-bit MLX (mlx-vlm 0.6.5, same affine/64/8 params) and diffed it
   against the mlx-community base — two conversions of the SAME weights, so the delta is pure
   converter drift (`out/converter_drift_base_vs_mlxcommunity.txt`). Result: **uniform ~0.003-0.004
   everywhere, no tent, and `o_proj`/`down_proj` are among the LOWEST-drift projections (0.0029/
   0.0033) — the opposite of the abliteration fingerprint.** So converter asymmetry is ~8x below
   the abliteration signal (o_proj/down_proj 0.024-0.026, tent peaking L33/36) and structureless.
   It cannot manufacture the finding. The early-band `*`-junk is illegible-band noise, not converter
   drift. Caveat closed.
3. Legibility metric still misleads (established on band-n14-fixed) — this finding rests
   on the readout tokens by eye, not on any per-layer score.
