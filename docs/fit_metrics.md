# Fit metrics + the open-questions agenda

Last updated: 2026-07-12

Every band fit generates data — per-item timing, memory, positions — that until now lived
only as unstructured log lines and got thrown away. This doc describes the structured store
that captures it (`scripts/fit_metrics.py` + a DuckDB), the read-only dashboard over it
(`scripts/fit_metrics_ui.py`), and the research questions the store exists to answer. It also
records, honestly, what we got wrong about memory this session and why the answer is
"measure, don't predict."

Companion visual explainers (local-only, `internal/`): `fit_anatomy.html` (where the memory
goes) and `corpus_onpolicy_explainer.html` (reading the corpus).

---

## 1. Why a metrics store

The fit is our own instrument, and the measure-first ethos applies to the fit itself: a run
should leave behind an analyzable dataset, not a log you `grep`. Framed dimensionally (one row
per item-fit):

- **`fact_item_fit`** — grain: one (run, item). Facts: `wall_time_s`, `peak_gb`, `sec_per_pos`,
  `sec_per_chunk`, `n_chunks`. Item attributes denormalized on the fact for join-free querying:
  `seq_len`, `n_positions`, `stratum`, `on_policy`, `chunk_size`. Deterministic surrogate key
  (`item_fit_key` = md5 of `run_id:item_index`); append-only (`ON CONFLICT DO NOTHING`).
- **`dim_run`** — one row per fit run: model, band, target, `n_items`, `enable_thinking`,
  `max_seq_len`, `shared_fraction_overall`/`_onpolicy`, `dropped_over_len`, `use_chain`,
  `git_sha`. Upserted on `run_id` (= md5 of the out-dir basename).
- Views: `v_peak_vs_seq`, `v_peak_vs_positions` (peak memory vs. fitted positions —
  the axis that actually drives peak, per §3/§4), `v_throughput`.

The DB is stable and shared across runs (`out/fit_metrics.duckdb`, gitignored), so every fit
accumulates rows and cross-run analysis becomes a query.

## 2. Usage

Runs under the **heylook venv** (duckdb lives there, not in the jlens venv); pure stdlib +
duckdb, no MLX.

```sh
# ingest a run's completed items (idempotent; re-run as more items finish)
cd <heylook repo> && uv run python <jlens>/scripts/fit_metrics.py --out <jlens>/out/band-n14-fixed

# print a view without ingesting
uv run python <jlens>/scripts/fit_metrics.py --query peak_vs_seq
uv run python <jlens>/scripts/fit_metrics.py --query peak_vs_positions
uv run python <jlens>/scripts/fit_metrics.py --query throughput

# the dashboard (read-only, styled to heylook v3's design system) on :8766
uv run python <jlens>/scripts/fit_metrics_ui.py --db <jlens>/out/fit_metrics.duckdb
```

The ingester parses the fit's completion lines (`item N/M done in Xs (P pos, Y s/pos) ... peak
ZGB`) and derives per-item `chunk_size` by scanning back to the nearest heartbeat's `chunk c/N`
(so a run that changed chunk mid-way — as band-n14-fixed did, 128→64 — records the true
per-item value).

## 3. The open questions this store is built to answer

Grouped, each with the data that answers it, whether we already collect it, and difficulty.
"Being collected" = the running fit emits it; the store just structures it.

### Memory mechanics (where predictions failed — see §4)
- **M1 — how does peak scale? ANSWERED, then CORRECTED (measured 2026-07-12): the driver is
  fitted POSITION count, not sequence length.** The original "~1.7GB/token" seq-length slope
  was a confound. Fuller data from band-n14-fixed, as (fitted_positions, seq_len, peak_gb):
  (1, 21, 65.3), (4, 27, 75.6), (47, 72, 162.6), (47, 73, 164.3), (47, 74, 166.0), (47, 77,
  161.1), (47, 78, 165.8). At fixed positions=47, peak is FLAT across seq 72->78
  (161-166GB) — a real seq-length driver at the old ~1.7GB/token slope would put seq 72 vs 78
  roughly 10GB apart; instead the spread is noise. Peak clusters by POSITION count instead:
  1-4 positions -> 65-76GB, 47 positions -> 161-166GB. **Corrected model: ~63GB base +
  ~2.1GB per fitted position.** The confound: the original sample's short-sequence items also
  happened to have few fitted positions, so seq length and position count were correlated in
  that sample and the wrong one got credited as the driver. See the corrected item-10 read in
  §4. `v_peak_vs_positions` (`--query peak_vs_positions`) is now the query to reach for —
  position count (`n_positions`), not `seq_len`, is the axis that actually matters;
  `v_peak_vs_seq` is kept for the seq-length residual check, not as the primary read.
- **M2 — what is the true internal breakdown of the peak? Still open, and now the priority
  follow-up.** Data: `mx.get_active_memory` / `get_cache_memory` sampled around chain-sweep
  phases. Not collected. Difficulty: medium (coarse split cheap; per-tensor hard). This is
  genuinely unexplained, not just under-measured: first-principles estimates of what the chain
  sweep should retain range **34GB-320GB** depending on assumptions, and none match the
  measured ~161GB for a 77-token item; chunk-independence (§4, misprediction 3) rules out the
  obvious dim-batch-cotangent hypothesis. Cheap to run; it is the prerequisite for any real
  reduction and the honest end of the guessing this section documents. NOT urgent (the current
  fit is unblocked by the §4 workarounds) but a standing liability — it blocks longer-context
  transfer experiments, the stock-model diff, and item-batching, all of which want headroom.
- **M3 — does checkpointing help at real scale, and where does the freed memory live?** Data:
  the `feat/checkpoint` branch (built, equality-gated, unproven at real scale) run on the real
  27B + phase-level memory reads. Not collected. Difficulty: easy-ish but GPU-gated. This is the
  one lever that could reduce a SINGLE item's peak — the actual fix for the item-10 class of
  problem (fit long items instead of dropping them via `JLENS_MAX_FIT_SEQ`, §4). Do M2 before
  M3 — instrumentation should inform where checkpointing would actually help.

### Throughput
- **T1 — optimal `chunk × item_batch × checkpoint`?** Data: the bench matrix. Partly collected
  (chunk 64 vs 128 already in the store). Difficulty: medium (GPU time).
- **T2 — is the "~20x chain speedup" real?** (Claimed, never head-to-head.) Data: time chain vs
  direct, same layers. Difficulty: easy on a shallow band.

### Fit quality — the actual research goal (hardest, most important)
- **Q1 — is the clean-corpus lens good, and does the "fidelity gate misleads" finding replicate
  off the degenerate band-5L?** Data: `readout.py` + fidelity gate on the new fit. Difficulty:
  easy once a lens lands.
- **Q2 — does fit-at-128 transfer to long contexts?** Data: fit @128, grade fidelity
  @1024/4096 by position depth. Difficulty: medium.
- **Q3 — does corpus diversity (`shared_fraction`) predict fit quality — is 0.5 the right
  gate?** Data: fit quality across corpora of varying diversity. Difficulty: high (multiple
  full fits). The store is the enabler: once per-run fit-quality metrics land in a
  `dim_fit_quality`, this is a cross-run join, not a research project.
- **Q4 — the abliterated-vs-stock diff (the finding we're chasing).** Data: a **stock** Qwen fit
  too. Difficulty: high (another overnight fit).

The store makes the **resource** questions (M1, T1, T2) trivial-to-medium and *frames* the
**quality** questions — it does not make the quality questions easy. Those are gated on a
finished lens plus a metric we still owe (the fidelity gate misleads on band layers; a
disposition-aware metric is the replacement).

## 4. What we got wrong about memory — the cautionary record

This session mispredicted MLX fit memory **four times**, each corrected only by measurement:

1. **"It plateaued at 107 GB."** It hadn't — the peak climbs across the chain sweep as
   cotangents accumulate down the band; the true single-item peak was ~166 GB.
2. **Checkpointing ~60 GB estimate.** Drawn as fact; the synthetic showed ~4.5% because
   per-block checkpointing doesn't touch the dominant (cross-sweep cotangent) consumer. Real-27B
   benefit remains unknown.
3. **"Chunk 64 halves memory."** Measured on twin items: seq 78 @ chunk 128 = **165.8 GB**;
   seq 77 @ chunk 64 = **161.1 GB** — a **2.8% reduction, not 50%**. **Chunk is free on speed but
   is not a memory lever.**
4. **"Chunk 64 fixes the long-item OOM."** Follows from (3) being wrong — it very likely does
   not; whether the long item OOMs depends on the peak-vs-seq slope (M1), which is what the
   store is now measuring.

The through-line: the peak is dominated by something **chunk-independent** we do not yet
understand, and reasoning about MLX's caching allocator from tensor arithmetic has been
unreliable. The discipline going forward: **the store measures; we do not predict.** M2
(real memory instrumentation) is the honest fix for the mispredictions.

### The resolution (measured 2026-07-12)

The peak was never really understood until we measured the metric none of the four
predictions above looked at: RSS. The SIGKILLs killing the fit at item transitions
(exit code 137) were not a per-item OOM and not a chunk-size problem — they were the
macOS memory-pressure / jetsam killer. MLX's caching allocator holds freed buffers and
never returns them to the OS; `mx.reset_peak_memory()` resets the peak *counter*, not
the pool, so the process's RSS stays pinned at the run's max-item high-water (~161GB on
the 27B band) for its entire life, even while fitting a tiny item. A fresh process
fitting a small item measured ~27GB RSS (just the weights), confirming the persistent
~161GB figure was the cache pool, not the item's working set. Chunk 128→64 gave only a
2.8% peak reduction (165.8GB vs 161.1GB on twin items) — confirming (3) above was
measuring the wrong lever: chunk is free on speed but does not touch the pool.

The fix: `mx.clear_cache()` between items in `fit_corpus` (fit.py), which actually frees
the pool and drops RSS to ~27GB between items, at negligible cost (re-allocation is
sub-second against a ~40-min item). Residual risk: during a big item's compute the
process still holds ~161GB and remains vulnerable to external memory pressure.

This is the fifth data point in the same discipline as the four mispredictions above:
the store — and here, an RSS monitor — measures; we do not predict.

### Item 10 dropped — likely unnecessary; the drop model was wrong (corrected 2026-07-12)

This is the sixth memory correction in this session's record, in the same discipline as
the five before it: **measure, don't predict.**

`clear_cache()` fixes the transition-pressure SIGKILL, but the original item-10 decision
also leaned on M1's now-corrected seq-length slope (~1.7GB/token, ~29GB intercept). Under
that model, item 10 (seq 126) extrapolated to ~245GB — over the 192GB ceiling — so it was
dropped. Under the corrected positions model (§3: ~63GB base + ~2.1GB/fitted-position),
item 10 has 47 fitted positions — the same count as the measured 161-166GB cluster above
— so its actual peak was probably **~163GB**, which likely fits under 192GB. **The drop
was likely unnecessary: a safe-but-probably-wrong call, made under the wrong
(sequence-length) model.** This is not a vindication of the original decision — dropping
was reasonable given what was measured at the time, and it did not cost correctness (the
lens still fit on the remaining 11 items) — but it means the corpus probably could have
kept all 12.

It also means `JLENS_MAX_FIT_SEQ` (the workaround, jlens commit `073cc04`) guards the
wrong quantity: it caps SEQUENCE length, but memory tracks fitted POSITIONS. The
memory-correct lever is the fitted-position count, which for on-policy items is capped by
`on_policy_max_tokens` (~47 in this corpus). In this corpus the max position count is 56
(item 11) -> predicted ~181GB by the positions model, still under the 192GB ceiling -> by
the positions model, **no item in this corpus actually needed dropping**.
`JLENS_MAX_FIT_SEQ` should be understood as a coarse safety net on a
correlated-but-not-causal quantity, not the memory-correct control.

Item 11 (56 fitted positions) is a pending empirical test of the corrected model —
predicted peak ~181GB (63 + 2.1x56) when it runs. Noted here as confirmation still owed,
not yet observed.

**Both of tonight's original fixes are still workarounds, not root fixes — and one of
them was tuned against the wrong axis.** `mx.clear_cache()` avoids the transition
SIGKILL and remains correct. `JLENS_MAX_FIT_SEQ` avoids the single-item OOM by dropping
items over a sequence-length cap, but that cap is only a proxy for the real driver
(fitted positions), so it can drop items — like item 10 — that would probably have fit.
Neither explains or reduces the underlying ~2.1GB/position, ~161GB-at-47-positions
footprint — that is M2 and M3 above, and they are the deeper fix: M2 (instrument the
memory, find where it goes) is the prerequisite; M3 (bench `feat/checkpoint` at real
scale) is the lever that could let high-position items fit instead of being dropped.
Neither is done — both are parked as the next session's priority, in that order.

## 5. Build order (free wins first)

1. **Done** — `fact_item_fit` + parser (`fit_metrics.py`) and the dashboard
   (`fit_metrics_ui.py`). M1 becomes a query as items land.
2. **Next** — wire the fit to emit metrics rows directly (not post-hoc parsing) + add the coarse
   phase-level memory reads (M2), so runs are self-instrumenting.
3. **Then** — fold the fidelity / disposition metrics + readout tokens into a `dim_fit_quality`,
   which unlocks the real questions (Q1, Q3).
