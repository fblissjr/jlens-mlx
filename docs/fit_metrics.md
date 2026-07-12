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
- Views: `v_peak_vs_seq`, `v_throughput`.

The DB is stable and shared across runs (`out/fit_metrics.duckdb`, gitignored), so every fit
accumulates rows and cross-run analysis becomes a query.

## 2. Usage

Runs under the **heylook venv** (duckdb lives there, not in the jlens venv); pure stdlib +
duckdb, no MLX.

```sh
# ingest a run's completed items (idempotent; re-run as more items finish)
cd <heylook repo> && uv run python <jlens>/scripts/fit_metrics.py --out out/band-n14-fixed

# print a view without ingesting
uv run python <jlens>/scripts/fit_metrics.py --query peak_vs_seq
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
- **M1 — how does peak scale with sequence length?** Data: `(seq_len, peak_gb)` per item —
  **being collected** (`v_peak_vs_seq`). Difficulty: **trivial** once items span a range of
  lengths. This is the load-bearing question: it decides whether a given corpus's longest item
  OOMs.
- **M2 — what is the true internal breakdown of the peak?** Data: `mx.get_active_memory` /
  `get_cache_memory` sampled around sweep phases. Not collected. Difficulty: medium (coarse
  split cheap; per-tensor hard). Our reconstructions have been unreliable — this needs real
  instrumentation.
- **M3 — does checkpointing help at real scale, and where does the freed memory live?** Data:
  the `feat/checkpoint` branch run on the real 27B + phase-level memory reads. Not collected.
  Difficulty: easy-ish but GPU-gated.

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

## 5. Build order (free wins first)

1. **Done** — `fact_item_fit` + parser (`fit_metrics.py`) and the dashboard
   (`fit_metrics_ui.py`). M1 becomes a query as items land.
2. **Next** — wire the fit to emit metrics rows directly (not post-hoc parsing) + add the coarse
   phase-level memory reads (M2), so runs are self-instrumenting.
3. **Then** — fold the fidelity / disposition metrics + readout tokens into a `dim_fit_quality`,
   which unlocks the real questions (Q1, Q3).
