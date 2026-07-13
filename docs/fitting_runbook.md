# Fitting runbook — band-lens own-fit

Last updated: 2026-07-12

Operational guide for running, watching, validating, and debugging a Jacobian-lens
band fit on the served abliterated Qwen3.5-27B. For the *why* (the math, the design,
the reference lineage) see [DESIGN.md](./DESIGN.md); this doc is the *how*.

The one-sentence model: `fit_band_corpus.py` builds a small on-policy corpus, runs a
hard **diversity gate** on it, then fits `J_l` over the product band (layers 16–47)
one corpus item at a time with per-item checkpointing. A supervisor restarts it
across native crashes; a monitor and a `progress.json` sidecar give live visibility.

---

## 0. Preflight — run BEFORE a long fit

These are cheap and catch the two failure classes that have actually bitten us
(a mismatched capture path; a degenerate corpus).

```sh
# (a) capture parity: the fit reads residuals cache-less, the server applies them
#     with a fresh cache. This proves they're numerically identical on this model.
JLENS_MODEL=/path/to/Qwen3.5-27B-model \
  uv run python scripts/check_capture_parity.py
# expect: CAPTURE PARITY OK ... worst rel_err ~0.0
```

Stop the **heylook server** if it is running — on-policy generation and the fit both
want the GPU, and `max_loaded_models=1` means they will fight. (The server is not
started automatically; only stop it if you started it.)

---

## 1. Launch

Always launch through the **supervisor** for anything longer than a smoke — a native
MLX crash (SIGABRT/SIGSEGV, no Python traceback) becomes a 15s restart instead of a
lost overnight run, because each item is checkpointed.

```sh
PYTHONFAULTHANDLER=1 \
JLENS_MODEL=/path/to/Qwen3.5-27B-model \
JLENS_LAYERS="$(echo {16..47} | tr ' ' ',')" \
JLENS_N=14 JLENS_MAX_SEQ_LEN=128 JLENS_CHUNK=128 \
JLENS_OUT=out/band-n14-fixed \
nohup caffeinate -i ./scripts/fit_band_supervisor.sh \
  > out/band-n14-fixed.log 2>&1 &
disown
```

- `caffeinate -i` keeps the Mac awake for the run; `nohup … & disown` survives the
  shell closing.
- Point `JLENS_OUT` at a **fresh** dir per run. If the dir already has a
  `ckpt/corpus.json`, the fit RESUMES from it (skips corpus build + completed items)
  — intended for restarts, surprising if you meant a clean run.

### Env vars

| var | default | meaning |
|---|---|---|
| `JLENS_MODEL` | *(required)* | absolute path to the model dir. Relative paths crash (`HFValidationError` — mlx-lm reads them as HF repo ids). |
| `JLENS_LAYERS` | shallow sample | comma list of source layers. Band = `16..47`. |
| `JLENS_N` | 6 | corpus prompt count (before length-drops). |
| `JLENS_MAX_SEQ_LEN` | 512 | truncation; **128** keeps every item on the fast GDN kernel (`MAX_T=128`). |
| `JLENS_CHUNK` | 128 | cotangent dim-batch size (saturates ~64; 128 is safe). NOT a memory
lever (measured 2.8% peak reduction 128->64 on twin items) — speed only. |
| `JLENS_MAX_FIT_SEQ` | unset (no cap) | skip any corpus item whose sequence length exceeds
this, advancing the checkpoint past it (the lens averages over the rest). A coarse memory
safety net for items intrinsically too big to fit — but peak scales with fitted POSITIONS,
not sequence length (see §5); the memory-correct lever is fitted-position count /
`JLENS_ONPOLICY_TOKENS`. |
| `JLENS_ONPOLICY_TOKENS` | 48 | tokens generated per on-policy completion. |
| `JLENS_OUT` | `out/band-<N>L` | run dir (checkpoint + log live here). |
| `JLENS_ALLOW_DEGENERATE` | unset | set `=1` to override the diversity gate (only after reading the decode). |
| `JLENS_FINALIZE` | unset | finalize a lens from an existing checkpoint without fitting the remaining items. |
| `JLENS_WATCHDOG_S` | 1800 | hang watchdog: dumps thread stacks if no chunk progress for this many seconds (0 disables). |
| `JLENS_MAX_RESTARTS` | 20 | supervisor: max restarts on unexpected exits. |
| `JLENS_FIT_CMD` | `uv run python scripts/fit_band_corpus.py` | supervisor: command to run each attempt (testability hook). |

---

## 2. Validate the corpus BEFORE it fits (the most important check)

The corpus builds in the first ~1 minute, *before* any GPU-hours are spent. The
diversity gate hard-stops (exit 3) a degenerate corpus automatically, but **read the
decode yourself** — the gate is a number, your eyes are the judgment.

```sh
# the number (top of the log):
grep 'diversity:' out/band-n14-fixed.log
# expect: shared_fraction well under 0.35. (0.598 was the degenerate run; 0.07 is healthy.)

# the actual completions — read them:
open out/band-n14-fixed/ckpt/corpus_decoded.md
```

What healthy looks like: distinct, content-bearing completions per item, and a live
harmful↔benign contrast (the illegal-opioid item refuses-and-explains; the plain
painkiller item advises). What degenerate looks like: the same boilerplate opener
("Here's a thinking process…") repeated across on-policy items — that was the
`enable_thinking` bug; the gate now catches it.

The gate checks **both** the overall shared-fraction and the on-policy sub-arm
(off-policy items can dilute on-policy degeneracy out of the average).

---

## 3. Monitor while it runs

Three ways, cheapest first.

```sh
# is it alive?
pgrep -fl fit_band_corpus        # the python process
pgrep -fl fit_band_supervisor    # the restart wrapper
# both present = running. fit gone but supervisor present = mid-restart (recovery working).

# watch it move — a new chunk line every ~60-70s:
tail -f out/band-n14-fixed.log
#   [HH:MM:SS]   item 1/12 chunk 3/40 (8%) item_elapsed=198s

# one-glance machine-readable status (rewritten every heartbeat):
cat out/band-n14-fixed/ckpt/progress.json | python3 -m json.tool
```

`progress.json` fields: `item`/`n_items`, `chunk`/`n_chunks`, `positions_done`/`_total`,
`sec_per_pos`, `eta_s`, `peak_gb`, `ts`. `sec_per_pos`/`eta_s` are **null until the
first item completes** — that is the honest "no rate yet" state, not a bug. The ETA is
positions-weighted (items vary in cost with their position count), so it does not
climb the way the old per-item-mean estimate did.

### The monitor UI (read-only, no GPU, stdlib only)

```sh
uv run python scripts/fit_monitor.py --out out/band-n14-fixed   # then open the printed localhost URL
```

Two tabs: **Run** (progress, ETA, peak memory, stall indicator, log tail) and
**Corpus** (every item's prompt + the model's completion, stratum, on/off-policy,
token + position counts, grouped so the harmful/benign contrast is adjacent, with the
diversity numbers up top). It only reads the sidecar files — it cannot perturb the fit.

### Metrics store + analytics dashboard

Two more tools, both documented in full in [fit_metrics.md](./fit_metrics.md). Both need
the **heylook venv** (duckdb lives there, not the jlens venv) — run them from the heylook
repo:

```sh
# ingest a run's completed items into the shared DuckDB store (idempotent; re-run as
# more items finish)
cd <heylook repo> && uv run python <jlens>/scripts/fit_metrics.py --out out/band-n14-fixed

# print a view without ingesting
uv run python <jlens>/scripts/fit_metrics.py --query peak_vs_seq
# peak vs. fitted positions -- the axis that actually drives peak memory (fit_metrics.md §3/§4)
uv run python <jlens>/scripts/fit_metrics.py --query peak_vs_positions
uv run python <jlens>/scripts/fit_metrics.py --query throughput

# analytics dashboard over the accumulated store, :8766
uv run python <jlens>/scripts/fit_metrics_ui.py --db <jlens>/out/fit_metrics.duckdb
```

`fit_monitor.py` is live/per-run visibility while a fit is in flight; `fit_metrics.py` +
`fit_metrics_ui.py` are the durable, cross-run analytical record.

### Is it progressing vs spinning?

```sh
ls -la out/band-n14-fixed/ckpt/            # jsum.safetensors + ckpt.json timestamps climb per item
cat out/band-n14-fixed/ckpt/ckpt.json      # carries the done-count
```

Pace sanity: ~60–70s/chunk × 40 chunks ≈ ~45 min/item; 12 items ≈ overnight. That per-item
time matches the pre-instrumentation runs — same kernel, same math.

---

## 4. Exit codes

| code | meaning | supervisor action |
|---|---|---|
| 0 | fit completed | stop (done) |
| 2 | config error (no `JLENS_MODEL`; `JLENS_FINALIZE` with no checkpoint) | stop (retry won't help) |
| 3 | corpus diversity gate failed (degenerate) | stop (retry won't help — fix the corpus) |
| other / signal | native crash, killed, etc. (137 specifically = SIGKILL, i.e. the macOS memory-pressure killer — see §5) | restart after 15s, up to `JLENS_MAX_RESTARTS` |

---

## 5. Debug — the failure modes we've actually seen

**Silent death, no traceback (native MLX crash).** SIGABRT (uncaught C++ exception)
or SIGSEGV at type teardown. The supervisor auto-restarts these and logs
`fit died with code X, restart n/MAX in 15s`; resume picks up at the next unfit item.
macOS crash reports land in `~/Library/Logs/DiagnosticReports/python3.13-*.ips` — the
faulting thread's frames name the culprit.

**`HFValidationError: Repo id must be in the form …`.** `JLENS_MODEL` was a relative
path; mlx-lm's `load` treats a non-existent-as-given relative path as an HF repo id.
Fix: absolute path. (The driver now resolves it, but pass absolute to be safe.)

**Wedged / no chunk progress.** The watchdog dumps all thread stacks into the log after
`JLENS_WATCHDOG_S` (default 30 min) *without* killing the process, so you get a
stack trace of where it's stuck while it's still alive. The monitor's "last update Xs
ago" turns red past 3 min as an earlier warning.

**Gate tripped (exit 3) but you believe the corpus is fine.** Read
`ckpt/corpus_decoded.md`. If it genuinely is fine (rare), re-launch with
`JLENS_ALLOW_DEGENERATE=1`. Do not do this reflexively — the gate exists because a
degenerate corpus cost 7 GPU-hours once.

**Resume did the wrong thing.** A run "skips build" because `JLENS_OUT` pointed at a
dir with an existing `ckpt/corpus.json`. Use a fresh `JLENS_OUT` for a clean run.

**Exit 137 (SIGKILL) at an item transition — macOS memory-pressure kill, not an
OOM.** The process dies between items with exit code 137 and the supervisor restarts
it (§4). This is a distinct failure mode from the native-crash class above. Root
cause: MLX's caching allocator holds freed buffers and never returns them to the OS.
`mx.reset_peak_memory()` (already called at every item start) resets the peak
*counter*, not the pool — so the process's RSS stays pinned at the run's max-item
high-water (~161GB on the 27B band) for its entire life, even while fitting a tiny
item. On a 192GB machine that leaves ~30GB for the OS, and a transient spike at an
item transition trips the jetsam killer. Confirmed by measuring a fresh process
fitting a tiny item at ~27GB RSS (just the model weights) — the persistent ~161GB
figure was the cache pool, not the item's working set. This behaves like a leak (RSS
pinned high, kills the process) but is technically cache retention (reclaimable,
bounded at the peak) — the distinction is what makes it fixable.

Chunk size is **not** the lever here: dropping `JLENS_CHUNK` 128→64 gave only a 2.8%
peak reduction (165.8GB vs 161.1GB measured on twin items) — chunk helps speed, not
this failure mode.

**Fix**: `mx.clear_cache()` between items in `fit_corpus` (fit.py) actually releases
the pool, dropping RSS to ~27GB between items — the next item re-allocates from a cold
pool in under a second, negligible against a ~40-min item.

**Residual risk**: during a big item's ~40-min compute the process still holds
~161GB and is vulnerable to EXTERNAL memory pressure — avoid running heavy memory
apps alongside a long fit (extra browser tabs, other model loads) while a large item
is in flight. The supervisor already auto-restarts a 137, but a persistent
external-pressure situation could loop.

**A single item is intrinsically too big to fit (still OOMs even with `clear_cache`).**
Distinct from the transition-pressure SIGKILL above: this is not about freeing the pool
BETWEEN items, it's that one item's own compute needs more than the machine has. Peak
scales ~linearly with fitted POSITION count, not sequence length (~63GB base + ~2.1GB per
fitted position — measured on the 27B band, corrected 2026-07-12; the earlier
~1.7GB/token seq-length slope was a confound, see `fit_metrics.md` §3/§4), so an item with
enough fitted positions extrapolates past the 192GB ceiling regardless of `clear_cache`.
Workaround: set `JLENS_MAX_FIT_SEQ` (§1) to skip items over a sequence-length cap — but
`JLENS_MAX_FIT_SEQ` guards sequence length, which is only correlated with (not the cause
of) the peak; the memory-correct lever is fitted-position count, capped for on-policy
items by `JLENS_ONPOLICY_TOKENS`. Treat `JLENS_MAX_FIT_SEQ` as a coarse safety net, not a
memory-correct control — the checkpoint advances past skipped items and the lens averages
over what's left. This is a workaround, not a fix: it drops data rather than making the
item fit, and can drop items unnecessarily since sequence length is an imperfect proxy
for the real driver. The deeper fix (understanding and possibly reducing the per-position
peak so items don't need to be dropped) is tracked as M2/M3 in `fit_metrics.md` §3 — not
done, parked for a future session.

---

## 6. After it finishes

The fit runs a **held-out fidelity gate** at the end (per-layer top-1/top-k/KL vs the
model's true logits) and stamps per-layer scores + corpus/fit provenance on the lens
sidecar. Output lands in `out/<run>/ckpt/` (the lens + `ckpt.json` + provenance).

Caveat that the report [`jspace_jlens_end_to_end.md`, §1] flags: on a quantized +
abliterated model the fidelity gate **misleads** on band layers — it can rank a
degenerate near-target layer above a meaningful mid-band one. Judge band layers
**qualitatively** (the readout tokens), not by final-logit agreement. Per-layer token
readouts (`scripts/readout.py`) need the model + GPU and a finished lens — run them
after the fit, or view the lens on the heylook v3 `jspace` page.

---

## Quick reference

```sh
# preflight
JLENS_MODEL=<abs> uv run python scripts/check_capture_parity.py

# launch (overnight)
PYTHONFAULTHANDLER=1 JLENS_MODEL=<abs> JLENS_LAYERS="$(echo {16..47}|tr ' ' ',')" \
  JLENS_N=14 JLENS_MAX_SEQ_LEN=128 JLENS_OUT=out/<run> \
  nohup caffeinate -i ./scripts/fit_band_supervisor.sh > out/<run>.log 2>&1 & disown

# validate corpus (first minute)
grep diversity: out/<run>.log && open out/<run>/ckpt/corpus_decoded.md

# watch
tail -f out/<run>.log
uv run python scripts/fit_monitor.py --out out/<run>

# is it alive / progressing
pgrep -fl fit_band_corpus ; cat out/<run>/ckpt/progress.json | python3 -m json.tool
```
