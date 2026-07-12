# mlx-lm PR #1389 / #1217 evaluation harness

Last updated: 2026-07-12

This directory is a self-contained port of the scripts and raw logs used to
empirically evaluate two upstream `mlx-lm` pull requests against real
Qwen3.5-27B (GDN-hybrid) shapes, and against jlens-mlx's own ported GDN
backward kernel (`jlens_mlx/providers/qwen3_5_gdn.py`, built for narrow-scope
interpretability fitting -- T<=128, scalar gating only).

## What was evaluated

- **PR #1389 -- "chunk-parallel gated delta ops"**: a chunk-parallel,
  differentiable reformulation of the gated-delta-rule ops used by
  `qwen3_5`/`qwen3_next` GDN layers (forward + backward), as an alternative
  to the existing sequential-ops implementation.
- **PR #1217 -- "Metal VJP kernel for `gated_delta_update`"**: a custom Metal
  backward kernel for the fused `gated_delta_update` op, plus a Python VJP
  fallback for non-Metal backends.

Both were compared against `mlx-lm` `main` (sequential ops, autodiff) and
against jlens-mlx's own kernel, for both forward/gradient correctness and
performance (per-layer micro-bench + end-to-end LoRA + generation).

## Provenance

| | |
|---|---|
| Date measured | 2026-07-12 |
| Hardware | Mac Studio M2 Ultra, 192 GB unified memory |
| `mlx` | 0.32.0 (PyPI) |
| `mlx-lm` baseline (`main`) | `15b522f` |
| `mlx-lm` PR #1389 branch | `6fc3a29` |
| `mlx-lm` PR #1217 branch | `29706ad` |
| Model | local Qwen3.5-27B "heretic" 8-bit MLX checkpoint (GDN dims: `Hk=16, Hv=48, Dk=Dv=128`, scalar gating) |

Each `mlx-lm` branch was installed editable into its own `uv` venv (`main`,
`pr-1389`, `pr-1217`) so results are directly comparable without any other
dependency drift.

## Results

### LoRA end-to-end (`mlx_lm.lora`, 27B 8-bit, batch 1, rank 8, 16 layers, seed 7)

| run | branch | tok/s | trainer peak | process peak | loss first->last | val first->last |
|---|---|---|---|---|---|---|
| short (~80 tok) | #1389 | 92-103 | 30.0 GB | -- | 2.556->0.133 | 2.691->0.181 |
| short | #1217 | 98-110 | 30.5 GB | -- | 2.557->0.133 | 2.691->0.181 |
| short | main | 55-62 | -- | -- | 2.564->0.132 | 2.691->0.180 |
| long (~670 tok) | #1389 | 143-145 | 35.9 GB | 37.9 GB | 1.442->0.839 | 1.444->0.859 |
| long | #1217 | 150-153 | 36.2 GB | 39.4 GB | 1.451->0.840 | 1.444->0.861 |
| long | main | ~50 avg | -- | 117.5 GB | 1.459->0.839 | 1.444->0.858 |

Identical val-loss trajectories across branches serve as an end-to-end
gradient cross-validation between the sequential and chunked/Metal paths.

### Inference non-regression (450-tok prompt, 200 gen tokens, greedy)

| branch | prompt tps | gen tps | peak |
|---|---|---|---|
| main | 180.4 | 21.62 | 29.652 GB |
| #1389 | 179.4 | 21.59 | 29.652 GB |
| #1217 | 180.4 | 21.64 | 29.652 GB |

Neither PR touches the generation path measurably -- expected, since both are
training-time (forward+backward) changes.

### Kernel micro-bench (single layer, 27B shape, B=1 fp32, fwd+bwd)

| T | #1389 chunked | sequential ops | #1217 Metal VJP | #1217 Python VJP | ours (jlens) |
|---|---|---|---|---|---|
| 128 | 6 ms / 0.26 GB | 34 ms / 1.36 GB | 3 ms / 0.63 GB | 44 ms / 1.82 GB | 3.1 ms / 0.02 GB |
| 512 | 21 ms / 0.57 GB | 203 ms / 4.28 GB | 12 ms / 2.06 GB | 181 ms / 4.63 GB | n/a (T cap) |
| 2048 | 91 ms / 2.16 GB | 2294 ms / 27.5 GB | 73 ms / 4.67 GB | 783 ms / 14.0 GB | n/a |

jlens-mlx's own kernel is narrow-scope (T<=128 by design, built for
interpretability fitting, not general training) so it has no data point past
T=128.

### Correctness summary

- Three-way parameter-level gradients (their Metal, their Python, ours) vs
  sequential autodiff: rel <= 2.6e-7, cos 1.0000000, including the real-27B
  GQA shape.
- PR #1389 forward: exact reformulation (fp32 rel <= 7.2e-6; bf16 differences
  are the same magnitude as the fused kernel's own rounding).
- PR #1389's raw `dg` (gate-cotangent) comparison vs sequential autodiff can
  look badly divergent at near-saturated gates (cos as low as 0.24 observed
  on one shape). This is fp32 conditioning of the log-domain backward
  (1/g amplification near g -> 0), not a bug -- it cancels exactly through
  `compute_g` at the `a`/`b` parameter leaves, which is what training
  actually uses (`da <= 1.8e-5`, `db <= 4.2e-6`). Worth a code comment
  upstream for future validators.
- Both PRs' own test suites pass locally (#1217: 5/5; #1389: 4/4).
- PR #1217 adds a `training=` kwarg passed unconditionally to
  `gated_delta_update` in `qwen3_5.py`/`qwen3_next.py`. This breaks
  old-signature monkey-patches; jlens-mlx's own `gdn_fit_patch` was updated
  defensively to absorb it (jlens-mlx commit `951dd76`).
- An instrumented LoRA run (`scripts/probe_1217_path.py`) confirmed all 96
  GDN layer calls took the Metal path on #1217 (the `Dk%32`/`Dv%4` dispatch
  gate is satisfied at the 27B's shapes).

## Directory layout

```
bench/upstream_pr_eval/
  README.md          -- this file
  scripts/            -- portable copies of the eval scripts (see below)
  logs/               -- raw stdout captures from the 2026-07-12 run
```

## Scripts

| script | what it does |
|---|---|
| `common.py` | shared helpers: synthetic GDN input generation, comparison stats (`max_abs`/`rel`/`cos`), timing, and the jlens kernel loader |
| `test_1389_ABC.py` | PR #1389 forward exactness (A), gradient correctness vs sequential (B), three-way vs jlens kernel (C) |
| `test_1389_B2_dg.py` | follow-up: isolates the raw-`dg` saturated-gate divergence and shows it cancels at the `a`/`b` parameter level |
| `test_1217_F.py` | PR #1217 Metal VJP vs Python VJP vs jlens kernel vs sequential autodiff |
| `perf_run.py` | single-config timed fwd+bwd micro-bench (one fresh process per config, invoked by `perf_driver.py`) |
| `perf_driver.py` | drives `perf_run.py` across a matrix of `(impl, T)` in fresh subprocesses with a per-run timeout |
| `probe_1217_path.py` | instrumented 2-iteration LoRA run that counts which VJP path (`metal` vs `python`) PR #1217 actually dispatches to |

These are byte-identical to the scripts that produced `logs/` except for
path/bootstrap plumbing: the originals had a scratch-workspace `sys.path`
entry and a hardcoded absolute path to `qwen3_5_gdn.py` and to a local model
checkpoint. Here:

- `common.py` locates the repo root via
  `Path(__file__).resolve().parents[3]` and loads
  `jlens_mlx/providers/qwen3_5_gdn.py` directly from that path (bypassing the
  `jlens_mlx` package `__init__`, same as the original -- this only needs the
  file to exist on disk, not the package to be installed).
- The other scripts that imported `common` add `scripts/`'s own directory to
  `sys.path` via `Path(__file__).resolve().parent` instead of a hardcoded
  scratch path.
- `probe_1217_path.py` (the only script that loads a real model) reads the
  model directory from the `JLENS_EVAL_MODEL` environment variable and fails
  fast with a clear message if it isn't set. There is no default containing
  any local path.

No numerics logic was changed anywhere in the port.

## Reproduction

Prerequisites:

- Three `uv`-managed clones of `mlx-lm` at the SHAs in the provenance table
  above, each with its own venv:

  ```
  git clone https://github.com/ml-explore/mlx-lm mlx-lm-main
  git -C mlx-lm-main checkout 15b522f

  git clone https://github.com/ml-explore/mlx-lm mlx-lm-pr1389
  git -C mlx-lm-pr1389 fetch origin pull/1389/head:pr-1389
  git -C mlx-lm-pr1389 checkout pr-1389   # resolves to 6fc3a29 as of this run

  git clone https://github.com/ml-explore/mlx-lm mlx-lm-pr1217
  git -C mlx-lm-pr1217 fetch origin pull/1217/head:pr-1217
  git -C mlx-lm-pr1217 checkout pr-1217   # resolves to 29706ad as of this run

  uv venv .venv-main    && uv pip install --python .venv-main    -e mlx-lm-main    "mlx==0.32.0"
  uv venv .venv-pr1389  && uv pip install --python .venv-pr1389  -e mlx-lm-pr1389  "mlx==0.32.0"
  uv venv .venv-pr1217  && uv pip install --python .venv-pr1217  -e mlx-lm-pr1217  "mlx==0.32.0"
  ```

- A local Qwen3.5-27B GDN model directory (only needed for
  `probe_1217_path.py`), exported as `JLENS_EVAL_MODEL`.
- A small LoRA fine-tune dataset directory named `data_lora` (train/valid
  JSONL in the `mlx_lm.lora` format) in the working directory --
  `probe_1217_path.py` references it by that relative name. Not shipped
  here; any small instruction-tuning JSONL set is sufficient for the
  path-dispatch smoke test (the point of that script is which VJP path is
  taken, not training quality).

Run order (each script is run under the venv for the branch it needs --
`gated_delta_ops_chunked` only exists on the #1389 branch,
`gated_delta_update_vjp_metal`/`gated_delta_update_vjp` only on the #1217
branch):

```
# Correctness -- PR #1389 (run under .venv-pr1389, jlens-mlx repo root as cwd)
.venv-pr1389/bin/python bench/upstream_pr_eval/scripts/test_1389_ABC.py
.venv-pr1389/bin/python bench/upstream_pr_eval/scripts/test_1389_B2_dg.py

# Correctness -- PR #1217 (run under .venv-pr1217)
.venv-pr1217/bin/python bench/upstream_pr_eval/scripts/test_1217_F.py

# Path-dispatch probe -- PR #1217 (run under .venv-pr1217, from a scratch
# working directory containing data_lora/)
JLENS_EVAL_MODEL=/path/to/Qwen3.5-27B-model \
  .venv-pr1217/bin/python bench/upstream_pr_eval/scripts/probe_1217_path.py

# Micro-bench -- section D is PR #1389 (chunked vs sequential ops),
# section G is PR #1217 (metal vs python update VJP). perf_run.py is
# invoked as a subprocess per config so peak-memory measurements don't
# leak between configs; perf_driver.py automates that.
.venv-pr1389/bin/python bench/upstream_pr_eval/scripts/perf_driver.py \
  .venv-pr1389/bin/python D chunked,sequential 128,512,2048
.venv-pr1217/bin/python bench/upstream_pr_eval/scripts/perf_driver.py \
  .venv-pr1217/bin/python G metal,python 128,512,2048

# End-to-end LoRA + generation comparisons (main / #1389 / #1217): run
# mlx_lm.lora and mlx_lm.generate directly under each branch's venv against
# the same model/data/seed. These were driven by ad hoc CLI invocations
# rather than a checked-in script; see logs/out_lora_*.txt and
# logs/out_generate.txt for the exact settings each run used (visible in
# each log's own banner/args).
```

## Raw logs

`logs/` contains the raw stdout captures from the actual 2026-07-12 run
described above -- these are the real outputs the results tables were
compiled from, not illustrative examples. The only edit made to them was
scrubbing: every absolute local filesystem path (this machine's home
directory, model checkpoint path) was replaced with a neutral placeholder
(`<MODEL_DIR>/...`). Numbers, tables, and tracebacks are otherwise untouched.

| log | corresponds to |
|---|---|
| `out_ABC.txt` | `test_1389_ABC.py` |
| `out_B2.txt` | `test_1389_B2_dg.py` |
| `out_F.txt` | `test_1217_F.py` |
| `out_D.txt` | micro-bench section D (PR #1389, `perf_driver.py`) |
| `out_G.txt` | micro-bench section G (PR #1217, `perf_driver.py`) |
| `out_generate.txt` | inference non-regression check (main / #1389 / #1217) |
| `out_lora_main.txt`, `out_lora_main_long.txt` | LoRA end-to-end on `main` (short / long) |
| `out_lora_1389.txt`, `out_lora_1389_long.txt` | LoRA end-to-end on PR #1389 (short / long) |
| `out_lora_1217.txt`, `out_lora_1217_long.txt` | LoRA end-to-end on PR #1217 (short / long) |
