"""Instrumented 2-iter LoRA run: counts which VJP path pr-1217 dispatches to."""
import os
import sys

import mlx_lm.models.gated_delta_vjp as gv
import mlx_lm.models.gated_delta_vjp_metal as gvm

counts = {"metal": 0, "python": 0}
_orig_metal = gvm.gated_delta_update_vjp_metal
_orig_py = gv.gated_delta_update_vjp


def wrap_metal(*a, **k):
    counts["metal"] += 1
    return _orig_metal(*a, **k)


def wrap_py(*a, **k):
    counts["python"] += 1
    return _orig_py(*a, **k)


gvm.gated_delta_update_vjp_metal = wrap_metal
gv.gated_delta_update_vjp = wrap_py

from mlx_lm import lora

model_dir = os.environ.get("JLENS_EVAL_MODEL")
if not model_dir:
    raise SystemExit(
        "JLENS_EVAL_MODEL is not set. Point it at a local Qwen3.5-27B GDN "
        "model directory (e.g. a Qwen3.5-27B heretic 8-bit MLX checkpoint) "
        "before running this probe -- see bench/upstream_pr_eval/README.md."
    )

sys.argv = ["lora", "--model", model_dir,
            "--train", "--data", "data_lora", "--iters", "2", "--batch-size", "1",
            "--max-seq-length", "512", "--steps-per-report", "1",
            "--steps-per-eval", "100", "--val-batches", "1",
            "--adapter-path", "adapters_probe", "--seed", "7"]
lora.main()
print(f"\nPATH COUNTS: metal={counts['metal']} python={counts['python']}")
