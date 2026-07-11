#!/usr/bin/env bash
# Pin the accuracy/speed numbers to artifacts.
#
# The README asserts figures (GDN kernel rel ~3e-7 / cos 1.0, dim-batching rel 2e-7,
# chain == direct cos 1.0) that had no saved verifier output on disk -- so they read
# as BELIEVED, not KNOWN. This re-runs the verifier ladder and tees each script's
# stdout to out/verify/<name>.log, turning the claims into reproducible artifacts.
#
# Metal-gated. Run FROM THE HEYLOOK DIR so `uv run` resolves the working mlx venv and
# the check scripts insert jlens-mlx onto sys.path (jlens-mlx's own venv is broken).
# GPU must be free (heylook server stopped AND no fit running):
#
#     cd <heylook repo>; bash <jlens-mlx>/scripts/capture_verifier_logs.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"          # jlens-mlx/scripts (relative; no absolute paths)
OUT="$HERE/../out/verify"
mkdir -p "$OUT"

# The synthetic/parity ladder. xcheck_*.py (torch cross-check vs Anthropic) is left out
# on purpose -- it needs a torch env, not the point of pinning our own numbers.
CHECKS=(
  check_qwen3_5_synthetic   # GDN kernel vs mx.vjp, forward parity, whole-fit cos, dim-batching, gate
  check_chain_vs_direct     # chain == direct (qwen3_5 synth + gpt2): the exactness gate
  check_gpt2_parity         # gpt2 apply/fit parity
  check_gemma2_synthetic    # gemma2 tail
  check_rmsnorm_seed        # the rms^2-vs-rms^3 seed bug regression guard
)

for chk in "${CHECKS[@]}"; do
  echo "=== $chk ==="
  uv run python "$HERE/$chk.py" 2>&1 | tee "$OUT/$chk.log"
  echo
done
echo "saved verifier logs -> $OUT  (commit them as the evidence behind the README numbers)"
