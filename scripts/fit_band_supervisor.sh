#!/usr/bin/env bash
# Restart-loop supervisor for scripts/fit_band_corpus.py -- see its module docstring for the
# full exit-code contract. A long/overnight band fit can die to a native MLX crash (SIGABRT from
# an uncaught C++ exception, SIGSEGV at type teardown) with no Python traceback -- this wraps the
# fit in a restart loop so that's a 15s hiccup instead of a lost run. Per-item checkpointing
# (fit.py's checkpoint_dir) makes each restart cheap: it resumes at the next un-fit item.
#
# Usage (run FROM THE REPO -- no absolute paths baked in):
#   ./scripts/fit_band_supervisor.sh
#   JLENS_MAX_RESTARTS=5 JLENS_MODEL=/path/to/model ./scripts/fit_band_supervisor.sh
#   nohup ./scripts/fit_band_supervisor.sh > out/band-fit.log 2>&1 &
#
# Env:
#   JLENS_MAX_RESTARTS  max restart attempts after an unexpected exit (default 20)
#   JLENS_FIT_CMD        override the command run each attempt (testability hook -- swap in a
#                        stub command instead of the real fit; default
#                        "uv run python scripts/fit_band_corpus.py")
#   (all other JLENS_* / model env vars pass through untouched to the fit command)
#
# Exit-code handling (see scripts/fit_band_corpus.py's docstring for the full contract):
#   0  -> fit completed successfully. Stop.
#   3  -> corpus diversity gate failed (degenerate corpus). Stop -- retrying won't fix the corpus.
#   2  -> config error (e.g. missing JLENS_MODEL). Stop -- retrying won't fix a config error.
#   *  -> unexpected (native crash / killed by signal / anything else). Log + restart after 15s,
#         up to JLENS_MAX_RESTARTS; exceeding that gives up with the last seen exit code.
set -u

max_restarts="${JLENS_MAX_RESTARTS:-20}"
fit_cmd="${JLENS_FIT_CMD:-uv run python scripts/fit_band_corpus.py}"

ts() { date +"%H:%M:%S"; }

restart=0
while true; do
    # Intentionally unquoted/word-split: fit_cmd is a command line, not a single token.
    # shellcheck disable=SC2086
    $fit_cmd "$@"
    code=$?

    if [ "$code" -eq 0 ]; then
        echo "[$(ts)] fit_band_supervisor: fit completed (exit 0) -- done"
        exit 0
    fi
    if [ "$code" -eq 3 ]; then
        echo "[$(ts)] fit_band_supervisor: corpus diversity gate failed (exit 3) -- not retrying"
        exit 3
    fi
    if [ "$code" -eq 2 ]; then
        echo "[$(ts)] fit_band_supervisor: config error (exit 2) -- not retrying"
        exit 2
    fi

    restart=$((restart + 1))
    if [ "$restart" -gt "$max_restarts" ]; then
        echo "[$(ts)] fit_band_supervisor: fit died with code $code, restart $restart/$max_restarts exceeds JLENS_MAX_RESTARTS -- giving up"
        exit "$code"
    fi
    echo "[$(ts)] fit died with code $code, restart $restart/$max_restarts in 15s"
    sleep 15
done
