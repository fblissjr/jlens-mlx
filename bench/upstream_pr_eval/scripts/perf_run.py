"""Single perf+memory config runner (fresh process per run).

Usage: perf_run.py <section> <impl> <T>
  section D impls: chunked | sequential   (PR #1389, ops-level fwd+bwd vjp)
  section G impls: metal | python         (PR #1217, update-level fwd+bwd vjp)
Realistic 27B GDN head shapes, B=1, fp32.
Prints: RESULT section=<s> impl=<i> T=<T> fwd_bwd_s=<t> peak_gb=<m>
"""
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REAL, make_inputs

section, impl, T = sys.argv[1], sys.argv[2], int(sys.argv[3])
B = 1
dims = REAL
inp = make_inputs(B, T, dims["Hk"], dims["Hv"], dims["Dk"], dims["Dv"],
                  dtype=mx.float32, seed=42)
st = mx.zeros((B, dims["Hv"], dims["Dv"], dims["Dk"]), dtype=mx.float32)
dy = inp["dy"]
cot_state = mx.zeros_like(st)

if section == "D":
    from mlx_lm.models.gated_delta import gated_delta_ops, gated_delta_ops_chunked
    ops = gated_delta_ops_chunked if impl == "chunked" else gated_delta_ops
    primals = [inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"]]

    def f(q, k, v, g, beta):
        return ops(q, k, v, g, beta, st, None)

    def run():
        _, grads = mx.vjp(f, primals, [dy, cot_state])
        mx.eval(*grads)
elif section == "G":
    if impl == "metal":
        from mlx_lm.models.gated_delta_vjp_metal import (
            gated_delta_update_vjp_metal as upd,
        )
    else:
        from mlx_lm.models.gated_delta_vjp import gated_delta_update_vjp as upd
    primals = [inp["q"], inp["k"], inp["v"], inp["a"], inp["b"]]

    def f(q, k, v, a, b):
        return upd(q, k, v, a, b, inp["A_log"], inp["dt_bias"], st, None)

    def run():
        _, grads = mx.vjp(f, primals, [dy, cot_state])
        mx.eval(*grads)
else:
    raise SystemExit("bad section")

# Warmup (also builds/compiles kernels), then timed runs.
t0 = time.perf_counter()
run()
warm = time.perf_counter() - t0
iters = 1 if warm > 20 else 2
mx.reset_peak_memory()
t0 = time.perf_counter()
for _ in range(iters):
    run()
dt = (time.perf_counter() - t0) / iters
print(f"RESULT section={section} impl={impl} T={T} fwd_bwd_s={dt:.3f} "
      f"peak_gb={mx.get_peak_memory() / 2**30:.2f} warmup_s={warm:.1f}", flush=True)
