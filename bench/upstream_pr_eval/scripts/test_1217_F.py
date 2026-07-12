"""PR #1217 section F: their Metal VJP vs their Python chunked VJP vs OUR kernel.

All three arms differentiate w.r.t. (q, k, v, a, b) so the gate-parameter
chain rule (a -> g, b -> beta) is handled identically by autodiff. Zero initial
state, cotangent on y only (zero on final state) -- matching our kernel's
zero-state-grad semantics.
"""
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REAL, load_jlens_gdn, make_inputs, stats

from mlx_lm.models.gated_delta import compute_g, gated_delta_ops
from mlx_lm.models.gated_delta_vjp import gated_delta_update_vjp
from mlx_lm.models.gated_delta_vjp_metal import gated_delta_update_vjp_metal

print("mlx", mx.__version__)
jl = load_jlens_gdn()

cases = [("smallGQA", 64, dict(Hk=2, Hv=4, Dk=32, Dv=32)),
         ("real27B", 64, REAL),
         ("real27B", 128, REAL)]

for tag, T, dims in cases:
    B = 1
    inp = make_inputs(B, T, dims["Hk"], dims["Hv"], dims["Dk"], dims["Dv"],
                      dtype=mx.float32, seed=5000 + T + dims["Hk"])
    st = mx.zeros((B, dims["Hv"], dims["Dv"], dims["Dk"]), dtype=mx.float32)
    dy = inp["dy"]
    cot_state = mx.zeros_like(st)
    A_log, dt_bias = inp["A_log"], inp["dt_bias"]
    primals = [inp["q"], inp["k"], inp["v"], inp["a"], inp["b"]]

    def f_metal(q, k, v, a, b):
        return gated_delta_update_vjp_metal(q, k, v, a, b, A_log, dt_bias, st, None)

    def f_python(q, k, v, a, b):
        return gated_delta_update_vjp(q, k, v, a, b, A_log, dt_bias, st, None)

    def f_ours(q, k, v, a, b):
        beta = mx.sigmoid(b)
        g = compute_g(A_log, a, dt_bias)
        y = jl._gdn_fit_recurrence(q, k, v, g, beta, st)
        return y, mx.stop_gradient(st)

    def f_refseq(q, k, v, a, b):
        beta = mx.sigmoid(b)
        g = compute_g(A_log, a, dt_bias)
        return gated_delta_ops(q, k, v, g, beta, st, None)

    (y_m, _), g_m = mx.vjp(f_metal, primals, [dy, cot_state])
    (y_p, _), g_p = mx.vjp(f_python, primals, [dy, cot_state])
    (y_o, _), g_o = mx.vjp(f_ours, primals, [dy, cot_state])
    (y_r, _), g_r = mx.vjp(f_refseq, primals, [dy, cot_state])
    mx.eval(y_m, y_p, y_o, y_r, *g_m, *g_p, *g_o, *g_r)

    print(f"\n{tag} T={T}  (ref = sequential gated_delta_ops autodiff)")
    fy_m, fy_p, fy_o = stats(y_m, y_r), stats(y_p, y_r), stats(y_o, y_r)
    print(f"   y     metal[rel={fy_m[1]:.2e}] python[rel={fy_p[1]:.2e}] ours[rel={fy_o[1]:.2e}]")
    for n, m, p, o, r in zip(["dq", "dk", "dv", "da", "db"], g_m, g_p, g_o, g_r):
        sm, sp, so = stats(m, r), stats(p, r), stats(o, r)
        smo = stats(m, o)
        print(f"   {n:3s} metal-vs-ref[rel={sm[1]:.2e} cos={sm[2]:.7f}]  "
              f"python-vs-ref[rel={sp[1]:.2e} cos={sp[2]:.7f}]  "
              f"ours-vs-ref[rel={so[1]:.2e} cos={so[2]:.7f}]  "
              f"metal-vs-ours[rel={smo[1]:.2e}]")

print("\nDONE F")
