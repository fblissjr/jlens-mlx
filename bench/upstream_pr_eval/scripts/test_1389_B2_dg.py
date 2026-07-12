"""Follow-up: is chunked dg divergence confined to saturated gates (g < 1e-12
clamp), and does it vanish at the parameter level (da via compute_g chain)?"""
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REAL, make_inputs, stats

from mlx_lm.models.gated_delta import (
    compute_g,
    gated_delta_ops,
    gated_delta_ops_chunked,
)

for tag, T, dims, seed in [("real27B", 128, REAL, 1128), ("smallGQA", 64, dict(Hk=2, Hv=4, Dk=32, Dv=32), 2066)]:
    B = 1
    inp = make_inputs(B, T, dims["Hk"], dims["Hv"], dims["Dk"], dims["Dv"],
                      dtype=mx.float32, seed=seed)
    st = mx.zeros((B, dims["Hv"], dims["Dv"], dims["Dk"]), dtype=mx.float32)
    dy = inp["dy"]
    cot = mx.zeros_like(st)
    g = inp["g"]
    print(f"\n{tag} T={T}: g range [{g.min().item():.3e}, {g.max().item():.3e}], "
          f"frac(g<1e-12)={(g < 1e-12).mean().item():.4f}, frac(g<1e-6)={(g < 1e-6).mean().item():.4f}")

    # 1) dg comparison, split by clamp region
    primals = [inp["q"], inp["k"], inp["v"], g, inp["beta"]]
    _, gs = mx.vjp(lambda q, k, v, g, b: gated_delta_ops(q, k, v, g, b, st, None), primals, [dy, cot])
    _, gc = mx.vjp(lambda q, k, v, g, b: gated_delta_ops_chunked(q, k, v, g, b, st, None), primals, [dy, cot])
    dg_s, dg_c = gs[3], gc[3]
    mx.eval(dg_s, dg_c)
    sat = g < 1e-11
    diff = mx.abs(dg_c - dg_s)
    print(f"  dg: max|diff| overall={diff.max().item():.3e}  "
          f"max|diff| where g>=1e-11={mx.where(sat, mx.zeros_like(diff), diff).max().item():.3e}  "
          f"max|diff| where g<1e-11={mx.where(sat, diff, mx.zeros_like(diff)).max().item():.3e}")
    unsat = stats(mx.where(sat, mx.zeros_like(dg_c), dg_c), mx.where(sat, mx.zeros_like(dg_s), dg_s))
    print(f"  dg restricted to unsaturated gates: rel={unsat[1]:.2e} cos={unsat[2]:.7f}")

    # 2) parameter-level: da, db (chain through compute_g / sigmoid)
    A_log, dt_bias = inp["A_log"], inp["dt_bias"]
    pr2 = [inp["q"], inp["k"], inp["v"], inp["a"], inp["b"]]

    def mk(ops):
        def f(q, k, v, a, b):
            return ops(q, k, v, compute_g(A_log, a, dt_bias), mx.sigmoid(b), st, None)
        return f

    _, gs2 = mx.vjp(mk(gated_delta_ops), pr2, [dy, cot])
    _, gc2 = mx.vjp(mk(gated_delta_ops_chunked), pr2, [dy, cot])
    mx.eval(*gs2, *gc2)
    for n, a_, b_ in zip(["dq", "dk", "dv", "da", "db"], gc2, gs2):
        s = stats(a_, b_)
        print(f"  param-level {n:3s}: rel={s[1]:.2e} cos={s[2]:.9f}")
print("\nDONE B2")
