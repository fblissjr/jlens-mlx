"""PR #1389 sections A (forward exactness), B (grad correctness), C (three-way vs our kernel)."""
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REAL, fmt, load_jlens_gdn, make_inputs, stats

from mlx_lm.models.gated_delta import (
    gated_delta_kernel,
    gated_delta_ops,
    gated_delta_ops_chunked,
)

print("mlx", mx.__version__, "| metal:", mx.metal.is_available())

SMALL = dict(Hk=4, Hv=4, Dk=32, Dv=32)


def zero_state(B, Hv, Dv, Dk):
    return mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)


# ---------------- A: forward exactness ----------------
print("\n=== A: forward exactness (chunked & kernel vs sequential ops) ===")
cases = []
for T in (17, 64, 128, 200, 512):
    cases.append(("small", T, SMALL))
for T in (128, 512):
    cases.append(("real27B", T, REAL))

for dtype in (mx.float32, mx.bfloat16):
    print(f"\n--- dtype={dtype} ---")
    for tag, T, dims in cases:
        B = 1
        inp = make_inputs(B, T, dims["Hk"], dims["Hv"], dims["Dk"], dims["Dv"],
                          dtype=dtype, seed=hash((tag, T)) % 2**31)
        st = zero_state(B, dims["Hv"], dims["Dv"], dims["Dk"])
        args = (inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"])
        y_seq, s_seq = gated_delta_ops(*args, st, None)
        y_chk, s_chk = gated_delta_ops_chunked(*args, st, None)
        mx.eval(y_seq, s_seq, y_chk, s_chk)
        sy = stats(y_chk, y_seq)
        ss = stats(s_chk, s_seq)
        line = (f"{tag:8s} T={T:<4d} chunked-vs-seq: "
                f"y[max_abs={sy[0]:.2e} rel={sy[1]:.2e} cos={sy[2]:.7f}] "
                f"state[max_abs={ss[0]:.2e} rel={ss[1]:.2e}]")
        y_ker, s_ker = gated_delta_kernel(*args, st, None)
        mx.eval(y_ker, s_ker)
        ky = stats(y_ker, y_seq)
        line += f" | kernel-vs-seq: y[max_abs={ky[0]:.2e} rel={ky[1]:.2e}]"
        print(line)


# ---------------- B: gradient correctness ----------------
print("\n=== B: vjp grads, chunked vs sequential (fp32, cotangent on y, zero on state) ===")
bcases = [("small", T, SMALL) for T in (17, 64, 128, 200)] + [("real27B", 128, REAL)]

for tag, T, dims in bcases:
    B = 1
    inp = make_inputs(B, T, dims["Hk"], dims["Hv"], dims["Dk"], dims["Dv"],
                      dtype=mx.float32, seed=1000 + T)
    st = zero_state(B, dims["Hv"], dims["Dv"], dims["Dk"])
    dy = inp["dy"]
    cot_state = mx.zeros_like(st)
    primals = [inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"]]

    def f_seq(q, k, v, g, beta):
        return gated_delta_ops(q, k, v, g, beta, st, None)

    def f_chk(q, k, v, g, beta):
        return gated_delta_ops_chunked(q, k, v, g, beta, st, None)

    _, g_seq = mx.vjp(f_seq, primals, [dy, cot_state])
    _, g_chk = mx.vjp(f_chk, primals, [dy, cot_state])
    mx.eval(*g_seq, *g_chk)
    names = ["dq", "dk", "dv", "dg", "dbeta"]
    parts = []
    for n, a, b in zip(names, g_chk, g_seq):
        s = stats(a, b)
        parts.append(f"{n}[rel={s[1]:.2e} cos={s[2]:.7f}]")
    print(f"{tag:8s} T={T:<4d} " + " ".join(parts))


# ---------------- C: three-way vs our jlens kernel ----------------
print("\n=== C: three-way VJP at T<=128, scalar gating, GQA (ours vs chunked vs sequential) ===")
jl = load_jlens_gdn()
ccases = [("smallGQA", 64, dict(Hk=2, Hv=4, Dk=32, Dv=32)),
          ("smallGQA", 128, dict(Hk=2, Hv=4, Dk=32, Dv=32)),
          ("real27B", 64, REAL),
          ("real27B", 128, REAL)]

for tag, T, dims in ccases:
    B = 1
    inp = make_inputs(B, T, dims["Hk"], dims["Hv"], dims["Dk"], dims["Dv"],
                      dtype=mx.float32, seed=2000 + T + dims["Hk"])
    st = zero_state(B, dims["Hv"], dims["Dv"], dims["Dk"])
    dy = inp["dy"]
    cot_state = mx.zeros_like(st)
    primals = [inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"]]

    def f_seq(q, k, v, g, beta):
        return gated_delta_ops(q, k, v, g, beta, st, None)

    def f_chk(q, k, v, g, beta):
        return gated_delta_ops_chunked(q, k, v, g, beta, st, None)

    _, g_seq = mx.vjp(f_seq, primals, [dy, cot_state])
    _, g_chk = mx.vjp(f_chk, primals, [dy, cot_state])
    g_ours = jl.gdn_kernel_vjp(inp["q"], inp["k"], inp["v"], inp["g"],
                               inp["beta"], st, dy)
    mx.eval(*g_seq, *g_chk, *g_ours)
    names = ["dq", "dk", "dv", "dg", "dbeta"]
    print(f"{tag:8s} T={T}")
    for n, o, c, s in zip(names, g_ours, g_chk, g_seq):
        so = stats(o, s)
        sc = stats(c, s)
        oc = stats(o, c)
        print(f"   {n:5s} ours-vs-seq[rel={so[1]:.2e} cos={so[2]:.7f}]  "
              f"chunked-vs-seq[rel={sc[1]:.2e} cos={sc[2]:.7f}]  "
              f"ours-vs-chunked[rel={oc[1]:.2e} cos={oc[2]:.7f}]")

print("\nDONE ABC")
