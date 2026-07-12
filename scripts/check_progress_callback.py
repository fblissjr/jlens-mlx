"""Progress-callback gate: the chunk-progress hook added to `generic_vjp.jacobian_via_vjp` and
`chain.fit_prompt_chain` (for operational visibility -- see JLENS_WATCHDOG_S / the fit_band_corpus
heartbeat) must actually fire on the REAL cotangent-chunk loop under Metal, not a mock, and must
NOT change any numerics (hard constraint: this is observability-only instrumentation). Reuses the
tiny synthetic qwen3_5 (GDN hybrid, no download) from check_qwen3_5_synthetic.py.

Checks:
  [1] generic_vjp.jacobian_via_vjp(progress=...) on a toy tail: invocations are monotonically
      increasing (1,total)..(total,total), final call has done==total, total matches ceil(D/C),
      and progress=None (the default) reproduces bit-identical output.
  [2] chain.fit_prompt_chain(progress=...) on the tiny qwen3_5 model (real GDN + FA blocks):
      same shape checks, PLUS the chained J still matches fit.fit_prompt's direct J (proves the
      progress wiring is instrumentation-only, no math change), PLUS progress=None reproduces
      bit-identical output.

Run (Metal, unsandboxed):  uv run python scripts/check_progress_callback.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jlens_mlx.capture import ModelAdapter  # noqa: E402
from jlens_mlx.chain import fit_prompt_chain  # noqa: E402
from jlens_mlx.fit import fit_prompt  # noqa: E402
from jlens_mlx.providers.generic_vjp import jacobian_via_vjp  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def _monotonic(calls: list[tuple[int, int]]) -> bool:
    return all(calls[i][0] < calls[i + 1][0] for i in range(len(calls) - 1))


def _load_tiny_qwen3_5():
    """Reuse (not duplicate) check_qwen3_5_synthetic.py's tiny_qwen3_5() -- no HF download, real
    GDN + full-attention blocks, GQA repeat 3 like the real 27B."""
    spec = importlib.util.spec_from_file_location(
        "check_qwen3_5_synthetic", ROOT / "scripts" / "check_qwen3_5_synthetic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.tiny_qwen3_5


def check_generic_vjp_progress() -> None:
    print("[1] generic_vjp.jacobian_via_vjp progress callback (toy tail, real mx.vjp)")
    D, S = 37, 6  # odd D -- exercises a ragged final chunk
    h = mx.random.normal((1, S, D))
    valid = mx.array([1, 2, 3])
    W = mx.random.normal((D, D)) * 0.1

    def tail(x):
        return x @ W

    calls: list[tuple[int, int]] = []
    J = jacobian_via_vjp(tail, h, valid, chunk_size=10, progress=lambda d, t: calls.append((d, t)))
    mx.eval(J)
    total = calls[-1][1] if calls else None
    expected_total = -(-D // 10)
    check("callback fired", len(calls) > 0, f"{len(calls)} calls: {calls}")
    check("monotonically increasing done", _monotonic(calls), f"{calls}")
    check("final call done==total", bool(calls) and calls[-1][0] == calls[-1][1],
          f"{calls[-1] if calls else None}")
    check("total matches ceil(D/chunk)", total == expected_total,
          f"total={total} expected={expected_total}")

    # default progress=None must reproduce the EXACT prior behavior (bit-identical -- proves the
    # callback is pure instrumentation, no math change).
    J2 = jacobian_via_vjp(tail, h, valid, chunk_size=10)
    mx.eval(J2)
    err = float(mx.abs(J - J2).max().item())
    check("progress=None bit-identical", err == 0.0, f"max_abs_diff={err:.2e}")


def check_chain_progress(tiny_qwen3_5) -> None:
    print("[2] chain.fit_prompt_chain progress callback (tiny qwen3_5, real GDN+FA blocks)")
    model = tiny_qwen3_5()
    ad = ModelAdapter(model)
    ids = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
    target = ad.n_layers - 1
    sources = [2, target]

    calls: list[tuple[int, int]] = []
    J_chain, _ = fit_prompt_chain(model, ids, sources, adapter=ad, target_layer=target,
                                  skip_first=2, chunk_size=10,
                                  progress=lambda d, t: calls.append((d, t)))
    mx.eval(list(J_chain.values()))
    check("callback fired", len(calls) > 0, f"{len(calls)} calls: {calls}")
    check("monotonically increasing done", _monotonic(calls), f"{calls}")
    check("final call done==total", bool(calls) and calls[-1][0] == calls[-1][1],
          f"{calls[-1] if calls else None}")

    # The chain J must still match the trusted direct-VJP baseline -- progress is instrumentation
    # only, never a math change (the hard constraint for this whole task).
    J_direct, _ = fit_prompt(model, ids, sources, adapter=ad, target_layer=target,
                             skip_first=2, chunk_size=10)
    for l in sources:
        a = np.asarray(J_chain[l], dtype=np.float64).ravel()
        b = np.asarray(J_direct[l], dtype=np.float64).ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        check(f"J_{l} chain==direct (cos)", cos > 0.99999, f"cos={cos:.6f}")

    # default progress=None must reproduce the exact prior behavior, bit-identical.
    J_chain2, _ = fit_prompt_chain(model, ids, sources, adapter=ad, target_layer=target,
                                   skip_first=2, chunk_size=10)
    mx.eval(list(J_chain2.values()))
    for l in sources:
        err = float(mx.abs(J_chain[l] - J_chain2[l]).max().item())
        check(f"J_{l} progress=None bit-identical", err == 0.0, f"max_abs_diff={err:.2e}")

    # fit_prompt (direct path) also threads progress through, per-layer (resets each layer).
    calls_direct: list[tuple[int, int]] = []
    fit_prompt(model, ids, sources, adapter=ad, target_layer=target, skip_first=2, chunk_size=10,
              progress=lambda d, t: calls_direct.append((d, t)))
    n_layer_resets = sum(1 for d, _ in calls_direct if d == 1)
    check("fit_prompt (direct) progress fires per layer", len(calls_direct) > 0 and
          n_layer_resets == len(sources), f"{calls_direct}")


def main() -> int:
    mx.random.seed(0)
    metal = mx.metal.is_available()
    print(f"metal={metal}")
    if not metal:
        print("NO METAL: this gate needs a real Metal chunk loop -- run on Apple silicon.")
        return 1

    check_generic_vjp_progress()
    tiny_qwen3_5 = _load_tiny_qwen3_5()
    check_chain_progress(tiny_qwen3_5)

    print(f"\nPROGRESS CALLBACK GATE {'PASS' if not FAILURES else 'FAIL: ' + ', '.join(FAILURES)}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
