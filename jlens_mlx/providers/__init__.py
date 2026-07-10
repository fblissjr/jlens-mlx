"""The VJP fit primitive + the deferred per-arch accelerators.

The direct-VJP baseline (jlens_mlx.fit + jacobian_via_vjp) is arch-agnostic; the only
per-arch piece is the "tail runner" (how to run decoder blocks with the right masks), whose
default lives in fit.make_tail. qwen3_5 (GDN hybrid) will get a faster accelerator (a Metal
GDN backward kernel + a GDN-aware tail) — see qwen3_5_gdn (deferred).
"""
from __future__ import annotations

from .generic_vjp import jacobian_via_vjp

__all__ = ["jacobian_via_vjp"]
