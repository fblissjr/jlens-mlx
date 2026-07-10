"""Per-architecture Jacobian providers, resolved by model_type.

Register accelerators in _register(). The default -- for any model_type without a
registered accelerator -- is GenericVjpProvider, which works on any differentiable
MLX model (slow, no custom kernel).
"""
from __future__ import annotations

from ..fit import PROVIDER_REGISTRY


def get_provider(model_type: str):
    """Return an accelerator provider for `model_type`, else GenericVjpProvider."""
    from .generic_vjp import GenericVjpProvider

    cls = PROVIDER_REGISTRY.get(model_type)
    return cls() if cls is not None else GenericVjpProvider()


def _register() -> None:
    from .qwen3_5_gdn import Qwen35GdnProvider

    PROVIDER_REGISTRY["qwen3_5"] = Qwen35GdnProvider


_register()
