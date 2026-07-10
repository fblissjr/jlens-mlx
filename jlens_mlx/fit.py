"""Generic, architecture-agnostic Jacobian-lens fitting driver.

Fitting decomposes into (1) this driver -- it chains per-layer Jacobians
J_{l-1} = J_l @ M_l and averages over a corpus -- and (2) a per-architecture
JacobianProvider that supplies M_l (the ONLY arch-specific part), resolved from
PROVIDER_REGISTRY by model_type. See docs/DESIGN.md.

Chain-indexing invariant: J_l transports acts[l] (residual AFTER layer l);
M_l is evaluated at layer l's INPUT acts[l-1]. Getting this wrong gives ~33-49%
rel error instead of ~3-5%.

STATUS: scaffold. The validated guts live in vendor/jlens_qwen36/ and are ported
here per MIGRATION.md step 2. Signatures are the contract; bodies raise until wired.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import mlx.core as mx

    from .corpus import Corpus


@runtime_checkable
class JacobianProvider(Protocol):
    """Supplies the per-layer Jacobian M_l = d(layer_l)/d(input) -- the only
    architecture-specific piece of the fit."""

    model_type: str

    def final_norm_jacobian(self, model) -> "mx.array":
        """Closed-form Jacobian of the model's final norm -- the chain seed J_L."""
        ...

    def layer_jacobian(self, model, layer_idx: int, input_acts: "mx.array") -> "mx.array":
        """M_l for decoder layer `layer_idx`, evaluated at that layer's INPUT
        activations (see the chain-indexing invariant above)."""
        ...


# model_type -> provider class. Populated by jlens_mlx.providers. The default,
# when a model_type has no registered accelerator, is the universal generic-VJP
# provider (see resolve_provider).
PROVIDER_REGISTRY: dict[str, type] = {}


def resolve_provider(model_type: str) -> JacobianProvider:
    """The accelerator for this arch if registered, else the generic VJP baseline."""
    from .providers import get_provider

    return get_provider(model_type)


def fit_lens(
    model,
    corpus: "Corpus",
    *,
    source_layers: list[int] | None = None,
    provider: JacobianProvider | None = None,
    out_dir: str | None = None,
):
    """Fit a Jacobian lens: chain M_l over `source_layers` averaged over `corpus`,
    seeded by the provider's final-norm Jacobian, and (if `out_dir`) save
    safetensors + a provenance sidecar (recipe + model SHA + position policy).

    `provider` defaults to resolve_provider(model.model_type). Returns the
    per-layer J dict.
    """
    raise NotImplementedError(
        "port the chain driver from vendor/jlens_qwen36/fit_analytic.py + analytic.py "
        "(strip qwen specifics into the provider) -- MIGRATION.md step 2"
    )
