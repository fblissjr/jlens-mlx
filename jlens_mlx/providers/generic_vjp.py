"""Universal Jacobian provider: mx.vjp (exact) or Rademacher-probed (fast).

Works on ANY differentiable MLX model -- no custom kernel. Slow, but the baseline
that keeps this from being a single-arch trainer. The default when a model_type has
no registered accelerator.

Port from vendor/jlens_qwen36/fit.py (per_layer_jacobian) + probing.py.
"""
from __future__ import annotations


class GenericVjpProvider:
    model_type = "*"

    def __init__(self, probes: int | None = None):
        # probes=None -> exact D one-hot VJPs; probes=k -> Rademacher estimator
        # M_hat = (1/k) sum_i v_i (v_i^T M), unbiased -- readout-grade, ~10x faster.
        self.probes = probes

    def final_norm_jacobian(self, model):
        raise NotImplementedError(
            "port the closed-form norm Jacobian from vendor/jlens_qwen36/analytic.py"
        )

    def layer_jacobian(self, model, layer_idx, input_acts):
        raise NotImplementedError(
            "port vendor/jlens_qwen36/fit.py::per_layer_jacobian (exact) + probing.py (probed)"
        )
