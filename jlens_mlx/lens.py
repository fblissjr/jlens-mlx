# NOTE: mirrors heylookitsanllm src/heylook_llm/jspace/lens.py -- keep in sync
# (fit-here / apply-there parity). save()/load() are jlens-mlx additions (the server only reads).
"""The fitted Jacobian lens: per-layer ``J_l`` transport matrices + apply.

A lens is loaded from the offline-converted safetensors (``{str(layer): J_l}``)
plus a JSON sidecar (``source_layers``, ``d_model``, ``final_logit_softcapping``).
``apply`` transports captured residuals into the final-layer basis and unembeds
them through the model's real head (via :class:`~heylook_llm.jspace.capture.ModelAdapter`).

Conversion from the reference PyTorch ``.pt`` (which needs torch) is a dev-time
step and is deliberately NOT in the server runtime -- see the offline converter
in the spike harness / plan.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx


class JSpaceLens:
    """Per-layer ``J_l`` matrices and the readout method.

    Attributes:
        jacobians: ``{layer_index: mx.array[d_model, d_model]}`` (float32).
        source_layers: Sorted fitted layer indices.
        d_model: Residual-stream width.
        softcap: The fit model's ``final_logit_softcapping`` (metadata only; the
            authoritative soft-cap used at apply time comes from the live model
            via the adapter).
        meta: The raw sidecar dict.
    """

    def __init__(self, jacobians: dict[int, mx.array], source_layers, d_model: int,
                 *, softcap: float | None = None, meta: dict | None = None) -> None:
        self.jacobians = {int(l): j.astype(mx.float32) for l, j in jacobians.items()}
        # Materialize now (on the loading thread). mx.load is lazy/mmap-backed; if
        # the first eval happened later on a generation worker thread it would read
        # via the CPU default stream that thread doesn't have -> a hard crash
        # ("There is no Stream(cpu, 0)").
        mx.eval(list(self.jacobians.values()))
        self.source_layers = sorted(int(l) for l in source_layers)
        self.d_model = int(d_model)
        self.softcap = softcap
        self.meta = meta or {}

    def __repr__(self) -> str:
        return (f"JSpaceLens(d_model={self.d_model}, "
                f"source_layers=[{self.source_layers[0]}..{self.source_layers[-1]}] "
                f"({len(self.source_layers)}))")

    @classmethod
    def from_files(cls, safetensors_path, sidecar_path) -> "JSpaceLens":
        """Load a converted lens: ``mx``-safetensors of ``J`` + JSON sidecar."""
        side = json.loads(Path(sidecar_path).read_text())
        arrays = mx.load(str(safetensors_path))       # {str(layer): mx.array}
        jacobians = {int(k): v for k, v in arrays.items()}
        return cls(jacobians=jacobians, source_layers=side["source_layers"],
                   d_model=side["d_model"],
                   softcap=side.get("final_logit_softcapping"), meta=side)

    def transport(self, residual: mx.array, layer: int) -> mx.array:
        """Map a residual ``[..., d_model]`` at ``layer`` into the final basis: ``J_l @ h``."""
        return residual.astype(mx.float32) @ self.jacobians[int(layer)].T

    def apply(self, adapter, residuals: dict[int, mx.array], *,
              positions=None, layers=None) -> dict[int, mx.array]:
        """Lens logits at ``positions`` for each requested layer.

        Args:
            adapter: A :class:`~heylook_llm.jspace.capture.ModelAdapter`.
            residuals: ``{layer: mx.array[seq_len, d_model]}`` from ``capture_residuals``.
            positions: Token positions to read out (Python indexing incl. negatives);
                ``None`` returns every position.
            layers: Subset of :attr:`source_layers` to read; defaults to all.

        Returns:
            ``{layer: mx.array[n_positions, vocab_size]}``.
        """
        layers = list(layers) if layers is not None else self.source_layers
        out: dict[int, mx.array] = {}
        for l in layers:
            h = residuals[int(l)]
            if positions is not None:
                seq = h.shape[0]
                idx = mx.array([p % seq for p in positions])   # normalize negatives
                h = h[idx]
            out[int(l)] = adapter.unembed(self.transport(h, l))
        return out


def save(jacobians, sidecar, out_dir):
    """Write a fitted lens: lens.safetensors ({str(layer): J_l}) + lens.sidecar.json.
    Inverse of JSpaceLens.from_files; the exact format the heylook registry loads."""
    from pathlib import Path as _Path
    out = _Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(out / "lens.safetensors"),
        {str(int(l)): j.astype(mx.float32) for l, j in jacobians.items()},
    )
    (out / "lens.sidecar.json").write_text(json.dumps(sidecar, indent=2))
    return out


def load(lens_dir):
    """Load a JSpaceLens from a dir holding lens.safetensors + lens.sidecar.json."""
    from pathlib import Path as _Path
    d = _Path(lens_dir)
    return JSpaceLens.from_files(d / "lens.safetensors", d / "lens.sidecar.json")
