"""jlens-mlx: modular Jacobian-lens (j-space) fitting on Apple silicon (MLX).

Public surface (see docs/DESIGN.md):
  - fit.fit_lens(...)             generic chain driver
  - fit.PROVIDER_REGISTRY         model_type -> JacobianProvider
  - corpus.Recipe / PositionMask  swappable, provenance-stamped fitting corpus
  - lens.save / lens.load / lens.apply
  - verify.parity_vs_oracle / verify.fidelity_gate / verify.diff

STATUS: scaffold (2026-07-10). The validated numeric core is vendored under
vendor/jlens_qwen36/ and ported per MIGRATION.md.
"""

__version__ = "0.0.1"
