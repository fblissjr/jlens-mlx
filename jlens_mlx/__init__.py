"""jlens-mlx: modular Jacobian-lens (j-space) fitting on Apple silicon (MLX).

Public surface (see docs/DESIGN.md):
  - fit.fit_lens(...)             direct end-to-end VJP fitter (Anthropic design)
  - fit.fit_prompt / make_tail / valid_positions
  - providers.jacobian_via_vjp    the per-output-dim VJP estimator
  - corpus.Recipe / PositionMask  swappable, provenance-stamped fitting corpus
  - lens.JSpaceLens / lens.save / lens.load   (apply: norm OUTSIDE J, real norm at decode)

STATUS (2026-07-10): baseline fitter GREEN on gpt2 (scripts/fit_gpt2_baseline.py).
Follows anthropics/jacobian-lens (direct autograd, no chain, no closed-form seed). The
qwen3_5 GDN Metal kernel is a deferred optional speed accelerator (providers/qwen3_5_gdn).
We port pieces (verified vs mx.vjp), never vendor.
"""

__version__ = "0.0.1"
