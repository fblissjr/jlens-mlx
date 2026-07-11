"""jlens-mlx: modular Jacobian-lens (j-space) fitting on Apple silicon (MLX).

Public surface (see docs/DESIGN.md):
  - fit.fit_lens(...)             direct end-to-end VJP fitter (Anthropic design)
  - fit.fit_prompt / make_tail / valid_positions
  - providers.jacobian_via_vjp    the per-output-dim VJP estimator
  - corpus.Recipe / PositionMask  swappable, provenance-stamped fitting corpus
  - lens.JSpaceLens / lens.save / lens.load   (apply: norm OUTSIDE J, real norm at decode)

Design: follows anthropics/jacobian-lens (direct end-to-end autograd, no chain-of-M_l, no
closed-form norm seed). The exact reverse-mode chain fitter (chain.py) is the verified default;
the direct VJP is the golden reference. We port pieces (verified vs mx.vjp), never vendor.

Coverage/status is not restated here (it rots -- the old dated STATUS line did): see README.md,
docs/DESIGN.md, and git history for what's GREEN and which arch families are covered.
"""

__version__ = "0.0.1"
