"""Fitting corpus: recipe, position mask, on-policy builder.

Corpus choice is load-bearing (closest to quantization-calibration data) -- see
docs/DESIGN.md. Everything here is swappable and provenance-stamped; there is NO
hardcoded WikiText and NO hardcoded "skip first 4 positions".
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PositionMask:
    """Which token positions the Jacobian is averaged over. Role/sink-aware, not a
    fixed skip count (that heuristic is for raw-text BOS sinks; wrong under ChatML)."""

    include_assistant: bool = True
    include_think: bool = True
    include_user: bool = False
    drop_bos_sink: bool = True
    drop_role_tokens: bool = True
    predicate: object | None = None  # optional (token_id, position, role) -> bool


@dataclass
class Recipe:
    """A named, reproducible fitting-corpus spec, stamped onto the lens sidecar as
    provenance (recipe + model SHA + position policy)."""

    name: str
    sources: list[str] = field(default_factory=list)   # dataset ids / prompt files
    weights: list[float] = field(default_factory=list)  # mixing weights over sources
    n_prompts: int = 100
    seed: int = 0
    chat_templated: bool = True
    on_policy_fraction: float = 0.0  # 0 = human text only; 0.5-0.7 for a chat lens
    positions: PositionMask = field(default_factory=PositionMask)


@dataclass
class Corpus:
    """Materialized fitting corpus: tokenized prompts + per-prompt position masks."""

    recipe: Recipe
    # prompts: list[mx.array]; masks: list[mx.array] -- populated by build_corpus


def build_corpus(model, processor, recipe: Recipe) -> Corpus:
    """Render `recipe` into a Corpus: chat-template the prompts, optionally sample
    on-policy generations, and apply the PositionMask.

    Port prompt loading from vendor/jlens_qwen36/prompts.py, then add chat
    templating + on-policy sampling + the mask (new here). See MIGRATION.md step 3.
    """
    raise NotImplementedError("port prompts.py + add chat/on-policy/position-mask")
