"""Unit tests for the corpus builder's pure logic (no model, no GPU, no network).

corpus.py is MLX-free at import (datasets/mlx are lazy-imported inside the loaders), so these
run on CPU and are safe alongside a live fit. The HF-loading + on-policy paths are covered by
the live smoke in scripts/ (network/GPU); here we lock the weighting, position-masking, and
prompt-extraction logic that a refactor could silently break.

Run (from the heylook dir / venv):  uv run pytest <jlens-mlx>/tests/test_corpus.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jlens_mlx.corpus import (  # noqa: E402
    ABLITERATED_QWEN, SAFETY_BENIGN, SAFETY_HARMFUL, WIKITEXT_CONTROL,
    Corpus, CorpusItem, PositionMask, Recipe, Stratum,
    _extract_prompt, _weighted_counts, build_positions,
)


# --- _weighted_counts ------------------------------------------------------------------------

@pytest.mark.parametrize("recipe", [ABLITERATED_QWEN, SAFETY_HARMFUL, SAFETY_BENIGN, WIKITEXT_CONTROL])
def test_weighted_counts_sum_exact(recipe):
    counts = _weighted_counts(recipe)
    assert sum(counts) == recipe.n_prompts
    assert len(counts) == len(recipe.strata)
    assert all(c >= 0 for c in counts)


def test_weighted_counts_single_stratum():
    r = Recipe(name="one", n_prompts=17, strata=[Stratum("x", 1.0, "chat")])
    assert _weighted_counts(r) == [17]


def test_weighted_counts_rounding_remainder_to_largest():
    # 3 equal weights over 10 -> [3,3,3] leaves 1; it lands on a stratum (deterministic), sum==10.
    r = Recipe(name="three", n_prompts=10,
               strata=[Stratum("a", 1 / 3, "chat"), Stratum("b", 1 / 3, "chat"),
                       Stratum("c", 1 / 3, "chat")])
    counts = _weighted_counts(r)
    assert sum(counts) == 10
    assert max(counts) - min(counts) <= 1  # spread by at most the remainder


def test_weighted_counts_zero_weight_stratum():
    r = Recipe(name="z", n_prompts=8,
               strata=[Stratum("a", 1.0, "chat"), Stratum("b", 0.0, "safety")])
    counts = _weighted_counts(r)
    assert sum(counts) == 8
    assert counts[1] == 0


# --- build_positions -------------------------------------------------------------------------

def test_positions_on_policy_span():
    # completion present (len > prefix+1): assistant span [prefix, len-1)
    pos = build_positions(list(range(20)), prompt_prefix_len=10, mask=PositionMask())
    assert pos == list(range(10, 19))


def test_positions_human_text_span():
    # no completion: content span inside the prompt, drop BOS/scaffold (skip_first) + last
    pos = build_positions(list(range(10)), prompt_prefix_len=10, mask=PositionMask(), skip_first=4)
    assert pos == list(range(4, 9))


def test_positions_think_only():
    pos = build_positions(list(range(20)), 10,
                          PositionMask(include_assistant=False, include_think=True),
                          think_span=(12, 16))
    assert pos == [12, 13, 14, 15]


def test_positions_assistant_minus_think():
    pos = build_positions(list(range(20)), 10,
                          PositionMask(include_assistant=True, include_think=False),
                          think_span=(12, 16))
    assert pos == [10, 11, 16, 17, 18]


def test_positions_predicate_filter():
    # keep only even token ids among the assistant span
    m = PositionMask(predicate=lambda tid, p, role: tid % 2 == 0)
    pos = build_positions(list(range(20)), 10, m)
    assert pos == [10, 12, 14, 16, 18]


def test_positions_too_short_returns_empty():
    assert build_positions([7], 1, PositionMask()) == []
    assert build_positions([], 0, PositionMask()) == []


def test_positions_never_empty_fallback():
    # a predicate that rejects everything -> fall back to the last valid index, never []
    m = PositionMask(predicate=lambda *_: False)
    pos = build_positions(list(range(20)), 10, m)
    assert pos == [18]


def test_positions_human_text_short_prefix_clamps():
    # prefix small: start clamps so the range is non-empty
    pos = build_positions(list(range(6)), prompt_prefix_len=6, mask=PositionMask(), skip_first=4)
    assert pos and all(0 <= p < 5 for p in pos)


# --- _extract_prompt -------------------------------------------------------------------------

def test_extract_prompt_plain_fields():
    assert _extract_prompt({"prompt": "hi"}) == "hi"
    assert _extract_prompt({"Goal": "do x"}) == "do x"
    assert _extract_prompt({"question": " spaced "}) == "spaced"


def test_extract_prompt_conversation_and_messages():
    assert _extract_prompt({"conversation": [{"role": "user", "content": "q1"}]}) == "q1"
    assert _extract_prompt({"messages": [{"from": "human", "value": "q2"}]}) == "q2"
    # first-turn fallback when no explicit user role
    assert _extract_prompt({"conversation": [{"content": "first"}]}) == "first"


def test_extract_prompt_field_hint_priority():
    row = {"prompt": "generic", "special": "hinted"}
    assert _extract_prompt(row, field_hint="special") == "hinted"


def test_extract_prompt_none_when_absent_or_empty():
    assert _extract_prompt({"junk": 1}) is None
    assert _extract_prompt({"prompt": "   "}) is None
    assert _extract_prompt({}) is None


# --- dataclasses -----------------------------------------------------------------------------

def test_corpus_accessors():
    items = [CorpusItem([1, 2, 3], [1], "chat", True), CorpusItem([4, 5], [0], "safety", False)]
    c = Corpus(recipe=WIKITEXT_CONTROL, items=items, provenance={"recipe": "x"})
    assert c.prompts_ids() == [[1, 2, 3], [4, 5]]
    assert c.positions() == [[1], [0]]


def test_recipe_defaults():
    r = Recipe(name="d")
    assert r.n_prompts == 300 and r.chat_templated and 0 <= r.on_policy_fraction <= 1
    assert isinstance(r.positions, PositionMask)
