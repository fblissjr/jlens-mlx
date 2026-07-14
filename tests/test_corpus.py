"""Unit tests for the corpus builder's pure logic (no model, no GPU, no network).

corpus.py is MLX-free at import (datasets/mlx are lazy-imported inside the loaders), so these
run on CPU and are safe alongside a live fit. The HF-loading + on-policy paths are covered by
the live smoke in scripts/ (network/GPU); here we lock the weighting, position-masking, and
prompt-extraction logic that a refactor could silently break.

Run (from the heylook dir / venv):  uv run pytest <jlens-mlx>/tests/test_corpus.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jlens_mlx.corpus import (  # noqa: E402
    ABLITERATED_QWEN, SAFETY_BENIGN, SAFETY_HARMFUL, WIKITEXT_CONTROL,
    Corpus, CorpusItem, PositionMask, Recipe, Stratum,
    _chat_ids, _extract_prompt, _prompt_fits, _weighted_counts,
    build_positions, decode_corpus, diversity_report, resolve_on_policy_fraction,
)


# --- decode_corpus (readable inspection render) ----------------------------------------------

class _FakeTok:
    """Marks whether special tokens were kept, so the test can assert the render + boundary split."""
    def decode(self, ids, skip_special_tokens=False):
        return f"[{'nosp' if skip_special_tokens else 'sp'}:{'-'.join(map(str, ids))}]"


def test_decode_corpus_shows_specials_and_splits_on_policy():
    c = Corpus(recipe=Recipe(name="t", strata=[Stratum("x", 1.0, "chat")]),
               items=[CorpusItem([1, 2, 3, 4, 5], [2, 3], "safety", on_policy=True),
                      CorpusItem([9, 8, 7], [0, 1], "benign", on_policy=False)],
               provenance={"recipe": "t", "strata": {"x": 2}, "seed": 0})
    md = decode_corpus(c, _FakeTok())
    # header + per-item metadata
    assert "Decoded corpus: t" in md
    assert "stratum=safety" in md and "on_policy=True" in md
    # special tokens SHOWN by default (skip_special_tokens=False -> the 'sp' marker)
    assert "[sp:" in md and "[nosp:" not in md
    # on-policy item is split at min(positions)=2: prompt ids[:2], response ids[2:]
    assert "model's on-policy response" in md
    assert "[sp:1-2]" in md and "[sp:3-4-5]" in md
    # human-text item decoded whole, no split
    assert "[sp:9-8-7]" in md


def test_decode_corpus_show_special_false_hides_them():
    c = Corpus(recipe=Recipe(name="t", strata=[Stratum("x", 1.0, "chat")]),
               items=[CorpusItem([1, 2], [0], "chat", on_policy=False)], provenance={})
    md = decode_corpus(c, _FakeTok(), show_special=False)
    assert "[nosp:1-2]" in md and "[sp:" not in md


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


def test_positions_human_text_content_span_role_aware():
    # No completion, but content_start/content_end are supplied (the ROLE-AWARE span from
    # _user_content_span, computed from the actual chat template): ids stand in for a
    # genprompt-included off-policy render whose real user content only spans [4, 14); tokens
    # 14..19 stand in for the trailing <|im_start|>assistant\n<think>... scaffold that
    # add_generation_prompt=True appended. Positions must start at content_start and stop BEFORE
    # content_end, never touching the trailing scaffold.
    ids = list(range(20))
    pos = build_positions(ids, prompt_prefix_len=20, mask=PositionMask(),
                          content_start=4, content_end=14)
    assert pos == list(range(4, 14))


def test_positions_human_text_skip_first_unchanged_when_content_span_omitted():
    # When content_start/content_end are NOT supplied, the old skip_first-based fallback must be
    # completely unchanged (no regression from adding the role-aware path).
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


# --- _prompt_fits (sequence-length gate) -----------------------------------------------------
# Bounds per-item fit cost so no single corpus item outlives a checkpoint window: the chain
# fitter carries an [C, S, D] cotangent, so wall-clock scales with S (sequence length). Long
# items (e.g. multi-paragraph math prompts) are DROPPED, not truncated -- a truncated prompt
# yields meaningless activations.

def test_prompt_fits_no_cap_always_true():
    assert _prompt_fits(10_000, None) is True
    assert _prompt_fits(10_000, None, reserve=999) is True


def test_prompt_fits_under_cap():
    assert _prompt_fits(300, 512) is True


def test_prompt_fits_over_cap():
    assert _prompt_fits(600, 512) is False


def test_prompt_fits_boundary_exact():
    # prefix == cap fits (<=), one over does not
    assert _prompt_fits(512, 512) is True
    assert _prompt_fits(513, 512) is False


def test_prompt_fits_reserve_for_on_policy_completion():
    # an on-policy item reserves room for the generated span: prefix + reserve must fit
    assert _prompt_fits(400, 512, reserve=64) is True      # 464 <= 512
    assert _prompt_fits(480, 512, reserve=64) is False     # 544 > 512
    assert _prompt_fits(448, 512, reserve=64) is True      # 512 == 512 boundary


def test_prompt_fits_negative_reserve_clamped():
    # a nonsensical negative reserve must not loosen the bound below the prefix itself
    assert _prompt_fits(512, 512, reserve=-100) is True
    assert _prompt_fits(513, 512, reserve=-100) is False


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
    assert r.enable_thinking is False


def test_corpus_json_roundtrip(tmp_path):
    # Serialize -> load must reproduce the items exactly (so a resumed fit reuses the same corpus,
    # skipping on-policy regen and keeping checkpoint item-order aligned).
    items = [CorpusItem([1, 2, 3, 4], [1, 2], "safety", True),
             CorpusItem([5, 6], [0], "chat", False)]
    c = Corpus(recipe=ABLITERATED_QWEN, items=items,
               provenance={"recipe": "x", "strata": {"a:b:c": 2}})
    p = tmp_path / "corpus.json"
    c.to_json(p)
    loaded = Corpus.from_json(p)
    assert loaded.prompts_ids() == c.prompts_ids()
    assert loaded.positions() == c.positions()
    assert [it.stratum for it in loaded.items] == ["safety", "chat"]
    assert [it.on_policy for it in loaded.items] == [True, False]
    assert loaded.provenance == c.provenance
    assert loaded.recipe.name == ABLITERATED_QWEN.name


# --- diversity_report (corpus diversity gate) ------------------------------------------------
# Regression proof for the 7h-of-GPU-time incident: a corpus whose fitted positions are mostly
# shared/boilerplate tokens (rather than diverse content) yields a near-useless averaged
# Jacobian. `shared_fraction` over the fitted-position tokens is the gate metric.

def test_diversity_report_structure_keys():
    items = [CorpusItem([1, 2, 3], [0, 1], "chat", True), CorpusItem([4, 5, 6], [0, 1], "safety", False)]
    d = diversity_report(items)
    for key in ("n_items", "total_positions", "unique_token_types", "shared_fraction", "on_policy", "off_policy"):
        assert key in d
    for sub in (d["on_policy"], d["off_policy"]):
        for key in ("n_items", "total_positions", "unique_token_types", "shared_fraction"):
            assert key in sub
    assert d["on_policy"]["n_items"] == 1 and d["off_policy"]["n_items"] == 1


def test_diversity_report_degenerate_shared_tokens():
    # 6 items whose fitted positions mostly point at the SAME repeated token id (99), varying
    # only one trailing position per item -- a near-degenerate/boilerplate-heavy corpus.
    items = []
    for i in range(6):
        ids = [10, 11, 99, 99, 99, 99, 12]
        positions = [2, 3, 4, 5]  # 4 fitted positions per item, all pointing at id 99 by default
        items.append(CorpusItem(list(ids), list(positions), "chat", False))
    # Vary the LAST fitted position for two items so it's not literally identical across the board.
    items[0].ids[5] = 77
    items[1].ids[5] = 78

    d = diversity_report(items)
    assert d["n_items"] == 6
    assert d["total_positions"] == 24  # 6 items * 4 positions each
    assert d["shared_fraction"] > 0.9


def test_diversity_report_diverse_disjoint_tokens():
    # 6 items whose fitted-position tokens are entirely disjoint token ids across items -- no
    # token id repeats across enough items to be "shared".
    items = [CorpusItem([i * 100, i * 100 + 1, i * 100 + 2], [0, 1, 2], "chat", False)
             for i in range(6)]
    d = diversity_report(items)
    assert d["n_items"] == 6
    assert d["total_positions"] == 18
    assert d["shared_fraction"] < 0.1


def test_diversity_report_band_n12b_corpus_regression():
    # The REAL degenerate corpus that cost 7h of GPU time: this is the regression proof that the
    # gate would have caught it. Fixture derived from a real checkpoint's corpus.json with token
    # ids ANONYMIZED via a dense equality-preserving remap -- diversity_report depends only on
    # id equality, so shared_fraction is preserved exactly while the fixture stays undecodable
    # (the raw corpus carries content that must not be committed; see decode_corpus's header).
    fixture = Path(__file__).parent / "fixtures" / "band-n12b-corpus.json"
    corpus = Corpus.from_json(fixture)
    d = diversity_report(corpus.items)
    assert d["shared_fraction"] > 0.5


# --- _chat_ids / enable_thinking (real tokenizer, chat-template-dependent) --------------------
# Confirms the core fix: passing enable_thinking explicitly always overrides mlx_lm's
# TokenizerWrapper.apply_chat_template implicit default (self.has_thinking when the kwarg is
# omitted). Needs a REAL local tokenizer dir (no weights loaded) -- skips cleanly if unset.

def test_chat_ids_enable_thinking_explicit_override():
    tok_path = os.environ.get("JLENS_TEST_TOKENIZER")
    if not tok_path:
        pytest.skip("JLENS_TEST_TOKENIZER not set")
    if not Path(tok_path).exists():
        pytest.skip(f"JLENS_TEST_TOKENIZER path does not exist: {tok_path}")

    def _render(tokenizer, enable_thinking: bool) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}], add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking)

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        candidate = AutoTokenizer.from_pretrained(tok_path)
        # A raw AutoTokenizer from a local mlx model dir may lack the jinja chat template (some
        # dirs ship it as a separate *.jinja file) -- verify the render actually carries a think
        # tag before trusting it; otherwise fall back to mlx_lm's loader.
        if getattr(candidate, "chat_template", None) and "<think>" in _render(candidate, False):
            tokenizer = candidate
    except Exception:
        tokenizer = None
    if tokenizer is None:
        from mlx_lm import load
        tokenizer = load(tok_path)[1]

    # Exercise the actual _chat_ids call path (tokenize=True) to confirm it runs end to end.
    ids_off = _chat_ids(tokenizer, "hi", add_generation_prompt=True, enable_thinking=False)
    ids_on = _chat_ids(tokenizer, "hi", add_generation_prompt=True, enable_thinking=True)
    assert ids_off and ids_on

    text_off = _render(tokenizer, enable_thinking=False)
    text_on = _render(tokenizer, enable_thinking=True)

    # enable_thinking=False -> a CLOSED empty think block trails the prompt.
    assert text_off.endswith("<think>\n\n</think>\n\n"), text_off[-60:]
    # enable_thinking=True -> an OPEN think tag (not closed) trails the prompt.
    assert text_on.endswith("<think>\n"), text_on[-60:]


# --- attention-sink floor (min_position / SINK_SKIP) ------------------------------------------

def test_sink_skip_constant():
    # Floor from the reference implementation this repo ports: early positions act as
    # attention sinks with atypical residual statistics and must not enter the Jacobian average.
    from jlens_mlx.corpus import SINK_SKIP
    assert SINK_SKIP == 16


def test_build_positions_min_position_floors_off_policy_content_start():
    ids = list(range(40))
    pos = build_positions(ids, 40, PositionMask(), content_start=5, content_end=30,
                          min_position=16)
    assert pos[0] == 16 and pos[-1] == 29


def test_build_positions_min_position_floors_on_policy_span():
    ids = list(range(40))  # completion spans [10, 39)
    pos = build_positions(ids, 10, PositionMask(), min_position=16)
    assert pos and pos[0] == 16 and all(p >= 16 for p in pos)


def test_build_positions_min_position_default_zero_keeps_old_behavior():
    ids = list(range(40))
    pos = build_positions(ids, 40, PositionMask(), content_start=5, content_end=30)
    assert pos[0] == 5


def test_build_positions_min_position_all_filtered_falls_back():
    # Everything below the floor: the [n-2] safety fallback must still fire (never empty).
    ids = list(range(12))
    pos = build_positions(ids, 12, PositionMask(), content_start=2, content_end=8,
                          min_position=16)
    assert pos == [10]


# --- resolve_on_policy_fraction (per-stratum on-policy override) ---------------------------------

def _strat(kind: str, frac=None) -> Stratum:
    return Stratum(hf_id="x", weight=0.5, kind=kind, on_policy_fraction=frac)


def test_resolve_on_policy_fraction_inherits_recipe_default_when_unset():
    r = Recipe(name="r", on_policy_fraction=0.6)
    assert resolve_on_policy_fraction(_strat("reasoning"), r) == 0.6  # None -> recipe default


def test_resolve_on_policy_fraction_stratum_override_wins():
    r = Recipe(name="r", on_policy_fraction=0.6)
    assert resolve_on_policy_fraction(_strat("safety", 1.0), r) == 1.0   # all-on-policy safety
    assert resolve_on_policy_fraction(_strat("reasoning", 0.5), r) == 0.5


def test_resolve_on_policy_fraction_override_zero_is_respected_not_treated_as_falsy():
    # 0.0 is a real value (human-text only), NOT "unset" -- must not fall through to the default.
    r = Recipe(name="r", on_policy_fraction=0.6)
    assert resolve_on_policy_fraction(_strat("control", 0.0), r) == 0.0
