"""Unit tests for the pure legibility classifier in verify.py (no model, no GPU, no tokenizer).

`_is_content_token` / `legibility_fraction` operate on already-decoded top-k strings, so these
run on CPU alongside a live fit. This is the regression coverage for the fidelity-gate-misleads
finding: on a served abliterated model, `fidelity_gate`'s final-logit agreement ranked a
degenerate near-final-layer readout ABOVE a semantically meaningful mid-band layer, because the
degenerate layer happened to match the model's own collapsed final output tokens. Legibility
scores a readout on its OWN top-k content, not against the (possibly degenerate) final logits --
these tests lock down that the motivating case comes out the RIGHT way round.

Run (from the heylook dir / venv):  uv run pytest <jlens-mlx>/tests/test_verify.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jlens_mlx.verify import _is_content_token, legibility_fraction  # noqa: E402


# --- _is_content_token -------------------------------------------------------------------------

def test_content_token_plain_word():
    assert _is_content_token("Paris") is True
    assert _is_content_token(" Paris") is True           # plain leading space stripped


def test_content_token_bpe_space_markers():
    assert _is_content_token("Ġcity") is True             # GPT-2/RoBERTa marker
    assert _is_content_token("▁city") is True             # SentencePiece marker


def test_content_token_contraction_and_hyphenation():
    assert _is_content_token("don't") is True
    assert _is_content_token("well-known") is True


def test_content_token_numbers():
    assert _is_content_token("42") is True
    assert _is_content_token("1990") is True


def test_content_token_non_latin_script():
    # CJK ideograph token, no hardcoded script list -- generalizes via str.isalpha().
    assert _is_content_token("東京") is True
    assert _is_content_token("Ġ東京") is True


def test_content_token_function_word():
    # Function words count as content per the stated rules (letters only, no punctuation).
    assert _is_content_token(" the") is True


def test_content_token_punctuation_runs_are_degenerate():
    for tok in ("__", "**", "___", "?.", "...).", '="."', " "):
        assert _is_content_token(tok) is False, tok


def test_content_token_empty_and_whitespace_only_are_degenerate():
    assert _is_content_token("") is False
    assert _is_content_token(" ") is False
    assert _is_content_token("Ġ") is False                # marker with nothing after it
    assert _is_content_token("\n") is False


def test_content_token_special_tokens_are_degenerate():
    assert _is_content_token("<|im_start|>") is False
    assert _is_content_token("<think>") is False
    assert _is_content_token("[INST]") is False


def test_content_token_leading_punctuation_is_degenerate():
    # Doesn't start with a letter/digit after stripping the marker/space -- ambiguous, so
    # conservative default (degenerate) applies.
    assert _is_content_token("-well") is False
    assert _is_content_token("'twas") is False


# --- legibility_fraction ------------------------------------------------------------------------

def test_legibility_fraction_all_content():
    frac = legibility_fraction([" Paris", " city", " France", " the"])
    assert frac == 1.0


def test_legibility_fraction_all_degenerate():
    frac = legibility_fraction([" __", "**", "___", "?.", " "])
    assert frac == 0.0


def test_legibility_fraction_mixed_matches_expected_ratio():
    # 2 of 4 are content -> exactly 0.5, not just "strictly between 0 and 1".
    tokens = [" Paris", "__", " city", "**"]
    assert legibility_fraction(tokens) == 0.5


def test_legibility_fraction_empty_list_is_nan():
    frac = legibility_fraction([])
    assert frac != frac                                   # nan != nan


def test_legibility_fraction_motivating_regression_case():
    # The case fidelity_gate got backwards: a band-5L-style readout with real content must
    # score HIGHER legibility than a near-final-layer-style readout of degenerate junk, even
    # though (in the real incident) the junk layer agreed better with the model's own
    # (degenerate) final logits.
    band_layer = legibility_fraction([" Paris", " city"])
    near_final_layer = legibility_fraction([" __", " ___"])
    assert band_layer > near_final_layer
    assert band_layer == 1.0 and near_final_layer == 0.0
