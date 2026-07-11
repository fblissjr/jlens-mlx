"""Decode a saved corpus.json back to readable text (LOCAL-ONLY inspection).

Renders each item's tokens with special tokens SHOWN, splitting the on-policy prompt from the
model's generated response. The output contains raw dataset prompts + the model's own completions
(incl. the safety strata / whatever the abliterated model generated) — keep it local, don't
commit or share. `fit_band_corpus.py` now writes this by default on a fresh build; this standalone
covers corpora built before that (or a re-decode).

Env: JLENS_MODEL (for the tokenizer) + one of JLENS_CORPUS (path to a corpus.json) or JLENS_OUT
(uses <JLENS_OUT>/ckpt/corpus.json). Writes <corpus dir>/corpus_decoded.md.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jlens_mlx.corpus import Corpus, decode_corpus  # noqa: E402


def _load_tokenizer(model_path: str):
    """Decode only needs the vocab, not the chat template — try the light AutoTokenizer first,
    fall back to mlx-lm's loader (which loads model weights too, slower)."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path)
    except Exception:
        from mlx_lm import load
        return load(model_path)[1]


def main() -> int:
    model_path = os.environ.get("JLENS_MODEL")
    corpus_json = os.environ.get("JLENS_CORPUS")
    if not corpus_json and os.environ.get("JLENS_OUT"):
        corpus_json = os.path.join(os.environ["JLENS_OUT"], "ckpt", "corpus.json")
    if not (model_path and corpus_json):
        print("set JLENS_MODEL and one of JLENS_CORPUS (a corpus.json) or JLENS_OUT")
        return 2
    if not os.path.exists(corpus_json):
        print(f"no corpus.json at {corpus_json}")
        return 2

    corpus = Corpus.from_json(corpus_json)
    tok = _load_tokenizer(model_path)
    out_path = os.path.join(os.path.dirname(corpus_json), "corpus_decoded.md")
    Path(out_path).write_text(decode_corpus(corpus, tok))
    print(f"decoded {len(corpus.items)} items -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
