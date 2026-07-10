"""Offline preview of a fitting corpus -- validate composition BEFORE the GPU run.

Builds a recipe with `model=None` (human-text only, no on-policy generation, no GPU/server)
and prints per-stratum counts, token-length stats, and position coverage. Use this to sanity
the corpus (did each stratum load? are prompts non-empty? is the mix right?) before committing
hours to the on-policy + fit run. The mlx-lm tokenizer (jinja chat template) is required.

Run (from the heylook dir, heylook venv):
  JLENS_MODEL=<served model dir> JLENS_RECIPE=ABLITERATED_QWEN \
    JLENS_N=40 uv run python <jlens-mlx>/scripts/build_corpus_preview.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlx_lm.tokenizer_utils import load as load_tokenizer  # noqa: E402

import jlens_mlx.corpus as C  # noqa: E402


def main() -> int:
    model_path = os.environ.get("JLENS_MODEL")
    if not model_path:
        print("set JLENS_MODEL to the served model dir (for its jinja chat template)")
        return 2
    recipe_name = os.environ.get("JLENS_RECIPE", "ABLITERATED_QWEN")
    recipe = getattr(C, recipe_name, None)
    if not isinstance(recipe, C.Recipe):
        print(f"unknown recipe {recipe_name!r}; available: "
              f"{[n for n in dir(C) if isinstance(getattr(C, n), C.Recipe)]}")
        return 2
    if os.environ.get("JLENS_N"):
        recipe = C.Recipe(name=recipe.name, strata=recipe.strata, n_prompts=int(os.environ["JLENS_N"]),
                          seed=recipe.seed, chat_templated=recipe.chat_templated,
                          on_policy_fraction=recipe.on_policy_fraction, positions=recipe.positions)

    tok = load_tokenizer(Path(model_path))
    print(f"recipe={recipe.name} n_prompts={recipe.n_prompts} "
          f"on_policy_fraction={recipe.on_policy_fraction} (preview: human-text only, model=None)")
    print(f"weighted counts: {C._weighted_counts(recipe)}")

    corpus = C.build_corpus(None, tok, recipe)   # model=None -> offline, no GPU

    by_kind: dict[str, list[int]] = {}
    for it in corpus.items:
        by_kind.setdefault(it.stratum, []).append(len(it.ids))
    print(f"\nbuilt {len(corpus.items)} items:")
    for kind, lens in sorted(by_kind.items()):
        print(f"  {kind:12s} n={len(lens):3d}  tok_len min/median/max = "
              f"{min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
    pos = [len(it.positions) for it in corpus.items]
    empty = sum(1 for p in pos if p == 0)
    print(f"\nposition coverage: mean {sum(pos)/max(len(pos),1):.1f} pos/prompt, "
          f"{empty} empty-mask prompt(s)")
    print(f"provenance: {corpus.provenance}")
    print("\nNOTE: on-policy generation (the assistant/think span) needs the model + GPU -- "
          "run the real fit with the server stopped. This preview only validates loading + "
          "templating + human-text masks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
