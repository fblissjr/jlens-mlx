"""Fitting corpus: strata, recipes, position mask, on-policy builder.

Corpus choice is load-bearing — fitting a Jacobian lens averages the network's input→output
Jacobian over the corpus's activation distribution, exactly like quantization-calibration data
(see docs/DESIGN.md). So the corpus is a first-class, swappable, provenance-stamped config:
NO hardcoded WikiText, NO hardcoded "skip first 4 positions".

For our target — an abliterated instruct Qwen thinking model — the load-bearing design is an
OVER-WEIGHTED safety stratum + a matched-benign contrast + a WikiText control, so "what
abliteration did to the refusal direction" can be read as a DIFFERENCE of two lenses (fit a
harmful lens and a benign lens that differ only in harmfulness; the diff localizes the moved
direction). On-policy generation is essential: fit at the model's OWN assistant/<think> token
positions, not a teacher model's shipped traces.

Dataset ids/licenses below were verified on the HF Hub. `build_corpus` (the HF-loading +
on-policy sampling pipeline) is the next implementation step; the recipes here are the spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PositionMask:
    """Which token positions the Jacobian is averaged over. Role/sink-aware, not a fixed skip
    count (the "skip first N" heuristic is for raw-text BOS sinks; wrong under ChatML)."""

    include_assistant: bool = True
    include_think: bool = True
    include_user: bool = False
    drop_bos_sink: bool = True
    drop_role_tokens: bool = True
    predicate: object | None = None  # optional (token_id, position, role) -> bool


@dataclass
class Stratum:
    """One weighted source in a recipe. `weight`s within a recipe sum to ~1.0."""

    hf_id: str
    weight: float
    kind: str                       # chat | reasoning | safety | benign | creative | multilingual | control
    config: str | None = None       # HF dataset config/subset (e.g. JBB "harmful"/"benign")
    split: str = "train"
    prompt_field: str = ""          # which field holds the user prompt (loader-specific)
    license: str = ""
    gated: bool = False
    note: str = ""


@dataclass
class Recipe:
    """A named, reproducible fitting-corpus spec, stamped onto the lens sidecar as provenance
    (recipe + model SHA + position policy). Counts are PROMPTS, not tokens (~100-500 suffices)."""

    name: str
    strata: list[Stratum] = field(default_factory=list)
    n_prompts: int = 300
    seed: int = 0
    chat_templated: bool = True       # render through the model's own chat template
    on_policy_fraction: float = 0.6   # 0 = human text only; 0.5-0.7 for a chat/reasoning lens
    positions: PositionMask = field(default_factory=PositionMask)


@dataclass
class Corpus:
    """Materialized fitting corpus: tokenized prompts + per-prompt position masks."""

    recipe: Recipe
    # prompts: list[mx.array]; masks: list[mx.array] -- populated by build_corpus


# --- Concrete recipes (from the HF-verified dataset study) ------------------------------------

#: Primary recipe for the abliterated Qwen. Over-weighted safety (25% vs its ~1-3% natural
#: frequency) so the refusal circuitry is ACTIVE at estimation time — the single most important
#: departure from a WikiText lens.
HERETIC_QWEN = Recipe(
    name="heretic-qwen-chat-safety-v1",
    n_prompts=300,
    on_policy_fraction=0.6,
    strata=[
        # 30% general chat / real user turns — on-distribution backbone
        Stratum("allenai/WildChat-1M", 0.30, "chat", license="ODC-BY",
                note="real in-the-wild convs; keep raw user prompts internal (ODC-BY)"),
        # 25% math+code reasoning with CoT — drives the <think> positions (use PROMPTS, regen on-policy)
        Stratum("open-r1/OpenR1-Math-220k", 0.10, "reasoning", license="Apache-2.0",
                note="math; shipped traces are R1's -> use prompts, regenerate on-policy"),
        Stratum("open-thoughts/OpenThoughts-114k", 0.08, "reasoning", license="Apache-2.0"),
        Stratum("nvidia/OpenCodeReasoning", 0.07, "reasoning", config="python", license="CC-BY-4.0"),
        # 25% safety-salient (OVER-WEIGHTED) — the differentiating stratum
        Stratum("JailbreakBench/JBB-Behaviors", 0.13, "safety", config="harmful",
                license="MIT", note="MIT/ungated; has a matched benign config -> difference-of-Jacobians"),
        Stratum("mlabonne/harmful_behaviors", 0.12, "safety", license="AdvBench-derived (MIT)",
                note="matches mlabonne's abliteration recipe; pair with mlabonne/harmless_alpaca"),
        # 12% creative / roleplay — uncensored deployment regime
        Stratum("Gryphe/Sonnet3.5-SlimOrcaDedupCleaned", 0.12, "creative", license="MIT",
                note="uncensored responses (no refusals) -> matches the abliterated regime"),
        # 8% multilingual breadth
        Stratum("CohereLabs/aya_dataset", 0.08, "multilingual", license="Apache-2.0"),
    ],
)

#: Matched harmful/benign pair — fit two lenses that differ ONLY in harmfulness; their diff
#: localizes the direction abliteration moved. Same prompts otherwise.
SAFETY_HARMFUL = Recipe(
    name="safety-harmful-v1", n_prompts=200, on_policy_fraction=0.6,
    strata=[
        Stratum("JailbreakBench/JBB-Behaviors", 0.5, "safety", config="harmful", license="MIT"),
        Stratum("mlabonne/harmful_behaviors", 0.5, "safety", license="AdvBench-derived (MIT)"),
    ],
)
SAFETY_BENIGN = Recipe(
    name="safety-benign-v1", n_prompts=200, on_policy_fraction=0.6,
    strata=[
        Stratum("JailbreakBench/JBB-Behaviors", 0.5, "benign", config="benign", license="MIT"),
        Stratum("mlabonne/harmless_alpaca", 0.5, "benign", license="AdvBench-derived (MIT)"),
    ],
)

#: The control arm — the reference literature's corpus (Neuronpedia's "Salesforce-wikitext",
#: Anthropic/solarkyle's WikiText/C4). Fit it BOTH raw (chat_templated=False, matches the
#: literature) and chat-templated (to isolate template-effect from content-effect). The headline
#: result: the stratified lens diverges from the WikiText lens specifically where abliteration acted.
WIKITEXT_CONTROL = Recipe(
    name="wikitext-control-v1", n_prompts=300, on_policy_fraction=0.0, chat_templated=False,
    strata=[Stratum("Salesforce/wikitext", 1.0, "control", config="wikitext-103-raw-v1",
                    license="CC-BY-SA-3.0", note="raw prose baseline; also allenai/c4 (ODC-BY)")],
)


def build_corpus(model, processor, recipe: Recipe) -> Corpus:
    """Render `recipe` into a Corpus (NEXT implementation step; this is the contract):

    1. For each Stratum, load `hf_id`[`config`/`split`], take a weighted sample so the strata
       hit `recipe.n_prompts` at the recipe's weights (seeded by `recipe.seed`).
    2. If `recipe.chat_templated`, wrap each prompt in the model's own chat template.
    3. If `recipe.on_policy_fraction > 0`, sample that fraction of completions from `model`
       itself (including <think>...</think>) rather than using the datasets' shipped traces.
    4. Apply `recipe.positions` (assistant/think tokens; drop BOS/sink/role) to build per-prompt
       masks, and tokenize. Return prompts + masks.

    Gated datasets (LMSYS/AdvBench/wildjailbreak) need an HF token; the recipes above deliberately
    prefer ungated MIT/ODC-BY substitutes. Keep raw user prompts internal (ODC-BY attribution).
    """
    raise NotImplementedError(
        "build_corpus: HF dataset loading + chat-templating + on-policy sampling + position mask"
    )
