"""Verification: the held-out fidelity gate and lens diffing. (Apply-vs-oracle parity is
exercised by the scripts/check_*.py verifiers -- gpt2/gemma2/qwen3_5 synthetic + chain==direct
-- not by a function here.)

A lens is not saved unless the fidelity gate passes. See docs/DESIGN.md.

Convention shared by the gate + diff: a lens readout at source layer ``l`` is
``adapter.unembed(J_l @ acts[l])`` -- the residual at ``l`` transported into the
final-layer basis and run through the model's REAL head (+ softcap). For the identity
lens (``l == target``) this is exactly the model's true logits, so the gate's identity
layer is a correctness tripwire (top-1 == 1.0, KL ~ 0), not just a quality score.
"""
from __future__ import annotations

import mlx.core as mx

from .capture import ModelAdapter, capture_residuals
from .fit import valid_positions


def _true_logits(ad: ModelAdapter, ids: list[int]) -> mx.array:
    """The model's real next-token logits for ``ids`` -> ``[S, vocab]`` (post final
    norm + head + softcap, exactly the distribution the lens approximates)."""
    return ad.logits(mx.array([list(ids)]))[0]


def _topk_set_overlap(a_logits: mx.array, b_logits: mx.array, k: int) -> float:
    """Mean over positions of |topk(a) ∩ topk(b)| / k. a,b: [n_pos, vocab]."""
    ia = mx.argsort(-a_logits, axis=-1)[:, :k]        # [n_pos, k]
    ib = mx.argsort(-b_logits, axis=-1)[:, :k]
    import numpy as np
    ia_np, ib_np = np.asarray(ia), np.asarray(ib)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ia_np, ib_np)]))


def fidelity_gate(model, lens, held_out, *, tokenize, adapter: ModelAdapter | None = None,
                  positions=None, skip_first: int = 4, top_k: int = 10,
                  min_top1: float = 0.0, min_topk_agreement: float = 0.5,
                  identity_max_kl: float = 5e-2) -> dict:
    """Per-layer agreement between the lens readout and the model's TRUE logits, on
    HELD-OUT prompts. Never grade a lens on its fit corpus.

    Metrics per source layer, averaged over held-out prompts x positions:
      - ``top1``: fraction where argmax(lens) == argmax(true).
      - ``topk``: mean |topk(lens) ∩ topk(true)| / k.
      - ``kl``: mean KL(softmax(true) || softmax(lens)) in nats.

    Args:
        model: mlx-lm model.
        lens: a :class:`~jlens_mlx.lens.JSpaceLens`.
        held_out: prompts NOT in the fit corpus (each accepted by ``tokenize``).
        tokenize: ``prompt -> list[int]``.
        positions: token positions to grade (default: ``valid_positions`` per prompt).
        top_k: overlap set size.
        min_top1 / min_topk_agreement: thresholds for ``passed`` (applied to the WORST
            non-identity layer -- the identity/target layer is excluded from the pass
            decision but reported; it should read ~1.0 / ~0 as a correctness check).

    Returns:
        ``{"per_layer": {l: {top1, topk, kl}}, "passed": bool, "worst_layer": int,
           "n_prompts": int, "identity_ok": bool|None}``.
    """
    ad = adapter or ModelAdapter(model)
    layers = list(lens.source_layers)
    target = int(lens.meta.get("target_layer", ad.n_layers - 1))
    agg = {l: {"top1": [], "topk": [], "kl": []} for l in layers}

    for prompt in held_out:
        ids = tokenize(prompt)
        pos = positions if positions is not None else valid_positions(len(ids), skip_first)
        true = _true_logits(ad, ids)                                   # [S, vocab]
        idx = mx.array([p % len(ids) for p in pos])
        true_p = true[idx]                                             # [n_pos, vocab]
        true_lsm = true_p - mx.logsumexp(true_p, axis=-1, keepdims=True)
        true_prob = mx.exp(true_lsm)
        res = capture_residuals(model, ids, layers, adapter=ad)
        lens_out = lens.apply(ad, res, positions=pos, layers=layers)   # {l: [n_pos, vocab]}
        for l in layers:
            ll = lens_out[l].astype(mx.float32)
            t1 = float((mx.argmax(ll, -1) == mx.argmax(true_p, -1)).astype(mx.float32).mean().item())
            lens_lsm = ll - mx.logsumexp(ll, axis=-1, keepdims=True)
            kl = float((true_prob * (true_lsm - lens_lsm)).sum(-1).mean().item())
            agg[l]["top1"].append(t1)
            agg[l]["topk"].append(_topk_set_overlap(true_p, ll, top_k))
            agg[l]["kl"].append(kl)

    per_layer = {l: {m: (sum(v) / len(v) if v else float("nan")) for m, v in d.items()}
                 for l, d in agg.items()}
    graded = [l for l in layers if l != target]
    worst = min(graded, key=lambda l: per_layer[l]["topk"]) if graded else target
    passed = bool(graded) and all(
        per_layer[l]["top1"] >= min_top1 and per_layer[l]["topk"] >= min_topk_agreement
        for l in graded)
    # Identity/target-layer tripwire: does the lens reproduce the model's TRUE logits?
    # Graded on KL (the distribution match), NOT top-1 exactness -- on a QUANTIZED model,
    # fp32-vs-native rounding swaps near-tied tokens so top-1 is <1.0 even for a correct
    # apply path (e.g. 8-bit 27B: identity top1~0.97, top10~0.99, KL~0.006). A broken apply
    # path gives a LARGE KL, which this catches; on fp32 (synthetic) KL is ~0.
    identity_ok = None
    if target in per_layer:
        identity_ok = per_layer[target]["kl"] < identity_max_kl
    return {"per_layer": per_layer, "passed": passed, "worst_layer": worst,
            "n_prompts": len(held_out), "identity_ok": identity_ok}


def diff(model, lens_a, lens_b, prompts, *, tokenize, adapter: ModelAdapter | None = None,
         positions=None, skip_first: int = 4, top_k: int = 15) -> dict:
    """Diff two lenses' readouts on ``prompts`` -- e.g. stock-Qwen vs abliterated. For
    the abliteration study, the diff IS the finding: which tokens each layer's lens
    moves toward/away relative to the other lens.

    Both lenses must share ``source_layers`` and ``d_model`` and are applied against the
    SAME model's activations (the point is to compare the two transport matrices, not two
    models). Per layer, returns the mean (over prompts x positions) lens-logit delta
    ``B - A`` and its top movers.

    Returns:
        ``{"per_layer": {l: {"top_up": [(token_id, delta)...], "top_down": [...],
           "l2": float}}, "source_layers": [...], "n": int}``.
    """
    import numpy as np
    ad = adapter or ModelAdapter(model)
    layers = sorted(set(lens_a.source_layers) & set(lens_b.source_layers))
    if not layers:
        raise ValueError("lenses share no source layers")
    sums = {l: None for l in layers}
    counts = {l: 0 for l in layers}

    for prompt in prompts:
        ids = tokenize(prompt)
        pos = positions if positions is not None else valid_positions(len(ids), skip_first)
        res = capture_residuals(model, ids, layers, adapter=ad)
        oa = lens_a.apply(ad, res, positions=pos, layers=layers)
        ob = lens_b.apply(ad, res, positions=pos, layers=layers)
        for l in layers:
            d = (ob[l].astype(mx.float32) - oa[l].astype(mx.float32))  # [n_pos, vocab]
            s = d.sum(axis=0)
            sums[l] = s if sums[l] is None else sums[l] + s
            counts[l] += d.shape[0]

    per_layer = {}
    for l in layers:
        mean = np.asarray(sums[l] / max(counts[l], 1))                 # [vocab]
        order = np.argsort(-mean)
        up = [(int(i), float(mean[i])) for i in order[:top_k]]
        down = [(int(i), float(mean[i])) for i in order[::-1][:top_k]]
        per_layer[l] = {"top_up": up, "top_down": down,
                        "l2": float(np.linalg.norm(mean))}
    return {"per_layer": per_layer, "source_layers": layers, "n": sum(counts.values())}


# --- legibility: a replacement ranking signal for band layers ---------------------------------
#
# `fidelity_gate` scores a layer's readout by agreement with the model's TRUE final logits.
# That is the right question for the identity/target-layer tripwire, but the WRONG question for
# ranking band (early/mid) layers: a lens is supposed to diverge from the final distribution at
# those layers -- that divergence is the useful signal, not an error. Empirically this backfires
# hard on a served abliterated model, where the true final logits are themselves near-degenerate
# (softmax collapsed onto punctuation/formatting tokens): a band layer whose top-k happened to
# also be junk (' __', '**', '___') scored HIGH final-logit agreement, while a band layer with
# real semantic content (' Paris', ' city') scored LOW, because it correctly diverged from the
# degenerate final distribution. `legibility_report` below scores a layer's readout on its OWN
# terms -- does its top-k decode to real words? -- rather than against the final logits.

def _is_content_token(tok: str) -> bool:
    """Classify one ALREADY-DECODED top-k token string as CONTENT vs DEGENERATE.

    No model/tokenizer access here -- pure string logic, so it is trivially unit-testable.
    Ambiguous cases are classified DEGENERATE (conservative default): this is a legibility
    signal, not a linguistic parser, and a false "content" is more misleading than a false
    "degenerate" when the whole point is separating meaningful readouts from junk.

    Rules:
      1. Strip a leading BPE space-marker if present -- GPT-2/RoBERTa-style ``Ġ``, SentencePiece
         ``▁``, or a plain leading space (possibly repeated). What decoded top-k strings look
         like depends on the tokenizer; all three show up in practice.
      2. Empty after stripping (pure whitespace/marker token) -> DEGENERATE.
      3. A bracket-wrapped special/structural token -- ``<|im_start|>``, ``<think>``, ``[INST]``
         -- (starts with ``<``/``[`` and ends with ``>``/``]``) -> DEGENERATE. This is the special-
         token marker convention already used for chat scaffolding elsewhere in this codebase
         (see ``corpus.decode_corpus``'s ``<|im_start|>assistant`` / ``<think>`` / ``<|im_end|>``
         examples) -- no new convention invented here.
      4. Otherwise: CONTENT iff the first character is a letter or digit (any script/unicode
         category -- ``str.isalpha()``/``str.isdigit()`` generalize past Latin, so CJK and other
         ideograph tokens are handled without a hardcoded script list) AND every remaining
         character is alphanumeric or one of ``'``/``-`` (contractions, hyphenated words).
         Anything else -- punctuation/symbol runs (``__``, ``**``, ``___``, ``?.``, ``...).``,
         ``="``), pure formatting (newlines/tabs) -- is DEGENERATE.
    """
    if not tok:
        return False
    s = tok
    if s[0] in ("Ġ", "▁"):
        s = s[1:]
    elif s[0] == " ":
        s = s.lstrip(" ")
    if not s:
        return False                                    # nothing left after the marker/space
    if s[0] in ("<", "[") and s[-1] in (">", "]"):
        return False                                     # bracket-wrapped special/structural token
    if not (s[0].isalpha() or s[0].isdigit()):
        return False
    return all(ch.isalnum() or ch in ("'", "-") for ch in s[1:])


def legibility_fraction(tokens: list[str]) -> float:
    """Fraction of ``tokens`` (already-decoded top-k strings, one readout's worth) that are
    CONTENT per :func:`_is_content_token`. ``[]`` -> ``nan`` (no positions graded, not 0)."""
    if not tokens:
        return float("nan")
    return sum(1 for t in tokens if _is_content_token(t)) / len(tokens)


def legibility_report(model, lens, held_out, *, tokenize, tokenizer, adapter: ModelAdapter | None = None,
                      positions=None, skip_first: int = 4, top_k: int = 10) -> dict:
    """Per-layer LEGIBILITY of the lens readout, on HELD-OUT prompts -- the ranking signal to use
    INSTEAD OF ``fidelity_gate``'s final-logit agreement when judging band (non-identity) layers.
    See the module-level note above this function for why.

    Metrics per source layer, averaged over held-out prompts (x positions within each prompt):
      - ``legibility``: mean :func:`legibility_fraction` of the layer's top-k decoded tokens --
        high = the readout's top-k are real words, low = punctuation/symbol junk.
      - ``entropy``: mean entropy (nats) of the layer's own output distribution -- reported
        alongside legibility for context (a collapsed, low-entropy distribution over junk tokens
        reads very differently from a confident, low-entropy distribution over real content).

    This does NOT replace ``fidelity_gate``: the identity-layer KL tripwire (does the lens
    reproduce the model's TRUE logits at the target layer) is still the correctness check and
    lives there, unchanged. This function only concerns itself with band-layer QUALITY.

    Args:
        model: mlx-lm model.
        lens: a :class:`~jlens_mlx.lens.JSpaceLens`.
        held_out: prompts NOT in the fit corpus.
        tokenize: ``prompt -> list[int]``.
        tokenizer: exposes ``decode(ids) -> str``, used to turn each top-k token id into the
            decoded string ``_is_content_token`` classifies (mirrors ``corpus.decode_corpus``'s
            ``tokenizer.decode`` usage).
        positions: token positions to grade (default: ``valid_positions`` per prompt).
        top_k: readout size to classify per position.

    Returns:
        ``{"per_layer": {l: {"legibility": float, "entropy": float}},
           "ranked": [layer, ...] sorted by legibility DESCENDING, "n_prompts": int}``.
        ``ranked`` is a distinct selection path from ``fidelity_gate``'s ``worst_layer``
        (ascending final-logit topk agreement) -- use ``ranked`` to pick a meaningful band layer,
        not ``fidelity_gate``'s output.
    """
    import numpy as np
    ad = adapter or ModelAdapter(model)
    layers = list(lens.source_layers)
    agg = {l: {"legibility": [], "entropy": []} for l in layers}

    for prompt in held_out:
        ids = tokenize(prompt)
        pos = positions if positions is not None else valid_positions(len(ids), skip_first)
        res = capture_residuals(model, ids, layers, adapter=ad)
        lens_out = lens.apply(ad, res, positions=pos, layers=layers)   # {l: [n_pos, vocab]}
        for l in layers:
            ll = lens_out[l].astype(mx.float32)
            lsm = ll - mx.logsumexp(ll, axis=-1, keepdims=True)
            prob = mx.exp(lsm)
            ent = float((-(prob * lsm).sum(-1)).mean().item())
            topk_ids = np.asarray(mx.argsort(-ll, axis=-1)[:, :top_k])  # [n_pos, top_k]
            fracs = [legibility_fraction([tokenizer.decode([int(t)]) for t in row])
                     for row in topk_ids]
            agg[l]["legibility"].append(sum(fracs) / len(fracs) if fracs else float("nan"))
            agg[l]["entropy"].append(ent)

    per_layer = {l: {m: (sum(v) / len(v) if v else float("nan")) for m, v in d.items()}
                 for l, d in agg.items()}
    ranked = sorted(layers, key=lambda l: per_layer[l]["legibility"], reverse=True)
    return {"per_layer": per_layer, "ranked": ranked, "n_prompts": len(held_out)}
