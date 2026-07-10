"""Verification: parity vs an oracle, the held-out fidelity gate, and lens diffing.

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


def parity_vs_oracle(lens, oracle) -> dict:
    """MLX apply vs a genuine jlens.apply() oracle: per-layer cosine + top-k
    overlap. Port migrated_from_scratch/mlx_apply*.py (the V1/V2 gate)."""
    raise NotImplementedError("port migrated_from_scratch/mlx_apply.py")


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
                  min_top1: float = 0.0, min_topk_agreement: float = 0.5) -> dict:
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
    identity_ok = None
    if target in per_layer:
        identity_ok = per_layer[target]["top1"] > 0.999 and per_layer[target]["kl"] < 1e-3
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
