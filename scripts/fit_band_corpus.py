"""First corpus-driven, band-targeted own-fit on the served abliterated Qwen3.5-27B.

The production path end to end: build a small real (on-policy) corpus with build_corpus, fit
`J_l` over the PRODUCT BAND (layers the server actually reads, features.band_layers = [0.25,0.75)
= 16..47) with fit_corpus, grade on held-out, and save with corpus + fit provenance.

SCOPE NOTE: band layers are the deep end (long tails); even dim-batched, a dense band fit over
100+ prompts is a multi-day job. Defaults here are a SCOPED proof — a few in-band layers over a
small corpus (~1-1.5h) — that validates the whole pipeline on real weights and yields an in-band
(product-relevant) lens, honestly provisional. The dense overnight run overrides JLENS_LAYERS +
JLENS_N. Metal-gated; run with the heylook server STOPPED (on-policy generation + the fit own the
GPU).

Env: JLENS_MODEL (req), JLENS_LAYERS (default a shallow-band sample), JLENS_N (corpus prompts,
default 6), JLENS_CHUNK (128), JLENS_ONPOLICY_TOKENS (48), JLENS_OUT.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jlens_mlx.corpus as C  # noqa: E402
from jlens_mlx import lens as lenslib, verify  # noqa: E402
from jlens_mlx.capture import ModelAdapter  # noqa: E402
from jlens_mlx.fit import _model_type, fit_corpus  # noqa: E402

# A compact, ungated, safety-weighted recipe for a FIRST band fit (the abliteration signal lives
# in the safety strata; a matched harmful/benign pair enables the later difference-of-Jacobians).
# Small n so a deep-band fit finishes in ~1-1.5h; the full ABLITERATED_QWEN recipe is the overnight run.
BAND_BOOTSTRAP = C.Recipe(
    name="band-bootstrap-v1", n_prompts=6, on_policy_fraction=0.6,
    strata=[
        C.Stratum("JailbreakBench/JBB-Behaviors", 0.34, "safety", config="behaviors",
                  split="harmful", license="MIT"),
        C.Stratum("JailbreakBench/JBB-Behaviors", 0.33, "benign", config="behaviors",
                  split="benign", license="MIT"),
        C.Stratum("open-r1/OpenR1-Math-220k", 0.33, "reasoning", license="Apache-2.0"),
    ],
)

# Held-out grading prompts (NOT in the corpus): neutral + reasoning + safety-adjacent.
HELD_OUT = [
    [{"role": "user", "content": "Suggest a simple recipe for a weeknight dinner."}],
    [{"role": "user", "content": "What household chemicals should never be mixed, and why?"}],
]


def _band(n: int) -> tuple[int, int]:
    return int(n * 0.25), int(n * 0.75)


def main() -> int:
    model_path = os.environ.get("JLENS_MODEL")
    if not model_path:
        print("set JLENS_MODEL to the served model dir")
        return 2

    t0 = time.perf_counter()
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    ad = ModelAdapter(model)
    n = ad.n_layers
    D = ad.layers[0].input_layernorm.weight.shape[0]
    mt = _model_type(ad)
    b_lo, b_hi = _band(n)
    print(f"load {time.perf_counter()-t0:.1f}s  n_layers={n} d_model={D} band=[{b_lo},{b_hi}) "
          f"arch={mt} peak={mx.get_peak_memory()/2**30:.1f}GB", flush=True)

    layers = ([int(x) for x in os.environ["JLENS_LAYERS"].split(",")]
              if os.environ.get("JLENS_LAYERS") else [40, 44, 47])  # shallow-band sample
    out_of_band = [l for l in layers if not (b_lo <= l < b_hi)]
    if out_of_band:
        print(f"  WARNING: {out_of_band} outside band [{b_lo},{b_hi}) -- the product won't read them",
              flush=True)
    chunk = int(os.environ.get("JLENS_CHUNK", 128))
    target = n - 1

    recipe = BAND_BOOTSTRAP
    if os.environ.get("JLENS_N"):
        recipe = C.Recipe(name=recipe.name, strata=recipe.strata, n_prompts=int(os.environ["JLENS_N"]),
                          seed=recipe.seed, chat_templated=recipe.chat_templated,
                          on_policy_fraction=recipe.on_policy_fraction, positions=recipe.positions)

    # Checkpoint dir: the serialized corpus + running J_sum live here so a killed fit RESUMES
    # (skips on-policy generation + completed items) instead of losing everything.
    out = os.environ.get("JLENS_OUT") or str(ROOT / "out" / f"band-{len(layers)}L")
    ckpt_dir = os.path.join(out, "ckpt")
    corpus_json = os.path.join(ckpt_dir, "corpus.json")

    if os.path.exists(corpus_json):
        corpus = C.Corpus.from_json(corpus_json)
        print(f"RESUME: loaded {len(corpus.items)} corpus items from {corpus_json} "
              f"(skipping build + on-policy generation)", flush=True)
    else:
        print(f"building corpus ({recipe.name}, n={recipe.n_prompts}, "
              f"on_policy={recipe.on_policy_fraction}) -- streaming + on-policy generation...",
              flush=True)
        tc = time.perf_counter()
        # max_seq_len bounds each item so no single fit outlives the session/checkpoint window
        # (over-long prompts are dropped, not truncated). Conservative default; tune per model.
        max_seq_len = int(os.environ.get("JLENS_MAX_SEQ_LEN", 512))
        corpus = C.build_corpus(model, tokenizer, recipe,
                                on_policy_max_tokens=int(os.environ.get("JLENS_ONPOLICY_TOKENS", 48)),
                                max_seq_len=max_seq_len)
        corpus.to_json(corpus_json)
        print(f"  corpus: {len(corpus.items)} items in {time.perf_counter()-tc:.1f}s  "
              f"{corpus.provenance['strata']}  on_policy={sum(it.on_policy for it in corpus.items)} "
              f"  dropped_over_len={corpus.provenance['dropped_over_len']} (max_seq_len={max_seq_len}) "
              f"(saved -> {corpus_json})", flush=True)
    tok_lens = [len(it.ids) for it in corpus.items]
    pos_lens = [len(it.positions) for it in corpus.items]
    print(f"  tok_len min/med/max={min(tok_lens)}/{sorted(tok_lens)[len(tok_lens)//2]}/{max(tok_lens)}"
          f"  pos/prompt mean={sum(pos_lens)/len(pos_lens):.1f}", flush=True)

    print(f"fitting band layers {layers} (target={target}, chunk={chunk}) over the corpus...",
          flush=True)

    def _progress(p):
        if p.get("resumed") is True:
            print(f"  RESUMED from checkpoint: {p['done']} item(s) done; continuing at "
                  f"item {p['next_idx']+1}/{p['n_total']}", flush=True)
            return
        if p.get("resumed") is False:
            print(f"  checkpoint not resumed: {p.get('reason')}", flush=True)
            return
        if p["skipped"]:
            print(f"  item {p['i']+1}/{p['n_total']}: SKIPPED (no usable positions)", flush=True)
            return
        eta_min = (p["eta_secs"] or 0) / 60
        print(f"  item {p['i']+1}/{p['n_total']} (on_policy={p['on_policy']} seq={p['seq_len']} "
              f"pos={p['n_pos']}): {p['secs']:.0f}s  |  elapsed {p['elapsed']/60:.1f}m  "
              f"eta ~{eta_min:.0f}m", flush=True)

    tf = time.perf_counter()
    if os.environ.get("JLENS_FINALIZE"):
        # Finalize from the checkpoint WITHOUT fitting remaining items -- for when a single item
        # is longer than the session window (per-item checkpointing can't split one item). Averages
        # the banked J_sum over the items already done. The dropped items are logged for honesty.
        from jlens_mlx.fit import _ckpt_load
        jsum, meta = _ckpt_load(ckpt_dir)
        if jsum is None:
            print("JLENS_FINALIZE set but no checkpoint found"); return 2
        n_items = int(meta["n_done"])
        done_idx = int(meta["next_idx"])
        jacobians = {l: jsum[l] / n_items for l in layers}
        dropped = len(corpus.items) - done_idx
        print(f"FINALIZE: averaging banked J_sum over {n_items} completed items "
              f"(dropping {dropped} un-fit item(s) {list(range(done_idx, len(corpus.items)))} -- "
              f"too long for the session window; see the length-cap note)", flush=True)
    else:
        jacobians, n_items = fit_corpus(model, corpus, source_layers=layers, adapter=ad,
                                        target_layer=target, chunk_size=chunk, progress=_progress,
                                        checkpoint_dir=ckpt_dir)
    mx.eval(list(jacobians.values()))
    print(f"fit {time.perf_counter()-tf:.1f}s over {n_items} items  "
          f"peak={mx.get_peak_memory()/2**30:.1f}GB", flush=True)
    for l in sorted(jacobians):
        J = jacobians[l]
        fro = float(mx.linalg.norm(J).item())
        print(f"  J_{l}: ||J||/sqrt(D)={fro/(D**0.5):.4f} finite={bool(mx.all(mx.isfinite(J)).item())}",
              flush=True)

    # Held-out fidelity gate (+ identity tripwire).
    grade = dict(jacobians)
    grade[target] = mx.eye(D, dtype=mx.float32)
    gl = sorted(set(layers) | {target})
    glens = lenslib.JSpaceLens(grade, gl, D, softcap=ad.softcap, meta={"target_layer": target})

    def tok(m):
        return list(tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=True))

    print("fidelity gate (held-out)...", flush=True)
    rep = verify.fidelity_gate(model, glens, HELD_OUT, tokenize=tok, adapter=ad,
                               skip_first=4, top_k=10, min_topk_agreement=0.0)
    for l in sorted(rep["per_layer"]):
        m = rep["per_layer"][l]
        tag = " (identity)" if l == target else (" [in-band]" if b_lo <= l < b_hi else "")
        print(f"  J_{l}{tag}: top1={m['top1']:.3f} top10={m['topk']:.3f} kl={m['kl']:.3f}", flush=True)
    print(f"  identity_ok={rep['identity_ok']}", flush=True)

    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        sha = ""
    model_id = os.path.basename(model_path.rstrip("/"))
    sidecar = {
        "source_layers": sorted(layers), "d_model": D, "final_logit_softcapping": ad.softcap,
        "target_layer": target, "band": [b_lo, b_hi],
        "in_band_layers": sorted(l for l in layers if b_lo <= l < b_hi),
        "chunk_size": chunk, "fit_kind": "band-corpus",
        "model_id": model_id, "hf_model_name": model_id, "fit_source": "jlens-mlx own-fit",
        "fit_date": datetime.date.today().isoformat(), "jlens_git_sha": sha, "arch": mt,
        "corpus": corpus.provenance,
        "fidelity": {str(l): rep["per_layer"][l] for l in layers if l in rep["per_layer"]},
        "fidelity_identity_ok": rep["identity_ok"],
        "note": "first band-targeted corpus fit; scoped proof (small corpus) -- dense band fit is the overnight run",
    }
    lenslib.save(jacobians, sidecar, out)
    print(f"saved lens -> {out}", flush=True)
    print(json.dumps({"layers": sorted(layers), "n_items": n_items,
                      "identity_ok": rep["identity_ok"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
