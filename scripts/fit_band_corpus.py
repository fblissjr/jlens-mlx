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
default 6), JLENS_CHUNK (128), JLENS_ONPOLICY_TOKENS (48), JLENS_OUT, JLENS_MAX_SEQ_LEN (512),
JLENS_ALLOW_DEGENERATE (override the corpus diversity gate), JLENS_FINALIZE (finalize from an
existing checkpoint without fitting remaining items), JLENS_WATCHDOG_S (hang watchdog: arms
`faulthandler.dump_traceback_later`, re-armed on every progress heartbeat so a wedged process >30
min with no chunk progress dumps thread stacks while still alive; default 1800, 0 disables).

Exit codes: 0 success; 2 config error (no JLENS_MODEL, or JLENS_FINALIZE with no checkpoint
found); 3 corpus diversity gate failed (degenerate/boilerplate-heavy corpus -- see
JLENS_ALLOW_DEGENERATE); any OTHER code (including death by signal, e.g. a native MLX
SIGABRT/SIGSEGV) is unexpected.

For a long/overnight run, wrap this in scripts/fit_band_supervisor.sh, which restarts on any
UNEXPECTED exit (checkpoint/resume makes each restart cheap) but stops on 0/2/3:

    nohup ./scripts/fit_band_supervisor.sh > out/band-fit.log 2>&1 &
"""
from __future__ import annotations

import datetime
import faulthandler
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
from jlens_mlx.fit import _hms, _model_type, fit_corpus  # noqa: E402

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


def _p(msg: str, **kw) -> None:
    """print() with an `[HH:MM:SS]` wall-clock prefix + flush=True (belt+braces vs any print()
    that forgot it -- silent multi-hour stretches were exactly the visibility bug report)."""
    kw.setdefault("flush", True)
    print(f"[{_hms()}] {msg}", **kw)


def _diversity_gate(diversity: dict) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Check the corpus diversity report on BOTH the overall `shared_fraction` AND (when present)
    the `on_policy` sub-arm's -- an overall-only check hides on-policy degeneracy that off-policy
    items dilute out of the average (observed: a historical corpus with overall 0.205 -- passes --
    whose on_policy sub-arm was 0.535 boilerplate). Returns `(hard, warn)`: lists of
    `(metric_name, shared_fraction)` exceeding 0.50 / in (0.35, 0.50] respectively; a metric lands
    in at most one of the two lists."""
    metrics = [("overall", float(diversity.get("shared_fraction", 0.0) or 0.0))]
    op = diversity.get("on_policy") or {}
    if op.get("n_items", 0) > 0:
        metrics.append(("on_policy", float(op.get("shared_fraction", 0.0) or 0.0)))
    hard = [(n, sf) for n, sf in metrics if sf > 0.5]
    warn = [(n, sf) for n, sf in metrics if 0.35 < sf <= 0.5]
    return hard, warn


def main() -> int:
    # Belt+braces vs the PYTHONFAULTHANDLER env var: explicit so the hang watchdog below (armed
    # further down, once JLENS_WATCHDOG_S is read) is guaranteed to have a handler to extend.
    faulthandler.enable()

    model_path = os.environ.get("JLENS_MODEL")
    if not model_path:
        _p("set JLENS_MODEL to the served model dir")
        return 2
    if os.path.exists(model_path):
        # mlx_lm.utils.load treats a non-existent-as-given relative path as an HF repo id
        # (HFValidationError) -- resolve a real local dir to absolute before it's passed in.
        model_path = os.path.abspath(model_path)

    t0 = time.perf_counter()
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    ad = ModelAdapter(model)
    n = ad.n_layers
    D = ad.layers[0].input_layernorm.weight.shape[0]
    mt = _model_type(ad)
    b_lo, b_hi = _band(n)
    _p(f"load {time.perf_counter()-t0:.1f}s  n_layers={n} d_model={D} band=[{b_lo},{b_hi}) "
       f"arch={mt} peak={mx.get_peak_memory()/2**30:.1f}GB")

    # `if x.strip()` tolerates a trailing/empty field -- macOS BSD `seq -s, A B`
    # emits a TRAILING separator ("16,...,47,"), which would otherwise int('')-crash.
    layers = ([int(x) for x in os.environ["JLENS_LAYERS"].split(",") if x.strip()]
              if os.environ.get("JLENS_LAYERS") else [40, 44, 47])  # shallow-band sample
    out_of_band = [l for l in layers if not (b_lo <= l < b_hi)]
    if out_of_band:
        _p(f"  WARNING: {out_of_band} outside band [{b_lo},{b_hi}) -- the product won't read them")
    chunk = int(os.environ.get("JLENS_CHUNK", 128))
    target = n - 1

    recipe = BAND_BOOTSTRAP
    if os.environ.get("JLENS_N"):
        import dataclasses
        # dataclasses.replace copies ALL fields (incl. enable_thinking) -- a hand-rolled
        # field list here silently dropped new Recipe fields once already.
        recipe = dataclasses.replace(recipe, n_prompts=int(os.environ["JLENS_N"]))

    # Checkpoint dir: the serialized corpus + running J_sum live here so a killed fit RESUMES
    # (skips on-policy generation + completed items) instead of losing everything.
    out = os.environ.get("JLENS_OUT") or str(ROOT / "out" / f"band-{len(layers)}L")
    ckpt_dir = os.path.join(out, "ckpt")
    corpus_json = os.path.join(ckpt_dir, "corpus.json")

    decoded_path = os.path.join(ckpt_dir, "corpus_decoded.md")
    if os.path.exists(corpus_json):
        corpus = C.Corpus.from_json(corpus_json)
        _p(f"RESUME: loaded {len(corpus.items)} corpus items from {corpus_json} "
           f"(skipping build + on-policy generation)")
        if not os.path.exists(decoded_path):
            # Older checkpoint predating the default decode-at-build -- regenerate it.
            try:
                Path(decoded_path).write_text(C.decode_corpus(corpus, tokenizer))
                _p(f"  decoded corpus -> {decoded_path} (local-only inspection, regenerated "
                   f"for an older checkpoint)")
            except Exception as e:
                _p(f"  (decode_corpus skipped: {e})")
    else:
        _p(f"building corpus ({recipe.name}, n={recipe.n_prompts}, "
           f"on_policy={recipe.on_policy_fraction}) -- streaming + on-policy generation...")
        tc = time.perf_counter()
        # max_seq_len bounds each item so no single fit outlives the session/checkpoint window
        # (over-long prompts are dropped, not truncated). Conservative default; tune per model.
        max_seq_len = int(os.environ.get("JLENS_MAX_SEQ_LEN", 512))
        corpus = C.build_corpus(model, tokenizer, recipe,
                                on_policy_max_tokens=int(os.environ.get("JLENS_ONPOLICY_TOKENS", 48)),
                                max_seq_len=max_seq_len, decoded_path=decoded_path)
        corpus.to_json(corpus_json)
        _p(f"  decoded corpus -> {decoded_path} (local-only inspection)")
        _p(f"  corpus: {len(corpus.items)} items in {time.perf_counter()-tc:.1f}s  "
           f"{corpus.provenance['strata']}  on_policy={sum(it.on_policy for it in corpus.items)} "
           f"  dropped_over_len={corpus.provenance['dropped_over_len']} (max_seq_len={max_seq_len}) "
           f"(saved -> {corpus_json})")
    tok_lens = [len(it.ids) for it in corpus.items]
    pos_lens = [len(it.positions) for it in corpus.items]
    _p(f"  tok_len min/med/max={min(tok_lens)}/{sorted(tok_lens)[len(tok_lens)//2]}/{max(tok_lens)}"
       f"  pos/prompt mean={sum(pos_lens)/len(pos_lens):.1f}")

    # Corpus diversity gate: works for BOTH the build branch and the resume branch (an older
    # checkpoint may not have provenance["diversity"] -- compute it live in that case). Checks
    # BOTH the overall fraction AND the on_policy sub-arm (see _diversity_gate) -- an overall-only
    # check can hide on-policy degeneracy that off-policy items dilute out of the average.
    diversity = corpus.provenance.get("diversity") or C.diversity_report(corpus.items)
    hard, warn = _diversity_gate(diversity)
    if hard and os.environ.get("JLENS_ALLOW_DEGENERATE") != "1":
        names = ", ".join(f"{name}_shared_fraction={sf:.2f}" for name, sf in hard)
        _p(f"corpus diversity gate FAILED: {names} > 0.50 (fitted positions are mostly "
           f"shared/boilerplate tokens across items) -- set JLENS_ALLOW_DEGENERATE=1 to override "
           f"if you've inspected corpus_decoded.md and are sure this is expected")
        return 3
    for name, sf in hard + warn:  # `hard` entries only reach here once JLENS_ALLOW_DEGENERATE overrode the exit
        _p(f"  ⚠ WARNING: corpus {name} shared_fraction={sf:.2f} > 0.35 -- corpus may be "
           f"degenerate/boilerplate-heavy; inspect corpus_decoded.md before a long fit")

    # GDN kernel-cliff guard: the fused Metal backward is only eligible for T <= MAX_T (=128); longer
    # items SILENTLY fall to the differentiable ops fallback -- much slower + per-step-state memory
    # blowup (OOM risk). Warn loudly and record kernel-eligibility so a corpus straddling the cliff
    # doesn't quietly balloon the fit. (Recorded in the sidecar as on_kernel_frac / max_tok_len.)
    max_t_val, n_over_maxt = None, 0
    try:
        from jlens_mlx.providers.qwen3_5_gdn import MAX_T
        from jlens_mlx.fit import _GDN_TAIL_ARCHS
        if mt in _GDN_TAIL_ARCHS:
            max_t_val = MAX_T
            n_over_maxt = sum(1 for t in tok_lens if t > MAX_T)
            if n_over_maxt:
                _p(f"  ⚠ WARNING: {n_over_maxt}/{len(tok_lens)} items exceed MAX_T={MAX_T} -> they "
                   f"fall to the SLOW differentiable ops fallback (much slower + memory blowup, OOM "
                   f"risk). Set JLENS_MAX_SEQ_LEN<={MAX_T} to keep every item on the fast GDN kernel.")
            else:
                _p(f"  all {len(tok_lens)} items <= MAX_T={MAX_T}: fully on the fast GDN kernel path.")
    except Exception:
        pass

    _p(f"fitting band layers {layers} (target={target}, chunk={chunk}) over the corpus...")

    def _progress(p):
        if p.get("resumed") is True:
            _p(f"  RESUMED from checkpoint: {p['done']} item(s) done; continuing at "
               f"item {p['next_idx']+1}/{p['n_total']}")
            return
        if p.get("resumed") is False:
            _p(f"  checkpoint not resumed: {p.get('reason')}")
            return
        if p["skipped"]:
            _p(f"  item {p['i']+1}/{p['n_total']}: SKIPPED (no usable positions)")
            return
        eta = p.get("eta_secs")
        eta_str = "eta pending (first item)" if eta is None else f"eta {eta/60:.0f}m (rate-based)"
        isp = p.get("item_sec_per_pos")
        isp_str = f"{isp:.1f}" if isp is not None else "?"
        peak = p.get("peak_gb")
        peak_str = f" | peak {peak:.1f}GB" if peak is not None else ""
        _p(f"  item {p['i']+1}/{p['n_total']} done in {p['secs']:.0f}s ({p['n_pos']} pos, "
           f"{isp_str} s/pos) | elapsed {p['elapsed']/60:.1f}m | {eta_str}{peak_str}")

    # Hang watchdog: if the process wedges (a native VJP hang, not a crash) with no chunk progress
    # for JLENS_WATCHDOG_S seconds, dump every thread's stack while the process is STILL ALIVE (vs
    # a silent multi-hour stall). Re-armed on every fit_corpus heartbeat tick below.
    watchdog_s = int(os.environ.get("JLENS_WATCHDOG_S", 1800))
    if watchdog_s > 0:
        faulthandler.dump_traceback_later(timeout=watchdog_s, repeat=True, exit=False)

    def _rearm_watchdog(_info=None) -> None:
        if watchdog_s > 0:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(timeout=watchdog_s, repeat=True, exit=False)

    tf = time.perf_counter()
    if os.environ.get("JLENS_FINALIZE"):
        # Finalize from the checkpoint WITHOUT fitting remaining items -- for when a single item
        # is longer than the session window (per-item checkpointing can't split one item). Averages
        # the banked J_sum over the items already done. The dropped items are logged for honesty.
        from jlens_mlx.fit import _ckpt_load
        jsum, meta = _ckpt_load(ckpt_dir)
        if jsum is None:
            _p("JLENS_FINALIZE set but no checkpoint found")
            return 2
        n_items = int(meta["n_done"])
        done_idx = int(meta["next_idx"])
        jacobians = {l: jsum[l] / n_items for l in layers}
        dropped = len(corpus.items) - done_idx
        _p(f"FINALIZE: averaging banked J_sum over {n_items} completed items "
           f"(dropping {dropped} un-fit item(s) {list(range(done_idx, len(corpus.items)))} -- "
           f"too long for the session window; see the length-cap note)")
    else:
        _max_fit_seq = int(os.environ["JLENS_MAX_FIT_SEQ"]) if os.environ.get("JLENS_MAX_FIT_SEQ") else None
        jacobians, n_items = fit_corpus(model, corpus, source_layers=layers, adapter=ad,
                                        target_layer=target, chunk_size=chunk, progress=_progress,
                                        checkpoint_dir=ckpt_dir, heartbeat=_rearm_watchdog,
                                        max_fit_seq=_max_fit_seq)
    mx.eval(list(jacobians.values()))
    _p(f"fit {time.perf_counter()-tf:.1f}s over {n_items} items  "
       f"peak={mx.get_peak_memory()/2**30:.1f}GB")
    for l in sorted(jacobians):
        J = jacobians[l]
        fro = float(mx.linalg.norm(J).item())
        _p(f"  J_{l}: ||J||/sqrt(D)={fro/(D**0.5):.4f} finite={bool(mx.all(mx.isfinite(J)).item())}")

    # Held-out fidelity gate (+ identity tripwire).
    grade = dict(jacobians)
    grade[target] = mx.eye(D, dtype=mx.float32)
    gl = sorted(set(layers) | {target})
    glens = lenslib.JSpaceLens(grade, gl, D, softcap=ad.softcap, meta={"target_layer": target})

    def tok(m):
        return list(tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=True,
                                                   enable_thinking=False))

    _p("fidelity gate (held-out)...")
    rep = verify.fidelity_gate(model, glens, HELD_OUT, tokenize=tok, adapter=ad,
                               skip_first=4, top_k=10, min_topk_agreement=0.0)
    for l in sorted(rep["per_layer"]):
        m = rep["per_layer"][l]
        tag = " (identity)" if l == target else (" [in-band]" if b_lo <= l < b_hi else "")
        _p(f"  J_{l}{tag}: top1={m['top1']:.3f} top10={m['topk']:.3f} kl={m['kl']:.3f}")
    _p(f"  identity_ok={rep['identity_ok']}")

    # Disposition-aware ranking (verify.legibility_report): rank band layers by whether their readout
    # is real CONTENT (' Paris') vs degenerate junk (' __') -- the RIGHT signal for band layers, which
    # are meant to diverge from the final logits (fidelity_gate's agreement metric misleads here, e.g.
    # band-5L). Recorded so the lens carries which band layers are legible.
    leg = verify.legibility_report(model, glens, HELD_OUT, tokenize=tok, tokenizer=tokenizer,
                                   adapter=ad, top_k=10)
    in_band_ranked = [l for l in leg["ranked"] if b_lo <= l < b_hi]
    _p("  legibility (band layers, most-meaningful first):")
    for l in in_band_ranked:
        lm = leg["per_layer"][l]
        _p(f"    J_{l}: legibility={lm['legibility']:.2f} entropy={lm['entropy']:.2f}")

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
        # Kernel-eligibility provenance: whether the whole corpus stayed on the fast GDN kernel
        # (T <= MAX_T) or some items hit the slow ops fallback -- so a lens records the conditions
        # it was fit under (max_tok_len, MAX_T, how many items straddled the cliff).
        "kernel": {"max_t": max_t_val, "max_tok_len": max(tok_lens), "n_over_max_t": n_over_maxt,
                   "all_on_kernel": (n_over_maxt == 0) if max_t_val is not None else None},
        "fidelity": {str(l): rep["per_layer"][l] for l in layers if l in rep["per_layer"]},
        "fidelity_identity_ok": rep["identity_ok"],
        "legibility": {str(l): leg["per_layer"][l] for l in layers if l in leg["per_layer"]},
        "legibility_ranked_band": in_band_ranked,   # band layers, most-meaningful readout first
        "note": "first band-targeted corpus fit; scoped proof (small corpus) -- dense band fit is the overnight run",
    }
    lenslib.save(jacobians, sidecar, out)
    _p(f"saved lens -> {out}")
    _p(json.dumps({"layers": sorted(layers), "n_items": n_items,
                   "identity_ok": rep["identity_ok"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
