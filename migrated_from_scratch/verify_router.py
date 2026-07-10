# V4: reproduce the hallucination-router AUC on solarkyle's e4b TriviaQA trace
# using the REAL heylook_llm.jspace.features module + the shipped e4b weights.
# Validates router_feature_vector + FeatureNormalizer + HallucinationRouter
# against the reference pipeline (analyze_router.py).
#   uv run python migrated_from_scratch/verify_router.py
import glob, json, os
import numpy as np
from heylook_llm.jspace.features import (
    HallucinationRouter, FeatureNormalizer, router_feature_vector, BASELINE_FEATURES)

# $HF_HOME = your local HF cache (contains hub/models--...); no home path baked in.
HUB = os.path.join(os.environ["HF_HOME"], "hub/models--solarkyle--jspace-lenses/snapshots")
trace = glob.glob(f"{HUB}/*/traces/uncertainty_trivia_gemma-4-e4b-it.jsonl")[0]
router_json = glob.glob(f"{HUB}/*/router/workspace_router_e4b.json")[0]

rows = [json.loads(l) for l in open(trace, encoding="utf-8")]
wrong = np.array([not r["correct"] for r in rows])
print(f"trace: {len(rows)} rows, accuracy {1 - wrong.mean():.3f}")

# Build the 14 features per row from stored trace fields (router_feature_vector
# derives the 5 entropy-trajectory features; baselines are stored directly).
feats = []
for r in rows:
    wf = router_feature_vector(r)                    # r has layer_entropies + 5 scalars
    wf.update({k: r[k] for k in BASELINE_FEATURES})
    feats.append(wf)

def auc(score, label):
    o = np.argsort(score); rank = np.empty(len(score)); rank[o] = np.arange(1, len(score) + 1)
    n1 = label.sum(); n0 = len(label) - n1
    return (rank[label].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)

aucs = {}
for variant in ("workspace_only", "combined"):
    router = HallucinationRouter.from_file(router_json, variant=variant)
    norm = FeatureNormalizer.fit(feats, router.features)     # per-model z-scoring
    risk = np.array([router.score(f, norm) for f in feats])
    aucs[variant] = auc(risk, wrong)
    print(f"{variant:14} AUC(P(wrong) vs wrong) = {aucs[variant]:.3f}  "
          f"[{len(router.features)} feats]")

# Context: output-confidence baseline alone (lower first-token logprob -> wronger).
base_auc = auc(-np.array([r["bl_first_token_logprob"] for r in rows]), wrong)
print(f"{'baseline-lp':14} AUC = {base_auc:.3f}  (first-token logprob only)")
print("V4", "PASS" if aucs["combined"] > 0.72 and aucs["workspace_only"] > 0.70 else "FAIL")
