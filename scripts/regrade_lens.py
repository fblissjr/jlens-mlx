"""Re-run the held-out fidelity gate on an already-saved lens and rewrite its sidecar
fidelity fields. Cheap (load + a couple held-out forwards); use after a gate-threshold
change so an existing artifact carries the corrected scores. JLENS_MODEL + JLENS_OUT."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jlens_mlx import lens as lenslib, verify  # noqa: E402
from jlens_mlx.capture import ModelAdapter  # noqa: E402

HELD_OUT = [
    [{"role": "user", "content": "Suggest a simple recipe for a weeknight dinner."}],
    [{"role": "user", "content": "How would you explain gravity to a ten-year-old?"}],
    [{"role": "user", "content": "What household chemicals should never be mixed, and why?"}],
]


def main() -> int:
    out = os.environ["JLENS_OUT"]
    model, tokenizer = load(os.environ["JLENS_MODEL"])
    mx.eval(model.parameters())
    ad = ModelAdapter(model)
    D = ad.layers[0].input_layernorm.weight.shape[0]

    lens = lenslib.load(out)
    target = int(lens.meta.get("target_layer", ad.n_layers - 1))
    jac = dict(lens.jacobians)
    jac[target] = mx.eye(D, dtype=mx.float32)  # identity for the tripwire
    grade_layers = sorted(set(lens.source_layers) | {target})
    graded = lenslib.JSpaceLens(jac, grade_layers, D, softcap=ad.softcap,
                                meta={"target_layer": target})

    def tok(m):
        return list(tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=True))

    rep = verify.fidelity_gate(model, graded, HELD_OUT[:2], tokenize=tok, adapter=ad,
                               skip_first=int(lens.meta.get("skip_first", 4)), top_k=10,
                               min_topk_agreement=0.0)
    for l in sorted(rep["per_layer"]):
        m = rep["per_layer"][l]
        tag = " (identity)" if l == target else ""
        print(f"  J_{l}{tag}: top1={m['top1']:.3f} top10={m['topk']:.3f} kl={m['kl']:.3f}")
    print(f"  identity_ok={rep['identity_ok']}  worst_layer={rep['worst_layer']}")

    side_path = Path(out) / "lens.sidecar.json"
    side = json.loads(side_path.read_text())
    side["fidelity"] = {str(l): rep["per_layer"][l]
                        for l in lens.source_layers if l in rep["per_layer"]}
    side["fidelity_identity_ok"] = rep["identity_ok"]
    side_path.write_text(json.dumps(side, indent=2))
    print(f"rewrote {side_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
