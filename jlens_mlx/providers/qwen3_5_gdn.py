"""qwen3_5 (Gated-DeltaNet hybrid) accelerator.

Analytic branch-Jacobian assembly + a custom Metal GDN backward kernel. MLX's
fused GDN kernel has no VJP and the ops fallback is ~22x slower; this makes a
full-depth fit affordable (~30-60x faster than the exact VJP, and measured MORE
accurate). Covers Qwen3.5/3.6-27B (our served heretic: 64 layers,
full_attention_interval=4 -> 48 GDN + 16 full-attn, d_model 5120).

Vendored seed (Apache-2.0): vendor/jlens_qwen36/{analytic_layer, analytic_attn,
custom_gdn_vjp, gdn_backward, patch_gdn, custom_gdn_patch}.py. Port per
MIGRATION.md step 2 WITHOUT changing numerics -- the seed is validated (analytic
branches vs mx.vjp, kernel vs ops, golden gate). Reach the text stack of
Qwen3_5ForConditionalGeneration via .language_model.model.
"""
from __future__ import annotations


class Qwen35GdnProvider:
    model_type = "qwen3_5"

    def final_norm_jacobian(self, model):
        raise NotImplementedError("port from vendor/jlens_qwen36/analytic.py")

    def layer_jacobian(self, model, layer_idx, input_acts):
        raise NotImplementedError(
            "port analytic assembly (analytic_layer/analytic_attn) + Metal GDN backward "
            "(custom_gdn_vjp/gdn_backward/patch_gdn/custom_gdn_patch) from vendor/jlens_qwen36/"
        )
