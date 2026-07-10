# End-to-end check of the analyze() pipeline (format_prompt chat-template +
# greedy gen + registry lens + readout) on the served 26B MoE, through the REAL
# module code the endpoint calls.
#   uv run python migrated_from_scratch/verify_endpoint.py
import os
os.environ["HEYLOOK_JSPACE_DIR"] = os.path.expanduser(
    "adapters/jspace")

from mlx_vlm.utils import load as vlm_load
from heylook_llm.jspace.registry import LensRegistry
from heylook_llm.jspace.analyze import analyze

MODEL_ID = "gemma-4-26b-a4b-it-8bit-mlx"
MODEL_PATH = os.path.expanduser("gemma-4-26b-a4b-it-8bit-mlx")


class Provider:
    def __init__(self, model, processor):
        self.model, self.processor, self.is_vlm, self.model_id = model, processor, True, MODEL_ID


reg = LensRegistry.from_env()
print("registry available:", reg.available())
lens = reg.get(MODEL_ID)
model, processor = vlm_load(MODEL_PATH)
prov = Provider(model, processor)

# Run through the FIXED endpoint path (_gated_analyze on a WORKER thread inside
# mx.stream) -- this is what run_in_threadpool crashed on (no thread-local stream).
from concurrent.futures import ThreadPoolExecutor
from heylook_llm.jspace_api import _gated_analyze
msgs = [{"role": "user", "content": "The Eiffel Tower is located in the city of"}]
with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-stream") as ex:
    res = ex.submit(_gated_analyze, prov, lens, msgs, 6, 6, True, False, None, None).result()
print("(ran _gated_analyze on a worker thread)")

print(f"\nanswer={res['answer']!r}  first_token={res['first_answer_token']!r}  risk={res['risk']}")
print("prompt tokens:", len(res["prompt_tokens"]), "| band layers:", res["band_layers"][:3], "..", res["band_layers"][-3:])
print("onset workspace (deepest 3 layers):")
for row in res["onset_strip"][-3:]:
    print(f"  L{row['layer']}: {[c['token'] for c in row['top_k']]}")
hm = res["heatmap"]
print(f"heatmap: {len(hm)} layers x {len(hm[0]['cells'])} positions" if hm else "heatmap: none")
print("feature keys:", sorted(res["features"].keys()))
print("DONE")
