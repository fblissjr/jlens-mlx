# Find the exact MLX op that hits the missing CPU stream on a worker thread.
import os
os.environ["HEYLOOK_JSPACE_DIR"] = os.path.expanduser("~/workspace/heylookitsanllm/adapters/jspace")
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mlx.core as mx
from mlx_vlm.utils import load as vlm_load
from heylook_llm.jspace.capture import ModelAdapter, capture_residuals
from heylook_llm.jspace.registry import LensRegistry
from heylook_llm.providers.common.generation_core import _get_generation_stream

MODEL_PATH = os.path.expanduser("~/Storage/llms/google/gemma-4-26b-a4b-it-8bit-mlx")
model, processor = vlm_load(MODEL_PATH)
lens = LensRegistry.from_env().get("gemma-4-26b-a4b-it-8bit-mlx")
ids = [2, 785, 30772, 8563, 374, 7407, 304, 279, 3283, 315]  # arbitrary

def step(msg):
    print(f"  {msg}", flush=True)

def worker(use_stream):
    step(f"thread start (use_stream={use_stream})")
    s = _get_generation_stream()
    ctx = mx.stream(s) if use_stream else _nullctx()
    with ctx:
        step("in ctx")
        x = mx.array([ids]); mx.eval(x); step("mx.array + eval")
        ad = ModelAdapter(model); step("adapter")
        h = ad.inner(mx.array([ids])); mx.eval(h); step("ad.inner forward + eval")
        lg = ad.logits(mx.array([ids])); mx.eval(lg); step("ad.logits + eval")
        nxt = int(mx.argmax(lg[0, -1]).item()); step(f"argmax.item = {nxt}")
        band = [7, 8, 9]
        res = capture_residuals(model, ids, band, adapter=ad); step(f"capture {list(res.keys())}")
        ll = lens.apply(ad, res, positions=[-1], layers=band); step("lens.apply")
        v = np.asarray(ll[band[0]][0]); step(f"np.asarray {v.shape}")
    step("DONE")

class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False

print("=== WITH mx.stream ===", flush=True)
with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-stream") as ex:
    ex.submit(worker, True).result()
