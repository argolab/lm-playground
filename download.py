"""Download the models that serve.sh runs: every models.json entry with a serve block.

Files go to HF_HUB_CACHE. Safe to rerun, since files already present are skipped.
Usage: python download.py [model id ...]   (no ids means all of them)
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import snapshot_download

# Weight formats and extra folders vLLM does not need.
SKIP = [
    "original/*",
    "metal/*",
    "onnx/*",
    "*.bin",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "*.tflite",
    "*.ckpt",
]


def fetch(model):
    serve = model["serve"]
    path = snapshot_download(serve["repo"], revision=serve["revision"], ignore_patterns=SKIP)
    if not any(name.endswith(".safetensors") for name in os.listdir(path)):
        # Some repositories only have .bin weights, for example GPT-J float16.
        keep_bin = [pattern for pattern in SKIP if pattern != "*.bin"]
        path = snapshot_download(
            serve["repo"], revision=serve["revision"], ignore_patterns=keep_bin
        )
    print("done", model["id"], path, flush=True)


def main():
    models = json.loads((Path(__file__).parent / "models.json").read_text())["models"]
    models = [m for m in models if "serve" in m]
    if len(sys.argv) > 1:
        models = [m for m in models if m["id"] in sys.argv[1:]]
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(fetch, models))


if __name__ == "__main__":
    main()
