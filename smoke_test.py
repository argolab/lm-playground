"""Check every online model through a running proxy.

Streams a short request to each model and checks that text arrives, then scores one sentence with
the first model that supports it. Exits with the number of failures.
Usage: python smoke_test.py [http://127.0.0.1:8000]
"""

import json
import math
import sys

import httpx

QUESTION = "Reply with one word: what is the capital of France?"


def request_body(model):
    """A short streamed request that only uses fields the model's capabilities list."""
    caps = model["caps"]
    body = {"model": model["id"], "stream": True, "max_tokens": 24}
    if model["kind"] == "base":
        body["prompt"] = "The capital of France is"
    else:
        body["messages"] = [{"role": "user", "content": QUESTION}]
    if "seed" in caps:
        body["seed"] = 1
    if "n" in caps:
        body["n"] = 2
    if "logprobs" in caps and "messages" in body:
        body.update(logprobs=True, top_logprobs=5)
    elif "logprobs" in caps:
        body["logprobs"] = 5
    return body


def check_generate(base, model):
    """Stream one request and return whether every completion produced text or reasoning."""
    body = request_body(model)
    texts = [""] * body.get("n", 1)
    reasoning, error, saw_logprobs = "", None, False
    with httpx.stream("POST", f"{base}/api/generate", json=body, timeout=180) as r:
        for line in r.read().decode().split("\n"):  # iter_lines also splits inside some events
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            if "error" in event:
                error = event["error"]
                break
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                texts[choice["index"]] += choice.get("text") or delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
                saw_logprobs = saw_logprobs or bool(choice.get("logprobs"))
    ok = error is None and all(text or reasoning for text in texts)
    print(
        f"{'ok  ' if ok else 'FAIL'} {model['id']:24s} logprobs={saw_logprobs} "
        f"reasoning={bool(reasoning)} | " + " | ".join(repr(text[:50]) for text in texts),
        f"| {str(error)[:200]}" if error else "",
    )
    return ok


def check_score(base, model):
    """Score one sentence and return whether the perplexity is a sensible number."""
    body = {"model": model["id"], "text": "The cat sat on the mat."}
    reply = httpx.post(f"{base}/api/score", json=body, timeout=60).json()
    scored = [lp for lp in reply["logprobs"] if lp is not None]
    perplexity = math.exp(-sum(scored) / len(scored))
    ok = len(reply["tokens"]) == len(reply["logprobs"]) and math.isfinite(perplexity)
    print(
        f"{'ok  ' if ok else 'FAIL'} score {model['id']}: {len(reply['tokens'])} tokens, "
        f"perplexity {perplexity:.1f}"
    )
    return ok


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    models = httpx.get(f"{base}/api/models", timeout=30).json()
    online = [m for m in models if m["online"]]
    for model in models:
        if not model["online"]:
            print(f"skip {model['id']} (offline)")
    results = [check_generate(base, model) for model in online]
    can_score = [m for m in online if "echo" in m["caps"]]
    if can_score:
        results.append(check_score(base, can_score[0]))
    sys.exit(results.count(False))


if __name__ == "__main__":
    main()
