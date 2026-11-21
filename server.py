"""Proxy for the playground page, on one port.

Serves index.html, lists the models in models.json, and relays requests to each model's
OpenAI-compatible endpoint (a vLLM server you run, or a hosted API). The page decides what goes
into a request; the proxy adds the API key and passes the reply through.
"""

import asyncio
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")
MODELS = {m["id"]: m for m in json.loads((HERE / "models.json").read_text())["models"]}

# No connection limit: a class sampling many completions at once would otherwise queue silently.
client = httpx.AsyncClient(
    timeout=httpx.Timeout(900, connect=10), limits=httpx.Limits(max_connections=None)
)
app = FastAPI()


def find_model(model_id):
    if model_id not in MODELS:
        raise HTTPException(404, f"unknown model {model_id}")
    return MODELS[model_id]


def auth_headers(model, request):
    """Bearer header from the key the page sent, else from the env var named in models.json."""
    key = request.headers.get("x-api-key") or os.environ.get(model.get("api_key_env", ""), "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def require(model, cap, feature):
    if cap not in model["caps"]:
        raise HTTPException(
            400,
            f"{feature} needs an endpoint with the '{cap}' capability, which {model['id']} "
            "does not list. Local vLLM servers have it.",
        )


async def post_json(model, request, url, body):
    """POST to a model's endpoint and return the JSON reply. Failures become HTTP errors."""
    try:
        r = await client.post(url, json=body, headers=auth_headers(model, request))
    except httpx.HTTPError as e:
        raise HTTPException(502, f"{model['id']} is not reachable: {e!r}") from e
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text[:500])
    return r.json()


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/models")
async def list_models():
    """Every model in models.json, without its serve block, plus whether its endpoint answers."""

    async def online(model):
        if model.get("api_key_env"):
            return True  # hosted API: the key may live only in the browser, so do not probe
        try:
            r = await client.get(model["base_url"] + "/models", timeout=2)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    status = await asyncio.gather(*(online(m) for m in MODELS.values()))
    return [
        {**{k: v for k, v in m.items() if k != "serve"}, "online": ok}
        for m, ok in zip(MODELS.values(), status)
    ]


async def relay(url, headers, body):
    """Forward one streaming request and yield the reply as it arrives.

    A failure is sent as one `error` event, the same shape hosted APIs use for errors mid-stream.
    The reply is not split into lines here: httpx also splits at characters such as U+2028, which
    a model may write inside an event.
    """
    try:
        async with client.stream("POST", url, json=body, headers=headers) as r:
            if r.status_code != 200:
                text = (await r.aread()).decode(errors="replace")
                yield f"data: {json.dumps({'error': text[:2000]})}\n\n"
                return
            async for chunk in r.aiter_bytes():
                yield chunk
    except httpx.HTTPError as e:
        yield f"data: {json.dumps({'error': repr(e)})}\n\n"


@app.post("/api/generate")
async def generate(request: Request):
    """Relay a streaming request to the model's chat endpoint, or its raw completion endpoint."""
    body = await request.json()
    model = find_model(body["model"])
    path = "/chat/completions" if "messages" in body else "/completions"
    events = relay(model["base_url"] + path, auth_headers(model, request), body)
    return StreamingResponse(events, media_type="text/event-stream")


@app.post("/api/tokenize")
async def tokenize(request: Request):
    """The exact prompt the model sees, split into tokens (vLLM's /tokenize and /detokenize)."""
    body = await request.json()
    model = find_model(body.pop("model"))
    require(model, "tokenize", "Rendered prompt and token view")
    root = model["base_url"].removesuffix("/v1")
    tokenized = await post_json(
        model,
        request,
        root + "/tokenize",
        {"model": model["id"], "return_token_strs": True, "add_generation_prompt": True, **body},
    )
    detokenized = await post_json(
        model, request, root + "/detokenize", {"model": model["id"], "tokens": tokenized["tokens"]}
    )
    return {
        "count": tokenized["count"],
        "token_strs": tokenized["token_strs"],
        "text": detokenized.get("prompt", ""),
    }


@app.post("/api/score")
async def score(request: Request):
    """Log-probability of each token of a given text (a completion that echoes the prompt)."""
    body = await request.json()
    model = find_model(body["model"])
    require(model, "echo", "Score text")
    reply = await post_json(
        model,
        request,
        model["base_url"] + "/completions",
        {
            "model": model["id"],
            "prompt": body["text"],
            "max_tokens": 0,
            "echo": True,
            "logprobs": 5,
        },
    )
    logprobs = reply["choices"][0]["logprobs"]
    return {
        "tokens": logprobs["tokens"],
        "logprobs": logprobs["token_logprobs"],
        "top": logprobs["top_logprobs"],
    }
