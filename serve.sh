#!/usr/bin/env bash
# Start, stop or check the local vLLM servers (models.json entries with a serve block) plus the playground proxy.
# Usage: ./serve.sh start | stop | status
# Models on one GPU start one after another so each measures free memory correctly; GPUs run in parallel.
cd "$(dirname "$0")"
if [ -f .env ]; then set -a; source .env; set +a; fi
PY=.venv/bin/python
mkdir -p logs

rows() { # id repo revision port gpu util max_len extra-flags, for models with a serve block
  $PY -c '
import json
from urllib.parse import urlparse
for m in json.load(open("models.json"))["models"]:
    s = m.get("serve")
    if not s: continue
    extra = ""
    if s.get("reasoning_parser"): extra += " --reasoning-parser " + s["reasoning_parser"]
    if s.get("dtype"): extra += " --dtype " + s["dtype"]
    print(m["id"], s["repo"], s["revision"], urlparse(m["base_url"]).port, s["gpu"], s["util"], s["max_len"], extra.strip() or "-")'
}

healthy() { curl -sf "http://127.0.0.1:$1/health" >/dev/null; }

start_gpu() {
  local gpu=$1
  rows | while read -r id repo rev port g util maxlen extra; do
    [ "$g" = "$gpu" ] || continue
    if healthy "$port"; then echo "[$id] already up on :$port"; continue; fi
    [ "$extra" = "-" ] && extra=""
    echo "[$id] starting on GPU $gpu :$port"
    CUDA_VISIBLE_DEVICES=$gpu nohup .venv/bin/vllm serve "$repo" --revision "$rev" --served-model-name "$id" \
      --port "$port" --gpu-memory-utilization "$util" --max-model-len "$maxlen" --max-num-seqs 48 \
      --max-logprobs 10 $extra > "logs/$id.log" 2>&1 &
    for _ in $(seq 1 180); do healthy "$port" && break; kill -0 $! 2>/dev/null || break; sleep 5; done
    healthy "$port" && echo "[$id] ready" || echo "[$id] FAILED, see logs/$id.log"
  done
}

case "${1:-}" in
  start)
    for gpu in $(rows | awk '{print $5}' | sort -u); do start_gpu "$gpu" & done
    wait
    # The proxy reads models.json only when it starts, so a running one is replaced.
    pkill -9 -u "$USER" -f 'uvicorn server:app' && sleep 1
    nohup .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > logs/proxy.log 2>&1 &
    echo "proxy on :8000. Tunnel: ssh -L 8000:$(hostname):8000 <login-node>  then open http://localhost:8000";;
  stop)
    rows | while read -r id rest; do pkill -u "$USER" -f -- "--served-model-name $id( |$)"; done
    pkill -u "$USER" -f 'uvicorn server:app'; echo stopped;;
  status)
    rows | while read -r id repo rev port rest; do healthy "$port" && echo "up   $id :$port" || echo "down $id :$port"; done
    curl -sf http://127.0.0.1:8000/api/models >/dev/null && echo "up   proxy :8000" || echo "down proxy :8000";;
  *) echo "usage: $0 start|stop|status"; exit 1;;
esac
