# LM Playground

Sample from language models in the style of the GPT-3 playground.

<!-- screen recording of the page goes here -->

**Try it:** Bring your own [OpenRouter](https://openrouter.ai) key to <https://argolab.github.io/lm-playground/>, or [run it on your own GPU](#serve-your-own-models).

## How to use

- **Complete** sends your text as a raw prompt and appends the model's continuation. 
- **Chat** sends a conversation through a model's chat template. 
- **Score text** under a model to get cross-entropy and perplexity.
- **Completions (n)** streams up to 16 samples side by side. Sort by total log-probs and tally their final answers into a majority vote.
- **Compare two models** side by side, like Base vs. Instruction-tuned checkpoints.
- **Show probabilities** colors generated tokens by probability, with the top alternatives on hover.
- **Show rendered prompt** and **Show tokens** to see the exact text and tokens the model receives.
- **Presets** loads demo prompts, and **Copy link to this state** puts the whole page state in a link you can point to.

Most of these features are only available on locally-hosted models.

## Local setup:

Install with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/argolab/lm-playground && cd lm-playground
uv run uvicorn server:app --port 8000
```

Open <http://localhost:8000> and paste an OpenRouter key into the page.

## Serve your own models:

Locally hosted models are served with [vLLM](https://docs.vllm.ai) servers.

```bash
uv sync --extra local
cp .env.example .env                # set HF_TOKEN and HF_HUB_CACHE
set -a; source .env; set +a
.venv/bin/python download.py        # every model with a serve block in models.json
./serve.sh start                    # one vLLM server per model
./serve.sh status
./serve.sh stop
```

`download.py <id> ...` downloads only the named models. Logs are written to `logs/`. On a cluster, run this on a compute node and start a tunnel to your laptop with
`ssh -L 8000:<compute-node>:8000 <login-node>`, then open <http://localhost:8000>.


## Configure models:

`models.json` lists every model that the page offers. 
By default, the page sets up config for the Olmo-3 family of models, GPT-2 XL, GPT-J 6B, Mistral 7B, and five models through OpenRouter.

```json
{
  "id": "olmo3-7b-base",
  "name": "Olmo 3 7B base",
  "kind": "base",
  "base_url": "http://127.0.0.1:8003/v1",
  "caps": [
    "completions",
    "logprobs",
    "echo",
    "tokenize",
    "n",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "stop",
    "penalties"
  ],
  "stats": { "Developer": "Ai2", "Parameters": "7.3B" },
  "serve": {
    "repo": "allenai/Olmo-3-1025-7B",
    "revision": "main",
    "gpu": 1,
    "util": 0.245,
    "max_len": 8192
  }
}
```

To add a model you serve: add an entry with a `serve` block on a free port and GPU, run
`download.py <id>`, then `./serve.sh start`, which leaves running model servers alone and
restarts the proxy. 

## Example: one model family across its training stages

The `models.json` in this repository is the set we used for our own language model behavior demo. [Olmo 3](https://huggingface.co/allenai/Olmo-3-1025-7B) from Ai2 publishes checkpoints
from each stage of training, so you can put the same prompt to each stage and see what that stage
added.

| Stage | Model id | What changed |
| --- | --- | --- |
| Pretraining, stage 1 | `olmo3-7b-stage1` | Next-token prediction on 5.9T tokens of web, code, papers and books. |
| Pretraining, stage 2 | `olmo3-7b-stage2` | 100B more tokens, heavy in math, code, question answering and instructions. |
| Pretraining, stage 3 (the base model) | `olmo3-7b-base` | 50B tokens of long documents, extending the context to 65K. |
| Supervised fine-tuning | `olmo3-7b-think-sft` | Trained to imitate chat responses with long reasoning traces. |
| Preference tuning (DPO) | `olmo3-7b-think-dpo` | Trained to prefer the chosen response of a chosen and rejected pair. |
| Reinforcement learning (final Think model) | `olmo3-7b-think` | Rewarded for final answers that check out as correct. |
| Instruct | `olmo3-7b-instruct` | The same three steps, tuned for short direct answers. |
| Reinforcement learning from the base model | `olmo3-7b-rlzero-math` | Reward alone on math, with no supervised fine-tuning first. |

Around these sit GPT-2 XL (2019), GPT-J 6B (2021), and Mistral 7B base and Instruct (2023) for a
view across years, gpt-oss-20b as an open reasoning model, and five hosted models through
OpenRouter. The 13 served models share four GPUs and take about 190 GB of disk.

Things to try:

- The preset "Calculus one-liner" on stage 1 and stage 2, to see what the math-heavy data added.
- The preset "Chat on a base model", which makes a base model hold a conversation with nothing but
  a prompt, as people did in 2020.
- "Grade-school math" with 16 completions and "Tally final answers", for a majority vote.
- The same question to the SFT, DPO and RL checkpoints with Compare two models.
- Score text on a passage across GPT-2 XL and a current model, to compare perplexity.

## License

MIT. See `LICENSE`.

## Development

```bash
# runs ruff check & format before each commit
uv run pre-commit install        
```
