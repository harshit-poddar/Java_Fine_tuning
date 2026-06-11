Here is the cleaned and properly formatted step-by-step guide. I have fixed the spacing and line breaks from the terminal output so you can copy the block below and save it directly as a single `.txt` or `.md` file.

```text
Serving the Model on the AMD Jupyter Pod — Step by Step

How to bring up a vLLM inference endpoint on the AMD JupyterLab pod and reach it from your laptop via localtunnel. Works for both tracks:
- Track 1 (agents): serve the agent model, capture fixtures + benchmarks.
- Track 3 (fine-tune): after training, serve base + LoRA adapter for eval.

Image to launch: vLLM 0.17.1 + ROCm 7.0 (serving). Use the Torch image only for training.

---

## 0. Conventions used below
- MODEL_API_KEY = abc-123 (change if you like; must match everywhere)
- served name = incident-llm (stable name so app config never changes)
- port = 8000 (vLLM and the tunnel MUST use the same port)

---

## 1. Open a Terminal and start tmux
In JupyterLab: File -> New -> Terminal, then:
```bash
tmux new -s vllm

```

Everything runs inside tmux so the server survives kernel restarts / browser disconnects.

* Detach: Ctrl-b then d
* Reattach: tmux attach -t vllm
* New window: Ctrl-b then c
* Switch windows: Ctrl-b then a number

---

## 2. Sanity-check the environment (1 minute, saves hours)

```bash
rocm-smi                 # GPU visible? ~192 GB VRAM free?
df -h /                  # disk headroom (pod is ~25 GB)
vllm --version           # confirm 0.17.1
echo $HF_HOME            # is a shared model cache configured?

```

If HF_HOME is empty and the platform provides a shared cache, set it so model downloads do NOT fill the 25 GB disk (ask organizers for the real path):

```bash
export HF_HOME=/path/to/shared/cache

```

Rule of thumb: a 4B model (~8 GB) fits the local disk; anything bigger needs the shared cache.

---

## 3. (Pod libraries) Install only your thin app deps — NEVER torch/vllm

The image already ships vLLM + Torch + the ROCm stack. Do not reinstall those (a stray `pip install torch/vllm` can pull a non-ROCm build and break the GPU). Only add lightweight app libraries if you'll run your code on the pod:

```bash
pip install pydantic pydantic-settings openai httpx

```

---

## 4. Start the vLLM server (Terminal window 1)

Track 1 / general serving (dev model):

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve Qwen/Qwen3-4B \
  --served-model-name incident-llm \
  --api-key abc-123 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --max_model_len 24272 \
  --gpu-memory-utilization 0.85

```

Wait until the log shows the server is running on port 8000 (first run also downloads/loads the weights).

Bigger demo model — same command, swap the repo (and keep the parser if you stay in the Qwen family):

```bash
# vllm serve Qwen/Qwen3-32B  ... (rest identical)

```

Track 3 — serve base + LoRA adapter (after training):

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --served-model-name incident-llm \
  --api-key abc-123 --port 8000 \
  --enable-lora \
  --lora-modules vuln-fix=/path/to/your/adapter \
  --trust-remote-code --max_model_len 24272 \
  --gpu-memory-utilization 0.85

# call with "model":"vuln-fix" to use the adapter, "incident-llm" for the base

```

---

## 5. Verify locally on the pod (Terminal window 2: Ctrl-b then c)

```bash
# is it alive?
curl http://localhost:8000/v1/models -H "Authorization: Bearer abc-123"

# does it generate?
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer abc-123" -H "Content-Type: application/json" \
  -d '{"model":"incident-llm","messages":[{"role":"user","content":"ping"}],"max_tokens":10}'

```

---

## 6. Tool-call check (do this BEFORE any fixture work — vLLM 0.17.1 is newer)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer abc-123" -H "Content-Type: application/json" \
  -d '{
    "model": "incident-llm",
    "messages": [{"role":"user","content":"What is the error rate of payments-api right now?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "query_metrics",
        "description": "Get current metrics for a service",
        "parameters": {"type":"object","properties":{"service":{"type":"string"}},"required":["service"]}
      }
    }],
    "max_tokens": 200
  }'

```

Success = response contains a tool_calls block calling query_metrics with {"service":"payments-api"}. If you get prose or a server-log parser error, the parser name is wrong for this version — check accepted names and restart:

```bash
vllm serve --help | grep -A3 tool-call-parser

```

---

## 7. Expose the endpoint to your laptop with localtunnel (Terminal window 3)

vLLM only listens on the pod's localhost; localtunnel makes it reachable from your laptop. SAME port as vLLM.

```bash
npx localtunnel --port 8000
# -> "your url is [https://something.loca.lt](https://something.loca.lt)"   <-- copy this URL

```

Notes:

* The URL changes every time you restart localtunnel — treat it as a per-session value.
* localtunnel shows a reminder page unless you send the header Bypass-Tunnel-Reminder: true (see below).
* Keep BOTH window 1 (vLLM) and window 3 (tunnel) alive — killing either drops the connection.

---

## 8. Smoke-test from your LAPTOP

```bash
curl -H "Content-Type: application/json" \
     -H "Bypass-Tunnel-Reminder: true" \
     -H "Authorization: Bearer abc-123" \
     -d '{"model":"incident-llm","messages":[{"role":"user","content":"ping"}],"max_tokens":10}' \
     [https://something.loca.lt/v1/chat/completions](https://something.loca.lt/v1/chat/completions)

```

A 200 OK with a completion means your laptop is now driving the pod GPU.

Windows PowerShell variant (note the --% and escaped quotes):

```powershell
curl.exe --% -H "Content-Type: application/json" -H "Bypass-Tunnel-Reminder: true" -H "Authorization: Bearer abc-123" -d "{\"model\":\"incident-llm\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":10}" [https://something.loca.lt/v1/chat/completions](https://something.loca.lt/v1/chat/completions)

```

---

## 9. Point your app at the tunnel (on your laptop)

In your repo .env:

```env
MODEL_MODE=live
MODEL_BASE_URL=[https://something.loca.lt/v1](https://something.loca.lt/v1)
MODEL_NAME=incident-llm
MODEL_API_KEY=abc-123

```

Your OpenAI-compatible client must send the bypass header:

```python
from openai import OpenAI

client = OpenAI(
    base_url=MODEL_BASE_URL,
    api_key=MODEL_API_KEY,
    default_headers={"Bypass-Tunnel-Reminder": "true"},
)

```

Now everything you built (agents / eval runner) runs locally against the real pod model — no code changes beyond the env + that header.

---

## 10. Do the session's real work

Run pre-written scripts only (no live coding in a GPU window):

* Track 1: capture golden-path fixtures, run the concurrency benchmark.
* Track 3: run run_eval.py for base, then base+adapter -> before/after table.

Watch the GPU during benchmarks (a spare window):

```bash
watch -n 2 rocm-smi

```

---

## 11. Shut down (the server burns GPU budget the whole time it is up)

```bash
# window 3: Ctrl-C to stop localtunnel
# window 1: Ctrl-C to stop vllm
tmux kill-session -t vllm

```

---

## Quick reference — the four moving parts

| Part | Where | Command / value |
| --- | --- | --- |
| Image | launch | vLLM 0.17.1 + ROCm 7.0 (Torch image for training) |
| vLLM server | pod, win 1 | vllm serve ... --port 8000 |
| Tunnel | pod, win 3 | npx localtunnel --port 8000 |
| App connection | laptop .env | MODEL_BASE_URL=/v1 + bypass header |

## Most likely snags (and the fix)

1. No shared cache / disk fills -> set HF_HOME, or use the 4B model.
2. Tool calls return prose -> wrong --tool-call-parser for 0.17.1; check vllm serve --help.
3. Tunnel reminder page instead of JSON -> missing Bypass-Tunnel-Reminder header.
4. GPU not found / weird torch errors -> something reinstalled torch/vllm; use a fresh pod from the correct image and don't touch those packages.

```

```