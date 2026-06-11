# Serving base + LoRA adapter with vLLM (pod only)

One vLLM server serves BOTH models for the eval: the base model under its HF
name and the tuned adapter under the name `vuln-fixer`. The eval runner then
hits the same endpoint twice with different `model` values.

## 1. Start the server (in tmux on the pod)

```bash
export HF_HOME=${HF_HOME:-/path/to/shared/hf-cache}   # base loads from cache

vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --enable-lora \
  --lora-modules vuln-fixer=finetune/outputs/lora_adapter \
  --max-lora-rank 32 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --port 8000
```

Notes:
- `--max-lora-rank 32` is REQUIRED: our adapter uses `lora_r=32`, vLLM's
  default cap is 16 and the server will refuse the adapter without it.
- `--lora-modules name=path`: the path is the adapter dir written by
  `train/sft_lora.py` (contains `adapter_config.json` +
  `adapter_model.safetensors`).
- If the base model env var was overridden for training (`FT_BASE_MODEL`),
  serve that same model id.

## 2. Curl smoke test

Base model:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [{"role": "user", "content": "Say OK."}],
    "max_tokens": 10
  }'
```

Tuned adapter (only the `model` field changes):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vuln-fixer",
    "messages": [{"role": "user", "content": "Say OK."}],
    "max_tokens": 10
  }'
```

Both must return a normal chat completion before starting the eval.

## 3. Eval against the served models

```bash
python finetune/eval/run_eval.py \
  --base-spec  "openai:http://localhost:8000/v1#Qwen/Qwen2.5-Coder-7B-Instruct" \
  --tuned-spec "openai:http://localhost:8000/v1#vuln-fixer"
```

Fallback if vLLM misbehaves on the ROCm stack (slower, no server needed):

```bash
python finetune/eval/run_eval.py \
  --base-spec  "hf:Qwen/Qwen2.5-Coder-7B-Instruct" \
  --tuned-spec "hf:Qwen/Qwen2.5-Coder-7B-Instruct#adapter=finetune/outputs/lora_adapter"
```
