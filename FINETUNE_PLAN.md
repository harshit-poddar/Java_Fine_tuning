# Project: Vulnerability-Aware Java Model — SFT-LoRA fine-tune (FINETUNING_001)

## Goal
Fine-tune a code LLM to fix vulnerabilities in Java code, and PROVE it beats the
base model with an objective before/after metric. Deadline: June 12.

## Hard constraints (do not violate)
- **GPU is remote and scarce.** All code runs CPU-only on my laptop EXCEPT the
  training and eval scripts, which I run manually on a remote pod. Do NOT assume
  a local GPU. Do NOT run training yourself.
- **Save LoRA adapters only** (megabytes). Never save full model checkpoints —
  remote disk is 25 GB. Base model loads from the HuggingFace shared cache.
- **Plain bf16 LoRA. No QLoRA / bitsandbytes** (unreliable on the ROCm stack).
- **Build the eval harness BEFORE the training script.** Measurement first.
- Every script must run on CPU for a dry-run with a tiny sample and a `--mock`
  or small-sample mode, so I can validate logic without the GPU.
- Python 3.10+, type hints, pydantic for any structured config, pytest for tests.

## Base model
`Qwen/Qwen2.5-Coder-7B-Instruct` (code-specialized). Make it configurable via env.

## Target CWEs (keep narrow)
CWE-89 (SQL injection), CWE-22 (path traversal), CWE-78 (command injection).

## Repo layout to create
```
finetune/
  data/
    prepare_dataset.py      # source -> cleaned instruction triplets -> train/val/test
    raw/                    # gitignored; I drop downloaded source data here
    processed/              # gitignored; script output
  eval/
    harness.py             # score a generated patch: compile + vuln-scan + format
    run_eval.py            # run a model over test set, emit before/after metrics table
  train/
    sft_lora.py            # TRL SFTTrainer + LoraConfig; saves adapter only
    config.py              # all hyperparams, pydantic-typed, env-overridable
  serve/
    serve_notes.md         # vLLM command to serve base + LoRA adapter for the demo
  requirements.txt
  README.md                # how to run each step, in order
```

## Build order + acceptance criteria

### Task 1 — Eval harness (`eval/harness.py`) — DO THIS FIRST
A function `score_patch(java_code: str, cwe: str) -> dict` returning:
- `compiles: bool` — write to a temp file, run `javac`, capture result.
- `vuln_fixed: bool` — run Semgrep with the rule for that CWE; True if no finding.
- `format_ok: bool` — output is a parseable Java code block.
- `score: float` — weighted combination.
Must degrade gracefully if `javac`/`semgrep` aren't installed (warn, mark None).
Include a `--selftest` that runs 2 hardcoded examples (one vulnerable, one fixed)
and prints their scores. **Acceptance:** selftest runs on CPU, no model needed.

### Task 2 — Dataset prep (`data/prepare_dataset.py`)
- Read source data from `data/raw/` (I will provide; assume NIST Juliet Java
  layout and a generic CSV of {vulnerable_code, fixed_code, cwe} as two input
  adapters — make the parser pluggable).
- Produce instruction triplets: `{system, input: vuln code + CWE description,
  output: patched code}` in JSONL.
- **Clean:** dedup, drop pairs whose `fixed_code` fails `javac`, filter to target
  CWEs, cap length to the model context.
- **Split** train/val/test ~80/10/10 with no cross-split near-duplicates.
- Print dataset stats (counts per CWE per split).
**Acceptance:** runs on CPU; given a tiny sample in `data/raw/`, emits valid
JSONL splits and stats.

### Task 3 — Training config (`train/config.py`)
Pydantic model: base_model, lora_r=32, lora_alpha=64,
target_modules=[q_proj,k_proj,v_proj,o_proj], lr=2e-4, cosine schedule,
epochs=3, bf16=True, gradient_checkpointing=True, grad_accum=8,
max_seq_len, output_dir, all env-overridable.

### Task 4 — SFT training script (`train/sft_lora.py`)
- TRL `SFTTrainer` + `peft.LoraConfig` from the config.
- Load base from HF cache; **save adapter only** to `output_dir`.
- bf16, gradient checkpointing, gradient accumulation, dataset streaming.
- A `--smoke` flag: 50 examples, ~100 steps, to validate the env on the pod.
- Log loss; save adapter + a small `run_metadata.json` (hyperparams, dataset size).
**Acceptance:** with `--smoke --dry-run` it builds the trainer and reports config
WITHOUT requiring a GPU (skip the actual `.train()` under `--dry-run`).

### Task 5 — Eval runner (`eval/run_eval.py`)
- Takes a model spec (base-only, or base+adapter) via an OpenAI-compatible
  endpoint URL (the served model) OR a local transformers load — make it pluggable.
- Runs the test set, calls `harness.score_patch` on each output, aggregates.
- Emits a markdown **before/after table**: compile rate, vuln-fixed rate,
  format rate, overall — base vs tuned.
**Acceptance:** with a mock model returning canned patches, produces the table on CPU.

### Task 6 — Serve notes + README
- `serve/serve_notes.md`: the exact vLLM command to serve base + LoRA adapter
  (`--enable-lora`, adapter path) on the pod, plus the curl smoke test.
- `README.md`: the run order — prepare data → (pod) smoke train → full train →
  serve → run_eval → read the table.

## What NOT to build
- No GRPO / RL (descoped; mention as future work only).
- No QLoRA, no full-model checkpointing.
- No frontend. This track's deliverable is the model + the metrics table.

## Definition of done
`README.md` lets me, in order: prep data locally, push to the pod, run a smoke
train, run the full train (adapter saved), serve it, and produce a before/after
metrics table that shows the tuned model beating the base model.