# FINETUNING_001 — vulnerability-fixing Java model (SFT-LoRA)

Fine-tunes `Qwen/Qwen2.5-Coder-7B-Instruct` to fix CWE-89 (SQLi), CWE-22
(path traversal) and CWE-78 (command injection) in Java, and proves the gain
with an objective before/after metrics table (compile rate, vuln-fixed rate,
format rate) against the base model.

- **Local machine (CPU):** data prep, all dry runs, all tests.
- **Pod (1× AMD MI300X, ROCm):** training, serving, the real eval. Plain bf16
  LoRA only — no QLoRA/bitsandbytes. Adapter-only saves (megabytes).

## Setup

```bash
pip install -r finetune/requirements.txt
# local CPU torch (pod already has ROCm torch):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Also needed: a JDK (`javac`) for the compile checks. Semgrep installs via the
requirements file on Linux only; on Windows the harness marks vuln checks
`n/a` and everything else still works.

Sanity check (CPU, no model):

```bash
python -m pytest finetune/tests -q        # full suite
python finetune/eval/harness.py --selftest
```

> Windows local note: run with `PYTHONUTF8=1` set (TRL reads bundled files
> assuming UTF-8). Linux is unaffected.

## Run order

### 1. Prepare the dataset (local)

Drop source data into `finetune/data/raw/`:
- CSVs with columns `vulnerable_code,fixed_code,cwe`, and/or
- a NIST Juliet Java tree (CWE89/CWE78/CWE23/CWE36 testcases).

```bash
python finetune/data/prepare_dataset.py
# no data yet? validate the pipeline with: python finetune/data/prepare_dataset.py --make-demo-data
```

Cleans (dedup, javac filter, length cap, target CWEs only), splits 80/10/10
with no near-duplicates across splits, writes
`finetune/data/processed/{train,val,test}.jsonl` + `stats.json`, prints
per-CWE counts. Commit/push the processed splits or copy them to the pod.

### 2. Validate the training config (local)

```bash
python finetune/train/sft_lora.py --smoke --dry-run
```

Builds dataset + LoRA + trainer config and prints the run report without
touching the model. Every hyperparameter is env-overridable via `FT_*`
(see `finetune/train/config.py`, e.g. `FT_BASE_MODEL`, `FT_EPOCHS=2`).

### 3. Train on the pod

```bash
git pull && pip install -r finetune/requirements.txt
semgrep scan --validate --config finetune/eval/rules   # one-time rule check
python finetune/train/sft_lora.py --smoke               # ~10 min env validation
python finetune/train/sft_lora.py                       # full train
```

Saves ONLY the LoRA adapter + `run_metadata.json` to
`finetune/outputs/lora_adapter/`.

### 4. Serve base + adapter (pod)

See [serve/serve_notes.md](serve/serve_notes.md) — one vLLM server with
`--enable-lora --max-lora-rank 32` serves both sides of the comparison.

### 5. Run the eval (pod)

```bash
python finetune/eval/run_eval.py \
  --base-spec  "openai:http://localhost:8000/v1#Qwen/Qwen2.5-Coder-7B-Instruct" \
  --tuned-spec "openai:http://localhost:8000/v1#vuln-fixer"
```

Writes `finetune/eval/results/results.md` — the before/after table that is
the project deliverable — plus per-item JSONL dumps for debugging.

CPU rehearsal of the whole eval path (mock models, no GPU):

```bash
python finetune/eval/run_eval.py --base-spec mock:echo --tuned-spec mock:gold
```

## How scoring works

`eval/harness.py:score_patch(java_code, cwe)`:
- `format_ok` — response contains a parseable Java code block
- `compiles` — `javac` accepts the extracted code
- `vuln_fixed` — Semgrep (bundled rules in `eval/rules/`) reports no finding
  for the target CWE
- `score` — weighted combination (0.2 / 0.4 / 0.4, renormalized when a tool
  is unavailable)

## Future work (descoped)

GRPO/RL-based refinement on harness scores was considered and deliberately
descoped to fit the GPU budget; the SFT pipeline and metrics here would be
its foundation.
