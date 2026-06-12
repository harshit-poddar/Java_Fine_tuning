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

It also hardens the mix automatically:

- **Negatives (~10%, `PREP_NEGATIVES_FRAC`):** already-secure examples whose
  expected output is "return it unchanged" — teaches restraint, so the model
  doesn't "fix" secure code. Sampled per split after splitting (no leakage).
- **Replay (optional, `--replay-file general.jsonl`):** mixes general
  instruction data into TRAIN only (capped by `PREP_REPLAY_FRAC`, default
  15%) so the adapter keeps its general instruction-following. val/test stay
  pure task data — the metric is never diluted.

### 2. Validate the training config (local)

```bash
python finetune/train/sft_lora.py --smoke --dry-run
```

Builds dataset + LoRA + trainer config and prints the run report without
touching the model. Every hyperparameter is env-overridable via `FT_*`
(see `finetune/train/config.py`, e.g. `FT_BASE_MODEL`, `FT_EPOCHS=2`).

### 3. Train on the pod (MI300X session)

```bash
git pull && pip install -r finetune/requirements.txt
source finetune/pod_env.sh                              # 32B / r=64 / seq-4096 profile
semgrep scan --validate --config finetune/eval/rules    # one-time rule check
python finetune/eval/harness.py --selftest              # tools sanity check
python finetune/train/sft_lora.py --smoke               # GATE: env validation
python finetune/train/sft_lora.py                       # full train
# crashed mid-train? checkpoints are on (FT_SAVE_STRATEGY=epoch):
# python finetune/train/sft_lora.py --resume
```

**Smoke gate:** if the 32B smoke run OOMs or misbehaves on ROCm, fall back via
`export FT_BASE_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct` (or the 7B) and re-smoke
— you lose minutes, not the session.

Saves ONLY the LoRA adapter + `run_metadata.json` (incl. rocm-smi VRAM/busy%
utilization evidence) to `finetune/outputs/lora_adapter/`.

#### Optional: model-in-the-loop data augmentation

With the base model served (step 4), the GPU can also *curate data*: generate
new vulnerable/fixed pairs, every one machine-verified (javac + semgrep)
before admission:

```bash
python finetune/data/augment_dataset.py \
  --endpoint "http://localhost:8000/v1#${FT_BASE_MODEL}" --per-cwe 50
python finetune/data/prepare_dataset.py    # dedups + rebuilds splits
```

CPU rehearsal: `python finetune/data/augment_dataset.py --endpoint mock --per-cwe 3`

#### Storage plan (incremental)

| Tier | Disk | Unlocks |
| --- | --- | --- |
| baseline | 25 GB | 7B weights + adapters + data |
| +1 | ~100 GB | 32B weights in `HF_HOME` + epoch checkpoints (`--resume`) |
| +2 | ~150 GB+ | both model scales cached + multiple ablation adapters + eval dumps |

Request the next tier only when the session plan needs it; the adapter-only
discipline stays regardless (full checkpoints are never written).

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

### 6. Optional: the scale ablation (time permitting)

Train a second adapter at 7B on the identical data and eval it the same way —
a 2x2 result ({7B, 32B} x {base, tuned}) showing how fine-tuning gain
interacts with model scale:

```bash
export FT_BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
export FT_OUTPUT_DIR=finetune/outputs/lora_adapter_7b
python finetune/train/sft_lora.py
# then serve + run_eval against the 7B pair; keep both results.md files
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
