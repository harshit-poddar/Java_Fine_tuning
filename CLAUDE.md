# CLAUDE.md — project context for Claude Code

This file is the shared brain for this repo. Read it before every task. The
detailed task spec with acceptance criteria lives in `FINETUNE_PLAN.md` —
follow its build order strictly.

## What we are building

FINETUNING_001 (TCS–AMD hackathon, Track 3): fine-tune a code LLM to fix
vulnerabilities in Java, and PROVE the gain with an objective before/after
metric against the base model. Deadline: **June 12**.

Pipeline: dataset prep → SFT-LoRA train → serve base+adapter → eval →
before/after metrics table. The metrics table IS the deliverable.

## Environment split (never confuse these)

- **This machine (local):** CPU-only. All code is written and dry-run here.
  There is NO local GPU. Never attempt to run real training or load large
  models here.
- **Remote pod (I run it manually, not you):** 1× AMD MI300X (192 GB VRAM),
  ROCm stack, ~4 GPU-hours/day, 25 GB persistent disk. I `git pull` and run
  scripts there in tmux. You never execute anything against the pod.

Every script must therefore have a CPU-safe mode (`--dry-run`, `--mock`, or
tiny-sample) so you can verify logic locally and prove it with tests.

## Hard constraints (do not violate, do not "improve")

1. **Build order:** eval harness FIRST, then dataset prep, then training,
   then eval runner. Never write training code before the harness is green.
2. **LoRA adapters only.** Never save full model checkpoints — the pod disk
   is 25 GB. Base model loads from the HuggingFace shared cache (`HF_HOME`).
3. **Plain bf16 LoRA. No QLoRA, no bitsandbytes** — unreliable on ROCm.
4. **No GRPO / RL code.** Descoped. Mention as future work in docs only.
5. **No frontend.** The deliverable is the model + metrics, not an app.
6. Base model: `Qwen/Qwen2.5-Coder-7B-Instruct`, configurable via env.
7. Target CWEs only: CWE-89 (SQLi), CWE-22 (path traversal), CWE-78
   (command injection). Do not widen scope.

## Engineering conventions

- Python 3.10+, type hints on public functions, pydantic for all config
  (env-overridable), pytest for tests.
- Determinism where possible: fixed seeds, pinned versions in
  `requirements.txt`.
- Each script is runnable standalone with `--help` and sane defaults.
- Fail loudly with actionable messages (e.g. "semgrep not found — install
  with: pip install semgrep") and degrade gracefully where the plan says so.
- Keep diffs small and focused; one task from FINETUNE_PLAN.md at a time.
- After finishing a task, run its acceptance check and the test suite, and
  show me the output before moving to the next task.

## Repo layout (create exactly this)

```
finetune/
  data/prepare_dataset.py    # raw sources -> instruction JSONL -> splits
  data/raw/                  # gitignored; human drops source data here
  data/processed/            # gitignored; script output
  eval/harness.py            # score_patch(): javac + semgrep + format
  eval/run_eval.py           # before/after metrics table (markdown)
  train/config.py            # pydantic hyperparams (lora_r=32, alpha=64, ...)
  train/sft_lora.py          # TRL SFTTrainer + LoraConfig; --smoke; adapter-only
  serve/serve_notes.md       # vLLM --enable-lora command + curl smoke test
  requirements.txt
  README.md                  # run order, local vs pod, exactly as executed
```

## GPU-budget discipline (why the rules above exist)

The pod window is for EXECUTION only: smoke train (`--smoke`: 50 examples,
~100 steps) → full train (~300–800 pairs, 2–3 epochs) → eval runs. All
debugging of logic happens locally on CPU first. A script that crashes on
the pod from a preventable bug wastes the scarcest resource we have.

## Definition of done

Following README.md in order, a human can: prep data locally → push → smoke
train on the pod → full train (adapter saved, megabytes) → serve
base+adapter via vLLM → run eval → read a markdown table showing the tuned
model beating the base model on compile rate and vuln-fixed rate.