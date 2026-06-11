"""SFT-LoRA training for FINETUNING_001 (TRL SFTTrainer + peft.LoraConfig).

Plain bf16 LoRA on `Qwen/Qwen2.5-Coder-7B-Instruct` (env-overridable, see
train/config.py). Saves the LoRA ADAPTER ONLY (megabytes) plus
run_metadata.json to the output dir - never a full checkpoint.

Modes:
  --dry-run   CPU-safe: load + validate the dataset, build LoraConfig and
              SFTConfig, print the full run report, then stop BEFORE the base
              model is touched. Run this locally before every pod session.
  --smoke     Cap the dataset at config.smoke_examples and the run at
              config.smoke_max_steps to validate the pod env cheaply.

Typical pod usage:
  python sft_lora.py --smoke          # ~10 min env validation
  python sft_lora.py                  # full train, adapter -> output_dir
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Optional

# TRL 1.x reads bundled template files without an explicit encoding; on
# Windows (local dry-runs only) that needs UTF-8 mode.
if platform.system() == "Windows":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.train.config import TrainConfig  # noqa: E402

REQUIRED_KEYS = ("system", "input", "output")


def load_records(path: Path, limit: Optional[int] = None) -> list[dict]:
    """Read instruction triplets from a prepare_dataset.py JSONL split."""
    if not path.exists():
        raise SystemExit(
            f"Dataset file not found: {path}\n"
            "Run finetune/data/prepare_dataset.py first (see README)."
        )
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [key for key in REQUIRED_KEYS if not record.get(key)]
            if missing:
                raise SystemExit(f"{path}:{line_no}: record missing keys {missing}")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise SystemExit(f"{path} is empty - re-run dataset prep.")
    return records


def to_chat_example(record: dict) -> dict:
    """Map a triplet to TRL's prompt/completion conversational format.

    TRL masks the prompt tokens automatically for this format, so loss is
    computed on the assistant completion only.
    """
    return {
        "prompt": [
            {"role": "system", "content": record["system"]},
            {"role": "user", "content": record["input"]},
        ],
        "completion": [{"role": "assistant", "content": record["output"]}],
    }


def build_lora_config(config: TrainConfig) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_sft_config(config: TrainConfig, smoke: bool, have_val: bool,
                     cpu_fallback: bool = False) -> Any:
    """Build the TRL SFTConfig.

    cpu_fallback=True is set automatically for --dry-run on machines without
    a bf16-capable GPU: transformers refuses bf16 without `use_cpu` there.
    Training itself never runs with it.
    """
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        max_steps=config.smoke_max_steps if smoke else -1,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        # float < 1 means "ratio of total steps" (warmup_ratio is deprecated)
        warmup_steps=config.warmup_ratio,
        bf16=config.bf16,
        use_cpu=cpu_fallback,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=config.max_seq_len,
        packing=False,
        eval_strategy="epoch" if have_val and not smoke else "no",
        logging_steps=config.logging_steps,
        save_strategy="no",  # adapter is saved once, explicitly, at the end
        seed=config.seed,
        report_to="none",
        model_init_kwargs={"dtype": "bfloat16" if config.bf16 else "float32"},
    )


def run(config: TrainConfig, smoke: bool, dry_run: bool) -> dict:
    """Build everything, train unless --dry-run, return the run metadata."""
    limit = config.smoke_examples if smoke else None
    train_records = load_records(config.train_file, limit)
    val_records = load_records(config.val_file) if config.val_file.exists() else []

    import torch

    cpu_fallback = dry_run and config.bf16 and not torch.cuda.is_available()
    lora_config = build_lora_config(config)
    sft_config = build_sft_config(config, smoke, bool(val_records), cpu_fallback)

    report = {
        "mode": "smoke" if smoke else "full",
        "dry_run": dry_run,
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "hyperparams": config.summary(),
        "lora": {
            "r": lora_config.r,
            "alpha": lora_config.lora_alpha,
            "dropout": lora_config.lora_dropout,
            "target_modules": sorted(lora_config.target_modules),
        },
        "max_steps": sft_config.max_steps,
    }
    print("== sft_lora run report ==")
    print(json.dumps(report, indent=2))

    if dry_run:
        print("\n--dry-run: config and dataset validated; stopping before model load.")
        return report

    import datasets
    import transformers
    import trl
    from trl import SFTTrainer

    train_ds = datasets.Dataset.from_list([to_chat_example(r) for r in train_records])
    val_ds = (
        datasets.Dataset.from_list([to_chat_example(r) for r in val_records])
        if val_records
        else None
    )

    print(f"\nloading base model {config.base_model} (from HF cache, bf16={config.bf16}) ...")
    trainer = SFTTrainer(
        model=config.base_model,  # str + model_init_kwargs -> TRL loads from cache
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
    )
    trainer.model.print_trainable_parameters()

    result = trainer.train()
    print(f"train_loss={result.training_loss:.4f}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))  # PEFT model -> adapter only
    trainer.processing_class.save_pretrained(str(output_dir))

    report["train_loss"] = result.training_loss
    report["log_history"] = trainer.state.log_history[-50:]
    report["versions"] = {
        "trl": trl.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"adapter + run_metadata.json saved to {output_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true",
                        help="50 examples / ~100 steps to validate the pod env")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate config + data on CPU; never loads the model")
    args = parser.parse_args()
    run(TrainConfig.from_env(), smoke=args.smoke, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
