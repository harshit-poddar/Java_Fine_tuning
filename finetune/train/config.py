"""Training hyperparameters for FINETUNING_001 (pydantic, env-overridable).

Every field can be overridden with an `FT_`-prefixed environment variable,
e.g. `FT_BASE_MODEL`, `FT_LORA_R=16`, `FT_EPOCHS=2`,
`FT_TARGET_MODULES=q_proj,v_proj`. Plain bf16 LoRA only - QLoRA/bitsandbytes
are deliberately unsupported (unreliable on ROCm).

    python config.py   # print the resolved config as JSON
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

ENV_PREFIX = "FT_"

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "lora_adapter"


class TrainConfig(BaseModel):
    """All knobs for train/sft_lora.py. Defaults follow FINETUNE_PLAN.md."""

    # model
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_seq_len: int = 2048

    # LoRA (adapter-only; full checkpoints are never saved)
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # optimization
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    epochs: float = 3.0
    per_device_batch_size: int = 2
    grad_accum: int = 8
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    logging_steps: int = 5

    # io
    train_file: Path = _DEFAULT_DATA_DIR / "train.jsonl"
    val_file: Path = _DEFAULT_DATA_DIR / "val.jsonl"
    output_dir: Path = _DEFAULT_OUTPUT_DIR

    # smoke mode (validates the pod environment cheaply)
    smoke_examples: int = 50
    smoke_max_steps: int = 100

    @field_validator("target_modules", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @classmethod
    def from_env(cls) -> "TrainConfig":
        """Build the config, overlaying any FT_<FIELD> environment variables."""
        overrides = {}
        for name in cls.model_fields:
            raw = os.getenv(ENV_PREFIX + name.upper())
            if raw is not None:
                overrides[name] = raw
        return cls(**overrides)

    def summary(self) -> dict:
        """JSON-safe dump (also stored in run_metadata.json by the trainer)."""
        return json.loads(self.model_dump_json())


if __name__ == "__main__":
    print(json.dumps(TrainConfig.from_env().summary(), indent=2))
