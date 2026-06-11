"""Tests for train/sft_lora.py. CPU-only.

The full tiny end-to-end training test (real .train() on a random 1-layer
model, adapter saved) is opt-in because it downloads a small HF model:
    FT_TINY_E2E=1 python -m pytest finetune/tests/test_sft_lora.py -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from finetune.train.config import TrainConfig
from finetune.train.sft_lora import (
    build_lora_config,
    build_sft_config,
    load_records,
    run,
    to_chat_example,
)

RECORD = {
    "system": "sys",
    "input": "fix this",
    "output": "```java\nclass A {}\n```",
    "cwe": "CWE-89",
    "vulnerable_code": "class A {}",
    "source_id": "t:0",
}


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def _records(n: int) -> list[dict]:
    return [{**RECORD, "input": f"fix this, case {i}", "source_id": f"t:{i}"} for i in range(n)]


# ----------------------------------------------------------------- load_records

def test_load_records_roundtrip(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "train.jsonl", _records(3))
    assert len(load_records(path)) == 3


def test_load_records_respects_limit(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "train.jsonl", _records(10))
    assert len(load_records(path, limit=4)) == 4


def test_load_records_missing_file_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="prepare_dataset"):
        load_records(tmp_path / "nope.jsonl")


def test_load_records_rejects_missing_keys(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "train.jsonl", [{"system": "s", "input": "i"}])
    with pytest.raises(SystemExit, match="output"):
        load_records(path)


# -------------------------------------------------------------------- mapping

def test_to_chat_example_prompt_completion_shape() -> None:
    example = to_chat_example(RECORD)
    assert [m["role"] for m in example["prompt"]] == ["system", "user"]
    assert example["completion"][0]["role"] == "assistant"
    assert example["completion"][0]["content"].startswith("```java")


# ------------------------------------------------------------- config builders

def test_build_lora_config_matches_train_config() -> None:
    lora = build_lora_config(TrainConfig())
    assert lora.r == 32
    assert lora.lora_alpha == 64
    assert set(lora.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert lora.task_type == "CAUSAL_LM"


def test_build_sft_config_full_vs_smoke() -> None:
    # cpu_fallback=True because this machine has no bf16 GPU; bf16 must
    # still survive in the config (it is what the pod will run with).
    config = TrainConfig()
    full = build_sft_config(config, smoke=False, have_val=True, cpu_fallback=True)
    assert full.max_steps == -1
    assert full.num_train_epochs == 3.0
    assert full.eval_strategy.value == "epoch"
    assert full.bf16 is True
    assert full.gradient_checkpointing is True
    assert full.save_strategy.value == "no"
    assert full.warmup_steps == pytest.approx(0.03)  # ratio-style warmup

    smoke = build_sft_config(config, smoke=True, have_val=True, cpu_fallback=True)
    assert smoke.max_steps == config.smoke_max_steps
    assert smoke.eval_strategy.value == "no"


# ------------------------------------------------------------ dry-run pipeline

def _tmp_config(tmp_path: Path, n_train: int = 8, **overrides) -> TrainConfig:
    train = _write_jsonl(tmp_path / "train.jsonl", _records(n_train))
    val = _write_jsonl(tmp_path / "val.jsonl", _records(2))
    return TrainConfig(
        train_file=train, val_file=val, output_dir=tmp_path / "adapter", **overrides
    )


def test_dry_run_reports_without_model(tmp_path: Path) -> None:
    config = _tmp_config(tmp_path)
    report = run(config, smoke=False, dry_run=True)
    assert report["dry_run"] is True
    assert report["train_examples"] == 8
    assert report["val_examples"] == 2
    assert report["lora"]["r"] == 32
    assert not (tmp_path / "adapter").exists()  # nothing written on dry-run


def test_dry_run_smoke_caps_examples(tmp_path: Path) -> None:
    config = _tmp_config(tmp_path, n_train=60)
    report = run(config, smoke=True, dry_run=True)
    assert report["train_examples"] == config.smoke_examples
    assert report["max_steps"] == config.smoke_max_steps


# ----------------------------------------------------- opt-in tiny end-to-end

@pytest.mark.skipif(os.getenv("FT_TINY_E2E") != "1", reason="set FT_TINY_E2E=1 to run")
def test_tiny_end_to_end_train_saves_adapter_only(tmp_path: Path) -> None:
    config = _tmp_config(
        tmp_path,
        n_train=4,
        base_model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        bf16=False,
        gradient_checkpointing=False,
        per_device_batch_size=1,
        grad_accum=1,
        max_seq_len=512,
        smoke_examples=4,
        smoke_max_steps=2,
        logging_steps=1,
    )
    report = run(config, smoke=True, dry_run=False)
    assert "train_loss" in report

    out = tmp_path / "adapter"
    files = {p.name for p in out.iterdir()}
    assert "adapter_config.json" in files
    assert "adapter_model.safetensors" in files
    assert "run_metadata.json" in files
    # adapter only - no full model checkpoint
    assert "model.safetensors" not in files
    assert not any(name.startswith("checkpoint-") for name in files)
    assert (out / "adapter_model.safetensors").stat().st_size < 50 * 1024 * 1024
