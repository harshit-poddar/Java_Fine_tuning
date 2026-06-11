"""Tests for train/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from finetune.train.config import TrainConfig


def test_defaults_match_plan() -> None:
    config = TrainConfig()
    assert config.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config.lora_r == 32
    assert config.lora_alpha == 64
    assert config.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert config.learning_rate == pytest.approx(2e-4)
    assert config.lr_scheduler_type == "cosine"
    assert config.epochs == 3.0
    assert config.bf16 is True
    assert config.gradient_checkpointing is True
    assert config.grad_accum == 8


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FT_BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    monkeypatch.setenv("FT_LORA_R", "16")
    monkeypatch.setenv("FT_LEARNING_RATE", "1e-4")
    monkeypatch.setenv("FT_BF16", "false")
    monkeypatch.setenv("FT_TARGET_MODULES", "q_proj, v_proj")
    monkeypatch.setenv("FT_OUTPUT_DIR", "/tmp/adapter")
    config = TrainConfig.from_env()
    assert config.base_model == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert config.lora_r == 16
    assert config.learning_rate == pytest.approx(1e-4)
    assert config.bf16 is False
    assert config.target_modules == ["q_proj", "v_proj"]
    assert config.output_dir == Path("/tmp/adapter")


def test_summary_is_json_safe() -> None:
    summary = TrainConfig().summary()
    assert isinstance(summary["output_dir"], str)
    assert summary["lora_r"] == 32
