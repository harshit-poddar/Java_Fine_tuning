"""Tests for data/augment_dataset.py. CPU-only; uses the mock generator."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from finetune.data.augment_dataset import (
    MockGenClient,
    augment,
    extract_json_pair,
    load_seeds,
    verify_pair,
)
from finetune.eval.harness import HarnessConfig

HAVE_JAVAC = shutil.which("javac") is not None

GOOD = "public class F { int y; }"
VULN = "public class V { int x; }"


# ------------------------------------------------------------- json extraction

def test_extract_plain_json() -> None:
    pair = extract_json_pair(json.dumps({"vulnerable_code": VULN, "fixed_code": GOOD}))
    assert pair == {"vulnerable_code": VULN, "fixed_code": GOOD}


def test_extract_json_wrapped_in_prose_and_fences() -> None:
    payload = json.dumps({"vulnerable_code": VULN, "fixed_code": GOOD})
    pair = extract_json_pair(f"Sure! Here you go:\n```json\n{payload}\n```\nDone.")
    assert pair is not None
    assert pair["fixed_code"] == GOOD


@pytest.mark.parametrize("text", [
    "no json here",
    "{\"vulnerable_code\": \"only one side\"}",
    "{\"vulnerable_code\": 1, \"fixed_code\": 2}",
    "{broken json",
])
def test_extract_rejects_bad_payloads(text: str) -> None:
    assert extract_json_pair(text) is None


# ----------------------------------------------------------------- verification

def test_verify_rejects_identical_pair() -> None:
    ok, reason = verify_pair(GOOD, GOOD, "CWE-89", HarnessConfig())
    assert not ok
    assert "identical" in reason


@pytest.mark.skipif(not HAVE_JAVAC, reason="javac not installed")
def test_verify_rejects_non_compiling_fix() -> None:
    ok, reason = verify_pair(VULN, "public class F { broken", "CWE-89", HarnessConfig())
    assert not ok
    assert "compile" in reason


def test_verify_admits_with_warning_when_tools_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    ok, reason = verify_pair(VULN, GOOD, "CWE-89", HarnessConfig())
    assert ok
    assert "semgrep unavailable" in reason
    assert "WARNING" in capsys.readouterr().err


# ----------------------------------------------------------------- end to end

def _seed_file(tmp_path: Path) -> Path:
    path = tmp_path / "train.jsonl"
    records = [
        {"system": "s", "input": "i", "output": "o", "cwe": cwe,
         "vulnerable_code": f"class Seed_{cwe.replace('-', '_')} {{}}",
         "source_id": f"{cwe}:0", "kind": "fix"}
        for cwe in ("CWE-89", "CWE-22", "CWE-78")
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_load_seeds_groups_by_cwe_and_skips_negatives(tmp_path: Path) -> None:
    path = _seed_file(tmp_path)
    extra = {"system": "s", "input": "i", "output": "o", "cwe": "CWE-89",
             "vulnerable_code": "neg", "source_id": "x#neg", "kind": "no_change"}
    path.write_text(path.read_text(encoding="utf-8") + json.dumps(extra) + "\n", encoding="utf-8")
    seeds = load_seeds(path)
    assert len(seeds["CWE-89"]) == 1  # the no_change record was skipped
    assert len(seeds["CWE-22"]) == 1


def test_augment_mock_end_to_end(tmp_path: Path) -> None:
    seeds = load_seeds(_seed_file(tmp_path))
    config = HarnessConfig() if HAVE_JAVAC else HarnessConfig(javac_bin="definitely-missing")
    rows, stats = augment(MockGenClient(), seeds, per_cwe=2, seed=42, harness_config=config)
    assert stats["admitted"] == len(rows) == 6  # 2 per CWE x 3 CWEs
    assert {r["cwe"] for r in rows} == {"CWE-89", "CWE-22", "CWE-78"}
    for row in rows:
        assert row["vulnerable_code"].strip() != row["fixed_code"].strip()

    out = tmp_path / "augmented.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["vulnerable_code", "fixed_code", "cwe"])
        writer.writeheader()
        writer.writerows(rows)
    with out.open(newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 6


def test_augment_gives_up_after_max_attempts(tmp_path: Path) -> None:
    class AlwaysGarbage:
        name = "garbage"

        def generate(self, system: str, user: str, record: dict) -> str:
            return "not json at all"

    seeds = load_seeds(_seed_file(tmp_path))
    rows, stats = augment(AlwaysGarbage(), seeds, per_cwe=2, seed=42)
    assert rows == []
    assert stats["unparseable"] == stats["generated"] == 18  # 3 CWEs x 2x3 attempts
