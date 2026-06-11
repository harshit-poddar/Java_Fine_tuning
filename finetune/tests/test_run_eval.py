"""Tests for eval/run_eval.py. CPU-only; uses mock clients throughout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune.eval.run_eval import (
    MockClient,
    aggregate,
    evaluate_model,
    load_test_records,
    make_client,
    render_markdown,
)

RECORD = {
    "system": "sys",
    "input": "Fix it.\n\n```java\npublic class A { int x; }\n```",
    "output": "```java\npublic class A { int y; }\n```",
    "cwe": "CWE-89",
    "vulnerable_code": "public class A { int x; }",
    "source_id": "t:0",
}


# ----------------------------------------------------------------- spec parsing

def test_make_client_mock() -> None:
    assert make_client("mock:gold").mode == "gold"


def test_make_client_openai() -> None:
    client = make_client("openai:http://localhost:8000/v1#served-name")
    assert client.base_url == "http://localhost:8000/v1"
    assert client.model == "served-name"


@pytest.mark.parametrize("spec", ["mock:nope", "openai:no-hash", "wat:x", "hf:m#bad=1"])
def test_make_client_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        make_client(spec)


# ---------------------------------------------------------------- mock clients

def test_echo_mock_returns_input_code() -> None:
    response = MockClient("echo").generate(RECORD["system"], RECORD["input"], RECORD)
    assert "int x;" in response
    assert response.startswith("```java")


def test_gold_mock_returns_reference_fix() -> None:
    response = MockClient("gold").generate(RECORD["system"], RECORD["input"], RECORD)
    assert response == RECORD["output"]


def test_refuse_mock_has_no_code() -> None:
    assert "```" not in MockClient("refuse").generate("s", "u", RECORD)


# ------------------------------------------------------------------- pipeline

def test_evaluate_model_scores_every_record() -> None:
    results = evaluate_model(MockClient("gold"), [RECORD, {**RECORD, "source_id": "t:1"}])
    assert len(results) == 2
    assert all(r["format_ok"] for r in results)
    assert {r["source_id"] for r in results} == {"t:0", "t:1"}


def test_evaluate_model_survives_generation_error() -> None:
    class Boom:
        name = "boom"

        def generate(self, system: str, user: str, record: dict) -> str:
            raise RuntimeError("connection lost")

    results = evaluate_model(Boom(), [RECORD])
    assert results[0]["format_ok"] is False
    assert results[0]["score"] == 0.0


# ------------------------------------------------------------------ aggregation

def _result(format_ok=True, compiles=True, vuln_fixed=True, score=1.0, cwe="CWE-89"):
    return {"format_ok": format_ok, "compiles": compiles, "vuln_fixed": vuln_fixed,
            "score": score, "cwe": cwe}


def test_aggregate_rates() -> None:
    summary = aggregate([
        _result(),
        _result(compiles=False, vuln_fixed=False, score=0.2),
        _result(vuln_fixed=None, score=0.6, cwe="CWE-78"),
    ])
    assert summary["n"] == 3
    assert summary["format"] == (3, 3)
    assert summary["compile"] == (2, 3)
    assert summary["vuln_fixed"] == (1, 2)  # None excluded
    assert summary["mean_score"] == pytest.approx(0.6)
    assert summary["per_cwe_vuln_fixed"]["CWE-89"] == (1, 2)
    assert summary["per_cwe_vuln_fixed"]["CWE-78"] is None


def test_aggregate_all_none_check() -> None:
    summary = aggregate([_result(vuln_fixed=None), _result(vuln_fixed=None)])
    assert summary["vuln_fixed"] is None


# -------------------------------------------------------------------- markdown

def test_render_markdown_table() -> None:
    base = aggregate([_result(compiles=False, vuln_fixed=False, score=0.2)])
    tuned = aggregate([_result()])
    table = render_markdown(base, tuned, "mock:echo", "mock:gold")
    assert "| metric | base | tuned |" in table
    assert "| compile rate | 0% (0/1) | 100% (1/1) |" in table
    assert "| vuln-fixed rate | 0% (0/1) | 100% (1/1) |" in table
    assert "| CWE-89 | 0% (0/1) | 100% (1/1) |" in table


def test_render_markdown_handles_missing_tool() -> None:
    base = aggregate([_result(vuln_fixed=None)])
    table = render_markdown(base, base, "a", "b")
    assert "n/a (tool missing)" in table


# ------------------------------------------------------------------ test file io

def test_load_test_records(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(RECORD) + "\n" + json.dumps(RECORD) + "\n", encoding="utf-8")
    assert len(load_test_records(path)) == 2
    assert len(load_test_records(path, limit=1)) == 1


def test_load_test_records_missing_file() -> None:
    with pytest.raises(SystemExit, match="prepare_dataset"):
        load_test_records(Path("does/not/exist.jsonl"))
