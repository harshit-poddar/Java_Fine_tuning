"""Before/after eval runner for FINETUNING_001.

Runs a "base" and a "tuned" model over the test split, scores every generated
patch with eval/harness.py, and emits a markdown before/after metrics table
(the project deliverable).

Model specs (pluggable):
  mock:echo                 returns the vulnerable code unchanged (lower bound)
  mock:gold                 returns the reference fix (upper bound)
  mock:refuse               returns prose with no code (format failure)
  openai:<base_url>#<model> OpenAI-compatible endpoint, e.g. a vLLM server:
                            openai:http://localhost:8000/v1#qwen-base
  hf:<model_id>[#adapter=<path>]  local transformers load (pod only),
                            optionally with a LoRA adapter applied.

CPU acceptance run (no model, no GPU):
  python run_eval.py --base-spec mock:echo --tuned-spec mock:gold
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.eval import harness  # noqa: E402

GEN_TEMPERATURE = 0.0
GEN_MAX_TOKENS = 1024


class ModelClient(Protocol):
    """One side of the before/after comparison."""

    name: str

    def generate(self, system: str, user: str, record: dict) -> str:
        """Return the raw model response for one test record."""
        ...


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

_JAVA_BLOCK_RE = re.compile(r"```java\n(.*?)```", re.DOTALL)


class MockClient:
    """Canned responses so the whole pipeline runs on CPU without a model."""

    def __init__(self, mode: str) -> None:
        if mode not in ("echo", "gold", "refuse"):
            raise ValueError(f"unknown mock mode {mode!r} (use echo|gold|refuse)")
        self.mode = mode
        self.name = f"mock:{mode}"

    def generate(self, system: str, user: str, record: dict) -> str:
        if self.mode == "gold":
            return record["output"]
        if self.mode == "refuse":
            return "I cannot modify this code."
        match = _JAVA_BLOCK_RE.search(user)  # echo: vulnerable code, unchanged
        code = match.group(1) if match else record.get("vulnerable_code", "")
        return f"```java\n{code}\n```"


class OpenAICompatClient:
    """Talks to any OpenAI-compatible /chat/completions endpoint (e.g. vLLM)."""

    def __init__(self, base_url: str, model: str, timeout_s: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.name = f"openai:{model}"

    def generate(self, system: str, user: str, record: dict) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": GEN_TEMPERATURE,
                "max_tokens": GEN_MAX_TOKENS,
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class TransformersClient:
    """Local transformers load, optionally with a LoRA adapter (pod use)."""

    def __init__(self, model_id: str, adapter: Optional[str] = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="bfloat16" if torch.cuda.is_available() else "float32",
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.name = f"hf:{model_id}" + (f"+{adapter}" if adapter else "")

    def generate(self, system: str, user: str, record: dict) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=GEN_MAX_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)


def make_client(spec: str) -> ModelClient:
    """Parse a model spec string into a client (see module docstring)."""
    kind, _, rest = spec.partition(":")
    if kind == "mock":
        return MockClient(rest)
    if kind == "openai":
        base_url, _, model = rest.rpartition("#")
        if not base_url or not model:
            raise ValueError(
                f"bad openai spec {spec!r} - expected openai:<base_url>#<served_model_name>"
            )
        return OpenAICompatClient(base_url, model)
    if kind == "hf":
        model_id, _, adapter_part = rest.partition("#")
        adapter = None
        if adapter_part:
            if not adapter_part.startswith("adapter="):
                raise ValueError(f"bad hf spec {spec!r} - expected hf:<model>[#adapter=<path>]")
            adapter = adapter_part[len("adapter="):]
        return TransformersClient(model_id, adapter)
    raise ValueError(f"unknown model spec {spec!r} (use mock:|openai:|hf:)")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_test_records(path: Path, limit: Optional[int] = None) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Test file not found: {path} - run prepare_dataset.py first.")
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise SystemExit(f"{path} is empty.")
    return records


def evaluate_model(client: ModelClient, records: list[dict]) -> list[dict]:
    """Generate + score every record; returns per-item result dicts."""
    results = []
    for i, record in enumerate(records, 1):
        try:
            response = client.generate(record["system"], record["input"], record)
        except Exception as error:  # endpoint hiccups shouldn't kill a long run
            print(f"WARNING: [{client.name}] item {i} generation failed: {error}", file=sys.stderr)
            response = ""
        scores = harness.score_patch(response, record["cwe"])
        results.append(
            {
                "source_id": record.get("source_id", str(i)),
                "cwe": record["cwe"],
                "response": response,
                **{k: scores[k] for k in ("format_ok", "compiles", "vuln_fixed", "score")},
                "details": scores["details"],
            }
        )
        print(f"[{client.name}] {i}/{len(records)} score={scores['score']:.2f}", file=sys.stderr)
    return results


def _rate(results: list[dict], key: str) -> Optional[tuple[int, int]]:
    """(passed, evaluated) over items where the check ran; None if it never ran."""
    valid = [r[key] for r in results if r[key] is not None]
    if not valid:
        return None
    return sum(1 for v in valid if v), len(valid)


def aggregate(results: list[dict]) -> dict[str, Any]:
    return {
        "n": len(results),
        "format": _rate(results, "format_ok"),
        "compile": _rate(results, "compiles"),
        "vuln_fixed": _rate(results, "vuln_fixed"),
        "mean_score": sum(r["score"] for r in results) / len(results),
        "per_cwe_vuln_fixed": {
            cwe: _rate([r for r in results if r["cwe"] == cwe], "vuln_fixed")
            for cwe in sorted({r["cwe"] for r in results})
        },
    }


def _cell(rate: Optional[tuple[int, int]]) -> str:
    if rate is None:
        return "n/a (tool missing)"
    passed, total = rate
    return f"{100 * passed / total:.0f}% ({passed}/{total})"


def render_markdown(base: dict, tuned: dict, base_name: str, tuned_name: str) -> str:
    """The before/after table - this is the deliverable."""
    lines = [
        "# Before/after evaluation",
        "",
        f"- base:  `{base_name}`",
        f"- tuned: `{tuned_name}`",
        f"- test items: {base['n']}",
        "",
        "| metric | base | tuned |",
        "|---|---|---|",
        f"| format rate | {_cell(base['format'])} | {_cell(tuned['format'])} |",
        f"| compile rate | {_cell(base['compile'])} | {_cell(tuned['compile'])} |",
        f"| vuln-fixed rate | {_cell(base['vuln_fixed'])} | {_cell(tuned['vuln_fixed'])} |",
        f"| mean score | {base['mean_score']:.3f} | {tuned['mean_score']:.3f} |",
        "",
        "## Vuln-fixed rate per CWE",
        "",
        "| CWE | base | tuned |",
        "|---|---|---|",
    ]
    for cwe in sorted(set(base["per_cwe_vuln_fixed"]) | set(tuned["per_cwe_vuln_fixed"])):
        lines.append(
            f"| {cwe} | {_cell(base['per_cwe_vuln_fixed'].get(cwe))} "
            f"| {_cell(tuned['per_cwe_vuln_fixed'].get(cwe))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_test = Path(__file__).resolve().parents[1] / "data" / "processed" / "test.jsonl"
    parser.add_argument("--test-file", type=Path, default=default_test)
    parser.add_argument("--base-spec", default="mock:echo", help="base model spec")
    parser.add_argument("--tuned-spec", default="mock:gold", help="tuned model spec")
    parser.add_argument("--limit", type=int, default=None, help="evaluate at most N items")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    records = load_test_records(args.test_file, args.limit)
    print(f"evaluating {len(records)} test items", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    names = {}
    for side, spec in (("base", args.base_spec), ("tuned", args.tuned_spec)):
        client = make_client(spec)
        names[side] = client.name
        results = evaluate_model(client, records)
        summaries[side] = aggregate(results)
        dump = args.out_dir / f"{side}_results.jsonl"
        with dump.open("w", encoding="utf-8") as fh:
            for result in results:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"per-item results -> {dump}", file=sys.stderr)

    table = render_markdown(summaries["base"], summaries["tuned"], names["base"], names["tuned"])
    (args.out_dir / "results.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"table -> {args.out_dir / 'results.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
