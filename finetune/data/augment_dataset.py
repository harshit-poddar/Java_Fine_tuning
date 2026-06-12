"""Model-in-the-loop dataset augmentation for FINETUNING_001 (pod use).

Uses a served LLM (the base model on the MI300X via vLLM) to generate NEW
vulnerable/fixed Java pairs for the target CWEs, seeded with examples from
the existing train split. Every candidate is machine-verified by the eval
harness before it is admitted:

  - fixed_code must compile (javac)
  - vulnerable_code must trigger the CWE's semgrep rule
  - fixed_code must NOT trigger it
  (semgrep checks are skipped with a warning where semgrep is unavailable)

Verified pairs are written as a CSV into data/raw/, then re-run
prepare_dataset.py to dedup against existing data and rebuild the splits.

Typical pod usage (base model already being served, see serve_notes.md):
  python augment_dataset.py --endpoint "http://localhost:8000/v1#Qwen/Qwen2.5-Coder-32B-Instruct" --per-cwe 50
  python prepare_dataset.py

CPU dry run without any model:
  python augment_dataset.py --endpoint mock --per-cwe 3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.data.prepare_dataset import (  # noqa: E402
    _DEMO_TEMPLATES,
    CWE_DESCRIPTIONS,
    TARGET_CWES,
)
from finetune.eval.harness import HarnessConfig, check_compiles, check_vuln_fixed  # noqa: E402
from finetune.eval.run_eval import OpenAICompatClient  # noqa: E402

GEN_SYSTEM = (
    "You create training data for a security code-fixing model. You write "
    "realistic Java code the way an average enterprise developer would, "
    "including the security mistakes they actually make."
)

GEN_USER_TEMPLATE = """\
Write ONE new, self-contained Java example of {cwe} ({description}).

Requirements:
- A single public class using only JDK imports (java.*, javax.*), 20-60 lines.
- vulnerable_code: contains exactly one {cwe} vulnerability.
- fixed_code: the same class with ONLY the vulnerability fixed (canonical
  secure pattern), identical signatures otherwise. Both versions must compile.
- Do NOT copy the example below; vary the domain (different class purpose,
  names, structure).

Example for inspiration only:
```java
{seed}
```

Respond with ONLY a JSON object, no markdown fences:
{{"vulnerable_code": "...", "fixed_code": "..."}}
"""


class MockGenClient:
    """Offline generator for CPU dry runs: emits demo-template variants."""

    name = "mock"

    def __init__(self) -> None:
        self._counter = 0

    def generate(self, system: str, user: str, record: dict) -> str:
        cwe = record["cwe"]
        vuln_tmpl, fixed_tmpl = _DEMO_TEMPLATES[cwe]
        self._counter += 1
        variant = {
            "cls": f"Generated{self._counter}",
            "table": f"table_{self._counter}",
            "col": f"col_{self._counter}",
            "dir": f"dir_{self._counter}",
            "cmd": "ping",
        }
        return json.dumps(
            {
                "vulnerable_code": vuln_tmpl.format(**variant),
                "fixed_code": fixed_tmpl.format(**variant),
            }
        )


def extract_json_pair(text: str) -> Optional[dict]:
    """Pull {"vulnerable_code": ..., "fixed_code": ...} out of a model response."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    vuln, fixed = data.get("vulnerable_code"), data.get("fixed_code")
    if not vuln or not fixed or not isinstance(vuln, str) or not isinstance(fixed, str):
        return None
    return {"vulnerable_code": vuln, "fixed_code": fixed}


def verify_pair(vulnerable: str, fixed: str, cwe: str,
                config: HarnessConfig) -> tuple[bool, str]:
    """Harness gate: only verified pairs become training data."""
    if vulnerable.strip() == fixed.strip():
        return False, "vulnerable and fixed are identical"
    compiles, detail = check_compiles(fixed, config)
    if compiles is False:
        return False, f"fixed_code does not compile: {detail[:200]}"
    if compiles is None:
        print("WARNING: javac unavailable - admitting pair without compile check",
              file=sys.stderr)
    still_vuln, _ = check_vuln_fixed(vulnerable, cwe, config)
    fixed_clean, detail = check_vuln_fixed(fixed, cwe, config)
    if still_vuln is None or fixed_clean is None:
        print("WARNING: semgrep unavailable - admitting pair without vuln check",
              file=sys.stderr)
        return True, "compile-only (semgrep unavailable)"
    if still_vuln is True:
        return False, "vulnerable_code does not trigger the CWE rule (bad label)"
    if fixed_clean is False:
        return False, f"fixed_code still triggers the rule: {detail[:200]}"
    return True, "verified"


def load_seeds(seed_file: Path) -> dict[str, list[str]]:
    """vulnerable_code seeds per CWE from an existing split (fix records only)."""
    seeds: dict[str, list[str]] = {cwe: [] for cwe in TARGET_CWES}
    if not seed_file.exists():
        raise SystemExit(f"Seed file not found: {seed_file} - run prepare_dataset.py first.")
    with seed_file.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("kind", "fix") == "fix" and record.get("cwe") in seeds:
                seeds[record["cwe"]].append(record["vulnerable_code"])
    return seeds


def augment(client, seeds: dict[str, list[str]], per_cwe: int, seed: int,
            harness_config: Optional[HarnessConfig] = None) -> tuple[list[dict], dict]:
    """Generate + verify pairs. Returns (rows, stats)."""
    config = harness_config or HarnessConfig()
    rng = random.Random(seed)
    rows: list[dict] = []
    stats = {"generated": 0, "unparseable": 0, "rejected": 0, "admitted": 0}
    for cwe in TARGET_CWES:
        if not seeds.get(cwe):
            print(f"WARNING: no seeds for {cwe} - skipping", file=sys.stderr)
            continue
        admitted = 0
        attempts = 0
        max_attempts = per_cwe * 3  # the model won't pass verification every time
        while admitted < per_cwe and attempts < max_attempts:
            attempts += 1
            stats["generated"] += 1
            user = GEN_USER_TEMPLATE.format(
                cwe=cwe, description=CWE_DESCRIPTIONS[cwe], seed=rng.choice(seeds[cwe])
            )
            try:
                response = client.generate(GEN_SYSTEM, user, {"cwe": cwe})
            except Exception as error:
                print(f"WARNING: generation failed for {cwe}: {error}", file=sys.stderr)
                continue
            pair = extract_json_pair(response)
            if pair is None:
                stats["unparseable"] += 1
                continue
            ok, reason = verify_pair(pair["vulnerable_code"], pair["fixed_code"], cwe, config)
            if not ok:
                stats["rejected"] += 1
                print(f"[{cwe}] rejected: {reason}", file=sys.stderr)
                continue
            admitted += 1
            stats["admitted"] += 1
            rows.append({**pair, "cwe": cwe})
            print(f"[{cwe}] admitted {admitted}/{per_cwe} ({reason})", file=sys.stderr)
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path(__file__).resolve().parent
    parser.add_argument("--endpoint", required=True,
                        help='"mock" or "<base_url>#<served_model>", e.g. '
                             '"http://localhost:8000/v1#Qwen/Qwen2.5-Coder-32B-Instruct"')
    parser.add_argument("--seed-file", type=Path, default=base / "processed" / "train.jsonl")
    parser.add_argument("--per-cwe", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="sampling temperature for generation diversity")
    parser.add_argument("--out", type=Path, default=base / "raw" / "augmented_pairs.csv")
    args = parser.parse_args()

    if args.endpoint == "mock":
        client = MockGenClient()
    else:
        base_url, _, model = args.endpoint.rpartition("#")
        if not base_url or not model:
            raise SystemExit(f'bad --endpoint {args.endpoint!r} - expected "<base_url>#<model>"')
        client = OpenAICompatClient(base_url, model, temperature=args.temperature)

    rows, stats = augment(client, load_seeds(args.seed_file), args.per_cwe, args.seed)
    if not rows:
        raise SystemExit(f"No pairs survived verification: {stats}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["vulnerable_code", "fixed_code", "cwe"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{json.dumps(stats)}")
    print(f"wrote {len(rows)} verified pairs to {args.out}")
    print("now re-run prepare_dataset.py to dedup + rebuild the splits")


if __name__ == "__main__":
    main()
