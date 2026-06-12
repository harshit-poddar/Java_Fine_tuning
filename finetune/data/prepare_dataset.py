"""Dataset preparation for FINETUNING_001.

Reads raw vulnerability/fix pairs from `data/raw/`, cleans them, converts them
to instruction triplets `{system, input, output}` and writes train/val/test
JSONL splits plus a stats report to `data/processed/`.

Input adapters (pluggable, auto-detected by default):
  - csv:    any `*.csv` in raw dir with columns vulnerable_code, fixed_code, cwe
  - juliet: NIST Juliet Java layout (CWE89_* / CWE78_* / CWE23|36_* .java test
            files); bad()/goodG2B() methods are extracted best-effort and
            rewrapped — pairs that do not survive the javac filter are dropped.
  - cvefixes: real-world CVE fix pairs (func_before -> func_after) for Java.
            Source is either a HuggingFace dataset (set env CVEFIXES_HF=<id>)
            or local `*cvefixes*.{jsonl,json,parquet}` files in the raw dir.
            Schema-flexible; every pair still runs the javac + semgrep filters,
            so noisy real-world rows are dropped, not trained on.

Cleaning: filter to target CWEs, exact dedup, drop pairs whose fixed_code
fails javac (skipped with a warning when javac is absent), cap length.
Splits are near-duplicate-aware: similar items always land in the same split.

CPU-only. Try it end to end with no real data:
    python prepare_dataset.py --make-demo-data
    python prepare_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Optional

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.eval.harness import HarnessConfig, check_compiles, check_vuln_fixed  # noqa: E402

TARGET_CWES = ("CWE-89", "CWE-22", "CWE-78")

CWE_DESCRIPTIONS: dict[str, str] = {
    "CWE-89": "SQL Injection: untrusted input is concatenated into a SQL query",
    "CWE-22": "Path Traversal: untrusted input is used to build a filesystem path",
    "CWE-78": "OS Command Injection: untrusted input reaches an OS command",
}

#: Shared by run_eval.py — the tuned and base model must be prompted identically.
SYSTEM_PROMPT = (
    "You are a senior application-security engineer. You fix vulnerabilities in "
    "Java code. Reply with ONLY the complete fixed Java code inside a single "
    "```java code block. Keep the original class and method signatures; change "
    "only what is needed to remove the vulnerability. If the given code is "
    "already secure, return it unchanged."
)

# Juliet uses child CWEs for path traversal; fold them into our target id.
JULIET_CWE_MAP = {"89": "CWE-89", "78": "CWE-78", "22": "CWE-22", "23": "CWE-22", "36": "CWE-22"}


class PrepConfig(BaseModel):
    """Knobs for cleaning/splitting. Every field is env-overridable."""

    max_chars: int = Field(default_factory=lambda: int(os.getenv("PREP_MAX_CHARS", "6000")))
    seed: int = Field(default_factory=lambda: int(os.getenv("PREP_SEED", "42")))
    near_dup_jaccard: float = Field(default_factory=lambda: float(os.getenv("PREP_NEAR_DUP_JACCARD", "0.85")))
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    compile_filter: bool = True
    #: parallel workers for the javac/semgrep verification (subprocess-bound,
    #: so threads scale to physical cores; 112 on the pod)
    workers: int = Field(default_factory=lambda: int(os.getenv("PREP_WORKERS", str(min(32, os.cpu_count() or 4)))))
    #: verify labels with the SAME semgrep rules the eval uses: vulnerable
    #: side must trigger the CWE rule, fixed side must not. Needs semgrep
    #: (pod/Linux) - skipped with a warning elsewhere.
    semgrep_filter: bool = Field(default_factory=lambda: os.getenv("PREP_SEMGREP_FILTER", "0") == "1")
    #: share of each split duplicated as already-secure "return it unchanged"
    #: examples - teaches the model restraint (no fixing secure code)
    negatives_frac: float = Field(default_factory=lambda: float(os.getenv("PREP_NEGATIVES_FRAC", "0.10")))
    #: cap on general instruction data mixed into TRAIN (anti-forgetting replay)
    replay_frac: float = Field(default_factory=lambda: float(os.getenv("PREP_REPLAY_FRAC", "0.15")))


class RawPair(BaseModel):
    vulnerable_code: str
    fixed_code: str
    cwe: str
    source_id: str
    kind: str = "fix"  # "fix" | "no_change"


def build_user_prompt(vulnerable_code: str, cwe: str) -> str:
    """The `input` field of a triplet. Also used verbatim by the eval runner."""
    return (
        f"Review the following Java code for a {cwe} vulnerability "
        f"({CWE_DESCRIPTIONS[cwe]}). If the vulnerability is present, fix it. "
        "If the code is already secure, return it unchanged.\n\n"
        f"```java\n{vulnerable_code.rstrip()}\n```"
    )


def make_negative(pair: RawPair) -> RawPair:
    """An already-secure example: input is the FIXED code, expected output is
    the same code unchanged. Derived in-split, so it can never leak the fix of
    a test item into training."""
    return pair.model_copy(
        update={
            "vulnerable_code": pair.fixed_code,
            "kind": "no_change",
            "source_id": pair.source_id + "#neg",
        }
    )


def _normalize_cwe_label(raw: str) -> Optional[str]:
    """'cwe-89'/'89' -> 'CWE-89'; None when out of scope (caller drops the pair)."""
    match = re.search(r"(\d+)", str(raw))
    if not match:
        return None
    canonical = JULIET_CWE_MAP.get(match.group(1))
    return canonical if canonical in TARGET_CWES else None


# ---------------------------------------------------------------------------
# Input adapters
# ---------------------------------------------------------------------------

def parse_csv_sources(raw_dir: Path) -> list[RawPair]:
    """Adapter: every *.csv in raw_dir with vulnerable_code/fixed_code/cwe columns."""
    pairs: list[RawPair] = []
    for csv_path in sorted(raw_dir.rglob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            required = {"vulnerable_code", "fixed_code", "cwe"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                print(
                    f"WARNING: {csv_path.name}: missing columns "
                    f"{sorted(required - set(reader.fieldnames or []))} - file skipped",
                    file=sys.stderr,
                )
                continue
            for i, row in enumerate(reader):
                pairs.append(
                    RawPair(
                        vulnerable_code=row["vulnerable_code"],
                        fixed_code=row["fixed_code"],
                        cwe=row["cwe"],
                        source_id=f"{csv_path.name}:{i}",
                    )
                )
    return pairs


_JULIET_FILE_RE = re.compile(r"CWE(\d+)_\w*?\.java$", re.IGNORECASE)
_METHOD_RE_TMPL = r"(?:public|private|protected)\s+(?:static\s+)?void\s+{name}\s*\([^)]*\)(?:\s*throws\s+[\w.,\s]+)?\s*\{{"


def _extract_method(source: str, name_regex: str) -> Optional[str]:
    """Return the full text of the first method whose name matches, via brace matching."""
    match = re.search(_METHOD_RE_TMPL.format(name=name_regex), source)
    if not match:
        return None
    start, depth = match.start(), 0
    for pos in range(match.end() - 1, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start : pos + 1]
    return None


#: Self-contained, compile-safe stand-ins for Juliet's testcasesupport.IO
#: helper library. Order matters: longer member names first so prefixes
#: (staticReturnsTrueOrFalse vs staticReturnsTrue) don't mis-replace.
_JULIET_IO_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("IO.writeLine(", "System.out.println("),
    ("IO.writeString(", "System.out.print("),
    ("IO.getDBConnection()",
     'java.sql.DriverManager.getConnection("jdbc:default:connection")'),
    ("IO.logger", "java.util.logging.Logger.getGlobal()"),
    ("IO.staticReturnsTrueOrFalse()", "(System.currentTimeMillis() % 2 == 0)"),
    ("IO.staticReturnsTrue()", 'Boolean.parseBoolean("true")'),
    ("IO.staticReturnsFalse()", 'Boolean.parseBoolean("false")'),
    ("IO.STATIC_FINAL_TRUE", "true"),
    ("IO.STATIC_FINAL_FALSE", "false"),
    ("IO.staticTrue", 'Boolean.parseBoolean("true")'),
    ("IO.staticFalse", 'Boolean.parseBoolean("false")'),
    ("IO.STATIC_FINAL_FIVE", "5"),
    ("IO.staticFive", 'Integer.parseInt("5")'),
)


def _juliet_strip(source: str) -> str:
    """Remove Juliet boilerplate so extracted methods can stand alone."""
    source = re.sub(r"^\s*package\s+[\w.]+\s*;\s*$", "", source, flags=re.MULTILINE)
    source = re.sub(r"^\s*import\s+testcasesupport\.[\w.*]+\s*;\s*$", "", source, flags=re.MULTILINE)
    source = re.sub(r"\bextends\s+AbstractTestCase(Base)?\b", "", source)
    for needle, replacement in _JULIET_IO_REPLACEMENTS:
        source = source.replace(needle, replacement)
    return source


def _wrap_method(method_src: str, imports: list[str], class_name: str) -> str:
    body = "\n".join(imports) + ("\n\n" if imports else "")
    return f"{body}public class {class_name} {{\n\n{method_src}\n}}\n"


#: Juliet "good" variants, in preference order. B2G ("bad source to good
#: sink") fixes the SINK - the secure-coding lesson we want to teach. G2B
#: ("good source to bad sink") merely swaps the input for a safe constant
#: while the sink stays vulnerable - a wrong lesson, used only as a last
#: resort and reliably killed later by the --semgrep-filter.
_JULIET_GOOD_VARIANTS = (
    (r"goodB2G\d*", "B2G"),
    (r"good1", "good1"),
    (r"goodG2B\d*", "G2B"),
)
_JULIET_GOOD_NAME_RE = re.compile(r"\bgood(?:B2G|G2B)?\d*\b")


# ---------------------------------------------------------------------------
# Sink-fix synthesis (CWE-78, CWE-22)
# ---------------------------------------------------------------------------
# Juliet Java ships NO sink-level fix for command injection or path
# traversal: every good() variant there is G2B - the tainted source is
# swapped for a constant while the vulnerable sink stays, which is the wrong
# lesson and is rightly rejected by --semgrep-filter (measured: 444/444
# CWE-78 and 888/888 CWE-22 pairs are G2B; CWE-89 is 2220/2220 B2G).
#
# So for these two CWEs we keep Juliet's bad() methods (rich source/flow
# variety) and SYNTHESIZE the fixed side by rewriting the one sink line into
# the secure pattern. Synthesized fixes go through the same javac and semgrep
# filters as every other pair, so a transform that misfires is dropped, not
# trained on.

SYNTH_CWES = ("CWE-78", "CWE-22")

_CWE78_EXEC_RE = re.compile(
    r"^(?P<i>[ \t]*)Process (?P<p>\w+) = Runtime\.getRuntime\(\)\.exec\("
    r"(?P<cmd>\w+) \+ (?P<data>\w+)\);[ \t]*$",
    re.MULTILINE,
)

_CWE22_CONCAT_RE = re.compile(
    r"^(?P<i>[ \t]*)File (?P<f>\w+) = new File\((?P<root>\w+) \+ (?P<data>\w+)\);[ \t]*$",
    re.MULTILINE,
)

_CWE22_DIRECT_RE = re.compile(
    r"^(?P<i>[ \t]*)File (?P<f>\w+) = new File\((?P<data>data)\);[ \t]*$",
    re.MULTILINE,
)


def _cwe78_fix(match: re.Match) -> str:
    """Allow-list the argument, then run it as its own ProcessBuilder token."""
    i, p, cmd, data = match.group("i", "p", "cmd", "data")
    return (
        f'{i}if (!{data}.matches("[A-Za-z0-9._-]+"))\n'
        f"{i}{{\n"
        f'{i}    throw new IllegalArgumentException("unsafe command argument: " + {data});\n'
        f"{i}}}\n"
        f"{i}java.util.List<String> commandTokens = new java.util.ArrayList<String>("
        f'java.util.Arrays.asList({cmd}.trim().split("\\\\s+")));\n'
        f"{i}commandTokens.add({data});\n"
        f"{i}Process {p} = new ProcessBuilder(commandTokens).start();"
    )


def _cwe22_guard(i: str, f: str, base_expr: str, data: str) -> str:
    """Resolve against a fixed base dir and reject paths that escape it."""
    return (
        f"{i}File allowedDir = new File({base_expr});\n"
        f"{i}File {f} = new File(allowedDir, {data});\n"
        f"{i}if (!{f}.toPath().normalize().startsWith(allowedDir.toPath().normalize()))\n"
        f"{i}{{\n"
        f'{i}    throw new IllegalArgumentException("path escapes the allowed directory: " + {data});\n'
        f"{i}}}"
    )


def _cwe22_concat_fix(match: re.Match) -> str:
    i, f, root, data = match.group("i", "f", "root", "data")
    return _cwe22_guard(i, f, root, data)


def _cwe22_direct_fix(match: re.Match) -> str:
    i, f, data = match.group("i", "f", "data")
    return _cwe22_guard(i, f, 'System.getProperty("user.dir")', data)


_SINK_SYNTHESIZERS: dict[str, tuple[tuple[re.Pattern, Callable], ...]] = {
    "CWE-78": ((_CWE78_EXEC_RE, _cwe78_fix),),
    "CWE-22": ((_CWE22_CONCAT_RE, _cwe22_concat_fix), (_CWE22_DIRECT_RE, _cwe22_direct_fix)),
}


def synthesize_fixed_code(cwe: str, vulnerable_code: str) -> Optional[str]:
    """Rewrite the first recognized sink in `vulnerable_code` into its secure
    form. Returns None when no known sink shape matches (caller drops the
    pair). Any extra unrecognized sinks survive untouched and are caught by
    the semgrep filter."""
    for pattern, builder in _SINK_SYNTHESIZERS.get(cwe, ()):
        fixed, n = pattern.subn(builder, vulnerable_code, count=1)
        if n:
            return fixed
    return None


def parse_juliet_sources(raw_dir: Path) -> list[RawPair]:
    """Adapter: NIST Juliet Java testcases. Best-effort method-level extraction;
    pairs whose rewrapped code does not compile are dropped later by the
    javac filter.

    The fixed side comes from two places:
      - CWE-89: Juliet's own good methods (all B2G - real PreparedStatement
        sink fixes), preferred per _JULIET_GOOD_VARIANTS.
      - CWE-78 / CWE-22: synthesized from bad() (see synthesize_fixed_code) -
        Juliet has no sink-level fixes for these, only G2B constant swaps.
    """
    pairs: list[RawPair] = []
    for java_path in sorted(raw_dir.rglob("*.java")):
        match = _JULIET_FILE_RE.search(java_path.name)
        if not match:
            continue
        cwe = JULIET_CWE_MAP.get(match.group(1))
        if cwe not in TARGET_CWES:
            continue
        source = _juliet_strip(java_path.read_text(encoding="utf-8", errors="replace"))
        if re.search(r"\bIO\.", source):
            continue  # uses an IO helper we have no stand-in for
        # java.* only: javax.servlet etc. are not in the JDK, and importing a
        # missing package is a hard compile error even when unused
        imports = [
            line.strip()
            for line in source.splitlines()
            if re.match(r"^\s*import\s+java\.[\w.*]+\s*;\s*$", line)
        ]
        bad = _extract_method(source, "bad")
        if not bad:
            continue
        stem = re.sub(r"\W", "_", java_path.stem)
        vulnerable = _wrap_method(bad, imports, f"{stem}_case")

        if cwe in SYNTH_CWES:
            fixed = synthesize_fixed_code(cwe, vulnerable)
            variant = "synth"
            if fixed is None:
                continue  # no recognized sink shape in bad()
        else:
            good, variant = None, None
            for pattern, label in _JULIET_GOOD_VARIANTS:
                good = _extract_method(source, pattern)
                if good:
                    variant = label
                    break
            if not good:
                continue
            fixed = _wrap_method(_JULIET_GOOD_NAME_RE.sub("bad", good),
                                 imports, f"{stem}_case")

        pairs.append(
            RawPair(
                vulnerable_code=vulnerable,
                fixed_code=fixed,
                cwe=cwe,
                source_id=f"{java_path.relative_to(raw_dir)}:{variant}",
            )
        )
    return pairs


# ---------------------------------------------------------------------------
# CVEfixes adapter (real-world CVE fix pairs)
# ---------------------------------------------------------------------------
# CVEfixes (Bhandari et al.) is method-level before/after code drawn from the
# commits that fixed real CVEs. There is no single canonical HuggingFace
# schema, so field names are matched flexibly. Two sources are supported:
#   1. env CVEFIXES_HF=<dataset_id>  -> loaded via `datasets` (split via
#      CVEFIXES_SPLIT, default "train").
#   2. local files matching `*cvefixes*.{jsonl,json,parquet}` in the raw dir.
# Either way the rows pass through the same javac + semgrep filters as every
# other pair, so real-world noise (won't-compile, label-disagrees) is dropped.

_CVEFIXES_LANG_FIELDS = ("programming_language", "lang", "language", "file_language")
_CVEFIXES_BEFORE_FIELDS = ("func_before", "code_before", "before", "vulnerable_code",
                           "source_before", "before_change", "vul_func")
_CVEFIXES_AFTER_FIELDS = ("func_after", "code_after", "after", "fixed_code",
                          "source_after", "after_change", "fix_func")
_CVEFIXES_CWE_FIELDS = ("cwe_id", "cwe", "cwe_ids", "cwes", "cwe_name")
_CVEFIXES_ID_FIELDS = ("cve_id", "cve", "commit_id", "hash", "id")


def _first_field(row: dict, fields: tuple[str, ...]) -> Optional[object]:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return None


def _cvefixes_is_java(row: dict) -> bool:
    """True for Java rows (and rows with no language field - CWE+filters decide)."""
    lang = _first_field(row, _CVEFIXES_LANG_FIELDS)
    if lang is None:
        return True
    text = str(lang).lower()
    return "java" in text and "javascript" not in text


def _cvefixes_cwe(row: dict) -> Optional[str]:
    """First in-scope target CWE mentioned in the row's CWE field, else None."""
    raw = _first_field(row, _CVEFIXES_CWE_FIELDS)
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        text = " ".join(str(x) for x in raw)
    else:
        text = str(raw)
    for num in re.findall(r"\d+", text):
        canonical = JULIET_CWE_MAP.get(num)
        if canonical in TARGET_CWES:
            return canonical
    return None


def _cvefixes_iter_rows(raw_dir: Path) -> Iterable[dict]:
    """Yield raw record dicts from HF (env CVEFIXES_HF) and/or local files."""
    hf_id = os.getenv("CVEFIXES_HF", "").strip()
    if hf_id:
        try:
            from datasets import load_dataset  # lazy: optional dependency
        except ImportError:
            print("WARNING: CVEFIXES_HF is set but the `datasets` package is not "
                  "installed - pip install datasets. Skipping HF source.", file=sys.stderr)
        else:
            split = os.getenv("CVEFIXES_SPLIT", "train")
            try:
                dataset = load_dataset(hf_id, split=split)
                for row in dataset:
                    yield dict(row)
            except Exception as exc:  # network / 404 / bad split - actionable, non-fatal
                print(f"WARNING: could not load HF dataset {hf_id!r} (split {split!r}): "
                      f"{exc}. Skipping HF source.", file=sys.stderr)

    for path in sorted(raw_dir.rglob("*cvefixes*")):
        suffix = path.suffix.lower()
        try:
            if suffix == ".jsonl":
                with path.open(encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            yield json.loads(line)
            elif suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                yield from (data if isinstance(data, list) else [data])
            elif suffix == ".parquet":
                try:
                    import pandas as pd  # lazy: optional dependency
                except ImportError:
                    print(f"WARNING: {path.name} needs pandas+pyarrow to read - "
                          "pip install pandas pyarrow. Skipping file.", file=sys.stderr)
                    continue
                for record in pd.read_parquet(path).to_dict(orient="records"):
                    yield record
        except Exception as exc:
            print(f"WARNING: failed to read {path.name}: {exc} - skipping", file=sys.stderr)


def parse_cvefixes_sources(raw_dir: Path) -> list[RawPair]:
    """Adapter: real-world CVE fix pairs for Java, from HF and/or local files.

    Returns [] (no warning) when no CVEfixes source is configured, so `auto`
    mode stays clean on Juliet-only runs."""
    pairs: list[RawPair] = []
    seen_ids = 0
    for row in _cvefixes_iter_rows(raw_dir):
        if not _cvefixes_is_java(row):
            continue
        cwe = _cvefixes_cwe(row)
        if cwe is None:
            continue
        before = _first_field(row, _CVEFIXES_BEFORE_FIELDS)
        after = _first_field(row, _CVEFIXES_AFTER_FIELDS)
        if not before or not after:
            continue
        before, after = str(before), str(after)
        if before.strip() == after.strip():
            continue  # method-level change was elsewhere; no signal here
        ident = _first_field(row, _CVEFIXES_ID_FIELDS) or seen_ids
        seen_ids += 1
        pairs.append(
            RawPair(
                vulnerable_code=before,
                fixed_code=after,
                cwe=cwe,
                source_id=f"cvefixes:{ident}:{seen_ids}",
            )
        )
    if pairs:
        print(f"cvefixes adapter: {len(pairs)} candidate Java pairs in scope "
              "(pre-filter)")
    return pairs


PARSERS: dict[str, Callable[[Path], list[RawPair]]] = {
    "csv": parse_csv_sources,
    "juliet": parse_juliet_sources,
    "cvefixes": parse_cvefixes_sources,
}


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _code_fingerprint(*codes: str) -> str:
    normalized = "\x00".join(re.sub(r"\s+", " ", c).strip() for c in codes)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _shingles(code: str, n: int = 4) -> frozenset[tuple[str, ...]]:
    tokens = re.findall(r"\w+", code.lower())
    if len(tokens) < n:
        return frozenset({tuple(tokens)})
    return frozenset(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def clean_pairs(pairs: list[RawPair], config: PrepConfig) -> tuple[list[RawPair], dict[str, int]]:
    """Filter to target CWEs, dedup, length-cap, and javac-check fixed_code."""
    drops = {"cwe_out_of_scope": 0, "empty": 0, "duplicate": 0, "too_long": 0,
             "fixed_fails_javac": 0, "vuln_not_flagged": 0, "fixed_still_flagged": 0}
    seen: set[str] = set()
    kept: list[RawPair] = []

    harness_cfg = HarnessConfig()
    compile_filter = config.compile_filter
    if compile_filter:
        probe, _ = check_compiles("public class _Probe {}", harness_cfg)
        if probe is None:  # javac missing -> degrade gracefully, keep pairs
            print("WARNING: javac not found - compile filter SKIPPED for this run", file=sys.stderr)
            compile_filter = False
    semgrep_filter = config.semgrep_filter
    if semgrep_filter:
        probe, _ = check_vuln_fixed("public class _Probe {}", TARGET_CWES[0], harness_cfg)
        if probe is None:  # semgrep missing -> degrade gracefully
            print("WARNING: semgrep not found - semgrep filter SKIPPED for this run "
                  "(run it on the pod before training)", file=sys.stderr)
            semgrep_filter = False

    # Phase 1 (sequential, cheap): scope, empty, dedup, length. Dedup is
    # order-dependent, so it must stay sequential for determinism.
    candidates: list[RawPair] = []
    for pair in pairs:
        cwe = _normalize_cwe_label(pair.cwe)
        if cwe is None:
            drops["cwe_out_of_scope"] += 1
            continue
        if not pair.vulnerable_code.strip() or not pair.fixed_code.strip():
            drops["empty"] += 1
            continue
        fingerprint = _code_fingerprint(pair.vulnerable_code, pair.fixed_code)
        if fingerprint in seen:
            drops["duplicate"] += 1
            continue
        triplet_len = (
            len(SYSTEM_PROMPT)
            + len(build_user_prompt(pair.vulnerable_code, cwe))
            + len(pair.fixed_code)
        )
        if triplet_len > config.max_chars:
            drops["too_long"] += 1
            continue
        seen.add(fingerprint)
        candidates.append(pair.model_copy(update={"cwe": cwe}))

    if not (compile_filter or semgrep_filter):
        return candidates, drops

    # Phase 2 (parallel): javac/semgrep verification. Subprocess-bound, so
    # threads give real parallelism; results keep input order (ex.map).
    def verify(pair: RawPair) -> Optional[str]:
        """Returns the drop-reason key, or None to keep the pair."""
        if compile_filter:
            compiles, _ = check_compiles(pair.fixed_code, harness_cfg)
            if compiles is False:
                return "fixed_fails_javac"
        if semgrep_filter:
            vuln_clean, _ = check_vuln_fixed(pair.vulnerable_code, pair.cwe, harness_cfg)
            if vuln_clean is True:  # "vulnerable" side doesn't trigger the rule
                return "vuln_not_flagged"
            fixed_clean, _ = check_vuln_fixed(pair.fixed_code, pair.cwe, harness_cfg)
            if fixed_clean is False:  # "fixed" side still vulnerable (e.g. G2B)
                return "fixed_still_flagged"
        return None

    print(f"verifying {len(candidates)} pairs with {config.workers} parallel workers "
          f"(javac={compile_filter}, semgrep={semgrep_filter}) ...", flush=True)
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        verdicts = list(executor.map(verify, candidates))
    for pair, verdict in zip(candidates, verdicts):
        if verdict is None:
            kept.append(pair)
        else:
            drops[verdict] += 1
    return kept, drops


# ---------------------------------------------------------------------------
# Near-duplicate-aware splitting
# ---------------------------------------------------------------------------

def _cluster_near_dups(pairs: list[RawPair], threshold: float) -> list[list[int]]:
    """Union-find clusters of pairs whose vulnerable_code shingle-Jaccard >= threshold."""
    n = len(pairs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    shingles = [_shingles(p.vulnerable_code) for p in pairs]
    if n > 3000:
        print(f"WARNING: O(n^2) near-dup scan over {n} pairs - this may take a while", file=sys.stderr)
    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard(shingles[i], shingles[j]) >= threshold:
                parent[find(i)] = find(j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def split_pairs(pairs: list[RawPair], config: PrepConfig) -> dict[str, list[RawPair]]:
    """~80/10/10 split that never places near-duplicates in different splits."""
    clusters = _cluster_near_dups(pairs, config.near_dup_jaccard)
    rng = random.Random(config.seed)
    rng.shuffle(clusters)

    total = len(pairs)
    names = ("train", "val", "test")
    deficits = {name: frac * total for name, frac in zip(names, config.fractions)}
    splits: dict[str, list[RawPair]] = {name: [] for name in names}

    # Guarantee val/test are non-empty when we have enough clusters.
    if len(clusters) >= 3:
        clusters.sort(key=len)
        for name in ("test", "val"):
            cluster = clusters.pop(0)
            splits[name].extend(pairs[i] for i in cluster)
            deficits[name] -= len(cluster)
        rng.shuffle(clusters)

    for cluster in sorted(clusters, key=len, reverse=True):
        target = max(names, key=lambda name: deficits[name])
        splits[target].extend(pairs[i] for i in cluster)
        deficits[target] -= len(cluster)

    for name in names:
        rng.shuffle(splits[name])
    return splits


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def to_record(pair: RawPair) -> dict:
    """Instruction triplet + metadata used by the eval runner."""
    return {
        "system": SYSTEM_PROMPT,
        "input": build_user_prompt(pair.vulnerable_code, pair.cwe),
        "output": f"```java\n{pair.fixed_code.rstrip()}\n```",
        "cwe": pair.cwe,
        "vulnerable_code": pair.vulnerable_code,
        "source_id": pair.source_id,
        "kind": pair.kind,
    }


def load_replay_records(path: Path) -> list[dict]:
    """General instruction data ({system, input, output} JSONL) for replay mixing."""
    records = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            raw = json.loads(line)
            missing = [k for k in ("system", "input", "output") if not raw.get(k)]
            if missing:
                raise SystemExit(f"{path}:{i + 1}: replay record missing keys {missing}")
            records.append(
                {
                    "system": raw["system"],
                    "input": raw["input"],
                    "output": raw["output"],
                    "cwe": "general",
                    "vulnerable_code": "",
                    "source_id": f"{path.name}:{i}",
                    "kind": "replay",
                }
            )
    if not records:
        raise SystemExit(f"Replay file {path} is empty.")
    return records


def write_splits(records_by_split: dict[str, list[dict]], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"splits": {}}
    for name, records in records_by_split.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        row: dict = {"total": len(records)}
        for cwe in sorted({r["cwe"] for r in records}):
            row[cwe] = sum(1 for r in records if r["cwe"] == cwe)
        for kind in ("no_change", "replay"):
            row[kind] = sum(1 for r in records if r["kind"] == kind)
        stats["splits"][name] = row
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def print_stats(stats: dict, drops: dict[str, int]) -> None:
    print("\n== dataset stats ==")
    cwes = sorted(
        {key for row in stats["splits"].values() for key in row}
        - {"total", "no_change", "replay"}
    )
    columns = [*cwes, "no_change", "replay", "total"]
    print(f"{'split':<8}" + "".join(f"{c:>11}" for c in columns))
    for name, row in stats["splits"].items():
        print(f"{name:<8}" + "".join(f"{row.get(c, 0):>11}" for c in columns))
    print("\ndropped during cleaning: " + json.dumps(drops))


# ---------------------------------------------------------------------------
# Demo data (synthetic, compiles with a bare JDK) for end-to-end dry runs
# ---------------------------------------------------------------------------

_DEMO_TEMPLATES: dict[str, tuple[str, str]] = {
    "CWE-89": (
        """import java.sql.*;

public class {cls} {{
    public ResultSet find(Connection conn, String value) throws SQLException {{
        Statement st = conn.createStatement();
        String query = "SELECT * FROM {table} WHERE {col} = '" + value + "'";
        return st.executeQuery(query);
    }}
}}
""",
        """import java.sql.*;

public class {cls} {{
    public ResultSet find(Connection conn, String value) throws SQLException {{
        PreparedStatement ps = conn.prepareStatement("SELECT * FROM {table} WHERE {col} = ?");
        ps.setString(1, value);
        return ps.executeQuery();
    }}
}}
""",
    ),
    "CWE-22": (
        """import java.io.*;

public class {cls} {{
    private static final String BASE = "/srv/{dir}/";

    public String read(String name) throws IOException {{
        BufferedReader reader = new BufferedReader(new FileReader(new File(BASE + name)));
        try {{
            return reader.readLine();
        }} finally {{
            reader.close();
        }}
    }}
}}
""",
        """import java.io.*;

public class {cls} {{
    private static final String BASE = "/srv/{dir}/";

    public String read(String name) throws IOException {{
        File base = new File(BASE);
        File target = new File(base, name);
        if (!target.getCanonicalPath().startsWith(base.getCanonicalPath() + File.separator)) {{
            throw new IOException("path escapes base directory");
        }}
        BufferedReader reader = new BufferedReader(new FileReader(target));
        try {{
            return reader.readLine();
        }} finally {{
            reader.close();
        }}
    }}
}}
""",
    ),
    "CWE-78": (
        """import java.io.IOException;

public class {cls} {{
    public Process run(String target) throws IOException {{
        return Runtime.getRuntime().exec("{cmd} " + target);
    }}
}}
""",
        """import java.io.IOException;

public class {cls} {{
    public Process run(String target) throws IOException {{
        if (!target.matches("[A-Za-z0-9._-]+")) {{
            throw new IllegalArgumentException("invalid argument: " + target);
        }}
        ProcessBuilder pb = new ProcessBuilder("{cmd}", target);
        return pb.start();
    }}
}}
""",
    ),
}

_DEMO_VARIANTS: dict[str, list[dict[str, str]]] = {
    "CWE-89": [
        {"cls": "UserLookup", "table": "users", "col": "username"},
        {"cls": "OrderSearch", "table": "orders", "col": "customer_id"},
        {"cls": "ProductFinder", "table": "products", "col": "sku"},
        {"cls": "SessionStore", "table": "sessions", "col": "token"},
        {"cls": "AuditQuery", "table": "audit_log", "col": "actor"},
        {"cls": "InvoiceLookup", "table": "invoices", "col": "number"},
    ],
    "CWE-22": [
        {"cls": "ReportReader", "dir": "reports"},
        {"cls": "AvatarLoader", "dir": "avatars"},
        {"cls": "TemplateStore", "dir": "templates"},
        {"cls": "LogFetcher", "dir": "logs"},
        {"cls": "ExportReader", "dir": "exports"},
        {"cls": "AttachmentStore", "dir": "attachments"},
    ],
    "CWE-78": [
        {"cls": "PingTool", "cmd": "ping"},
        {"cls": "TraceTool", "cmd": "traceroute"},
        {"cls": "DnsLookup", "cmd": "nslookup"},
        {"cls": "WhoisClient", "cmd": "whois"},
        {"cls": "HostResolver", "cmd": "host"},
        {"cls": "DigClient", "cmd": "dig"},
    ],
}


def make_demo_data(raw_dir: Path) -> Path:
    """Write a small synthetic CSV (18 pairs + 1 deliberate duplicate) for dry runs."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "demo_pairs.csv"
    rows = []
    for cwe, variants in _DEMO_VARIANTS.items():
        vuln_tmpl, fixed_tmpl = _DEMO_TEMPLATES[cwe]
        for variant in variants:
            rows.append((vuln_tmpl.format(**variant), fixed_tmpl.format(**variant), cwe))
    rows.append(rows[0])  # deliberate duplicate: exercises the dedup step
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["vulnerable_code", "fixed_code", "cwe"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} demo pairs to {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def prepare(raw_dir: Path, out_dir: Path, config: PrepConfig, parser_name: str = "auto",
            replay_file: Optional[Path] = None) -> dict:
    """Full pipeline: parse -> clean -> split -> negatives -> replay -> write.

    Negatives are sampled per split AFTER splitting, so a negative always
    lives in the same split as the pair it came from (no leakage). Replay
    data is mixed into TRAIN only - val/test must stay pure task data so the
    metric is untouched.
    """
    if parser_name == "auto":
        pairs = [pair for parse in PARSERS.values() for pair in parse(raw_dir)]
    else:
        pairs = PARSERS[parser_name](raw_dir)
    if not pairs:
        raise SystemExit(
            f"No input pairs found in {raw_dir}. Drop a CSV (vulnerable_code,fixed_code,cwe) "
            "or a Juliet tree there, or run with --make-demo-data first."
        )
    print(f"parsed {len(pairs)} raw pairs from {raw_dir}")
    kept, drops = clean_pairs(pairs, config)
    if not kept:
        raise SystemExit(f"All {len(pairs)} pairs were dropped during cleaning: {drops}")
    splits = split_pairs(kept, config)

    rng = random.Random(config.seed + 1)
    negatives_added = 0
    if config.negatives_frac > 0:
        for name, split_members in splits.items():
            if not split_members:
                continue
            count = min(len(split_members),
                        max(1, round(len(split_members) * config.negatives_frac)))
            split_members.extend(make_negative(p) for p in rng.sample(split_members, count))
            rng.shuffle(split_members)
            negatives_added += count

    records_by_split = {name: [to_record(p) for p in members] for name, members in splits.items()}

    replay_added = 0
    if replay_file is not None:
        replay = load_replay_records(replay_file)
        rng.shuffle(replay)
        cap = int(len(records_by_split["train"]) * config.replay_frac)
        chosen = replay[:cap]
        if len(replay) > cap:
            print(f"replay file has {len(replay)} records; mixing in {cap} "
                  f"(replay_frac={config.replay_frac})")
        records_by_split["train"].extend(chosen)
        rng.shuffle(records_by_split["train"])
        replay_added = len(chosen)

    stats = write_splits(records_by_split, out_dir)
    stats["drops"] = drops
    stats["negatives_added"] = negatives_added
    stats["replay_added"] = replay_added
    print_stats(stats, drops)
    print(f"\nwrote splits to {out_dir}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path(__file__).resolve().parent
    parser.add_argument("--raw-dir", type=Path, default=base / "raw")
    parser.add_argument("--out-dir", type=Path, default=base / "processed")
    parser.add_argument("--parser", choices=["auto", *PARSERS], default="auto")
    parser.add_argument("--seed", type=int, default=None, help="override PREP_SEED")
    parser.add_argument("--max-chars", type=int, default=None, help="override PREP_MAX_CHARS")
    parser.add_argument("--no-compile-filter", action="store_true",
                        help="skip the javac check on fixed_code")
    parser.add_argument("--semgrep-filter", action="store_true",
                        help="drop pairs whose labels disagree with the eval's semgrep "
                             "rules (needs semgrep; run on the pod before training)")
    parser.add_argument("--negatives-frac", type=float, default=None,
                        help="override PREP_NEGATIVES_FRAC (0 disables negatives)")
    parser.add_argument("--replay-file", type=Path, default=None,
                        help="JSONL of general {system,input,output} records to mix "
                             "into TRAIN (anti-forgetting replay, capped by PREP_REPLAY_FRAC)")
    parser.add_argument("--make-demo-data", action="store_true",
                        help="write a tiny synthetic CSV into --raw-dir and exit")
    args = parser.parse_args()

    if args.make_demo_data:
        make_demo_data(args.raw_dir)
        return

    config = PrepConfig()
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})
    if args.max_chars is not None:
        config = config.model_copy(update={"max_chars": args.max_chars})
    if args.no_compile_filter:
        config = config.model_copy(update={"compile_filter": False})
    if args.semgrep_filter:
        config = config.model_copy(update={"semgrep_filter": True})
    if args.negatives_frac is not None:
        config = config.model_copy(update={"negatives_frac": args.negatives_frac})
    prepare(args.raw_dir, args.out_dir, config, args.parser, args.replay_file)


if __name__ == "__main__":
    main()
