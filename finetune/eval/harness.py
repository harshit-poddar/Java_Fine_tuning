"""Eval harness for FINETUNING_001.

Scores a generated Java patch on three axes:
  - format_ok:   the model output contains a parseable Java code block
  - compiles:    the extracted code compiles with `javac`
  - vuln_fixed:  Semgrep (with a bundled rule for the target CWE) reports no finding

Degrades gracefully when `javac` or `semgrep` are not installed: the
corresponding check returns None and a warning is printed once. Runs entirely
on CPU; no model is needed.

Usage:
    python harness.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

RULES_DIR = Path(__file__).resolve().parent / "rules"

#: Supported CWEs -> bundled semgrep rule file.
CWE_RULES: dict[str, str] = {
    "CWE-89": "cwe_89.yml",
    "CWE-22": "cwe_22.yml",
    "CWE-78": "cwe_78.yml",
}

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"WARNING: {message}", file=sys.stderr)


class HarnessConfig(BaseModel):
    """Scoring weights and tool settings. Every field is env-overridable."""

    javac_bin: str = Field(default_factory=lambda: os.getenv("HARNESS_JAVAC", "javac"))
    semgrep_bin: str = Field(default_factory=lambda: os.getenv("HARNESS_SEMGREP", "semgrep"))
    javac_timeout_s: int = Field(default_factory=lambda: int(os.getenv("HARNESS_JAVAC_TIMEOUT", "60")))
    semgrep_timeout_s: int = Field(default_factory=lambda: int(os.getenv("HARNESS_SEMGREP_TIMEOUT", "120")))
    weight_format: float = Field(default_factory=lambda: float(os.getenv("HARNESS_W_FORMAT", "0.2")))
    weight_compile: float = Field(default_factory=lambda: float(os.getenv("HARNESS_W_COMPILE", "0.4")))
    weight_vuln: float = Field(default_factory=lambda: float(os.getenv("HARNESS_W_VULN", "0.4")))


def normalize_cwe(cwe: str) -> str:
    """Normalize 'cwe-89' / '89' / 'CWE89' to 'CWE-89'. Raises on unsupported CWEs."""
    match = re.search(r"(\d+)", str(cwe))
    if not match:
        raise ValueError(f"Cannot parse CWE id from {cwe!r}")
    canonical = f"CWE-{match.group(1)}"
    if canonical not in CWE_RULES:
        raise ValueError(
            f"Unsupported CWE {canonical!r}. Supported: {sorted(CWE_RULES)} "
            "(project scope is fixed to these three)."
        )
    return canonical


_FENCE_RE = re.compile(r"```(?:[Jj]ava)?[ \t]*\n(.*?)```", re.DOTALL)
_TYPE_DECL_RE = re.compile(
    r"\b(?:class|interface|enum|record)\s+[A-Za-z_$][\w$]*"
)
_PUBLIC_TYPE_RE = re.compile(
    r"\bpublic\s+(?:final\s+|abstract\s+|strictfp\s+)*(?:class|interface|enum|record)\s+([A-Za-z_$][\w$]*)"
)
_ANY_TYPE_RE = re.compile(
    r"\b(?:class|interface|enum|record)\s+([A-Za-z_$][\w$]*)"
)


def extract_java_code(text: str) -> Optional[str]:
    """Pull the Java code out of a model response.

    Accepts a fenced ```java block (preferred), any fenced block whose content
    declares a Java type, or raw text that itself declares a Java type.
    Returns None when no plausible Java code block is found.
    """
    if not text or not text.strip():
        return None
    for block in _FENCE_RE.findall(text):
        if _TYPE_DECL_RE.search(block):
            return block.strip()
    if _TYPE_DECL_RE.search(text) and "```" not in text:
        return text.strip()
    return None


def _main_type_name(code: str) -> str:
    """Class name to use for the .java file (javac requires it to match)."""
    match = _PUBLIC_TYPE_RE.search(code) or _ANY_TYPE_RE.search(code)
    return match.group(1) if match else "Snippet"


def check_compiles(code: str, config: HarnessConfig) -> tuple[Optional[bool], str]:
    """Compile `code` with javac. Returns (result, detail); result is None if javac is missing."""
    javac = shutil.which(config.javac_bin)
    if javac is None:
        _warn_once(
            "javac",
            "javac not found - compile check skipped (install a JDK, e.g. "
            "winget install EclipseAdoptium.Temurin.17.JDK, or set HARNESS_JAVAC)",
        )
        return None, "javac not found"
    with tempfile.TemporaryDirectory(prefix="harness_javac_") as tmp:
        src = Path(tmp) / f"{_main_type_name(code)}.java"
        src.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [javac, "-encoding", "UTF-8", "-nowarn", "-d", tmp, str(src)],
                capture_output=True,
                text=True,
                timeout=config.javac_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, f"javac timed out after {config.javac_timeout_s}s"
        if proc.returncode == 0:
            return True, "compiled"
        return False, (proc.stderr or proc.stdout).strip()[:2000]


def check_vuln_fixed(code: str, cwe: str, config: HarnessConfig) -> tuple[Optional[bool], str]:
    """Run semgrep with the bundled rule for `cwe`. True = no finding (vuln gone).

    Returns (result, detail); result is None if semgrep is missing or errors.
    """
    semgrep = shutil.which(config.semgrep_bin)
    if semgrep is None:
        _warn_once(
            "semgrep",
            "semgrep not found - vuln check skipped (Linux/pod: pip install semgrep; "
            "no native Windows support, use WSL locally if needed)",
        )
        return None, "semgrep not found"
    rule_file = RULES_DIR / CWE_RULES[cwe]
    if not rule_file.exists():
        raise FileNotFoundError(f"Bundled semgrep rule missing: {rule_file}")
    with tempfile.TemporaryDirectory(prefix="harness_semgrep_") as tmp:
        src = Path(tmp) / f"{_main_type_name(code)}.java"
        src.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [semgrep, "scan", "--quiet", "--json", "--metrics=off",
                 "--config", str(rule_file), str(src)],
                capture_output=True,
                text=True,
                timeout=config.semgrep_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return None, f"semgrep timed out after {config.semgrep_timeout_s}s"
    try:
        results = json.loads(proc.stdout)["results"]
    except (json.JSONDecodeError, KeyError):
        return None, f"semgrep output unparseable (exit {proc.returncode}): {proc.stderr.strip()[:500]}"
    if results:
        findings = ", ".join(sorted({r.get("check_id", "?") for r in results}))
        return False, f"{len(results)} finding(s): {findings}"
    return True, "no findings"


def score_patch(java_code: str, cwe: str, config: Optional[HarnessConfig] = None) -> dict:
    """Score a generated patch for a single test item.

    Args:
        java_code: raw model output (may contain markdown fences and prose).
        cwe: target CWE, e.g. "CWE-89" (also accepts "89" / "cwe-89").
        config: optional HarnessConfig override.

    Returns:
        dict with keys: format_ok (bool), compiles (bool | None),
        vuln_fixed (bool | None), score (float in [0, 1]), details (dict).
        None means the corresponding tool is unavailable; such checks are
        excluded from the score (weights renormalized).
    """
    cfg = config or HarnessConfig()
    canonical_cwe = normalize_cwe(cwe)

    code = extract_java_code(java_code)
    format_ok = code is not None

    if format_ok:
        compiles, compile_detail = check_compiles(code, cfg)
        vuln_fixed, vuln_detail = check_vuln_fixed(code, canonical_cwe, cfg)
    else:
        # No code to check: hard fail rather than "tool unavailable".
        compiles, compile_detail = False, "no java code block found"
        vuln_fixed, vuln_detail = False, "no java code block found"

    weighted: list[tuple[float, bool]] = [(cfg.weight_format, format_ok)]
    if compiles is not None:
        weighted.append((cfg.weight_compile, compiles))
    if vuln_fixed is not None:
        weighted.append((cfg.weight_vuln, vuln_fixed))
    total_weight = sum(w for w, _ in weighted)
    score = sum(w for w, ok in weighted if ok) / total_weight if total_weight else 0.0

    return {
        "format_ok": format_ok,
        "compiles": compiles,
        "vuln_fixed": vuln_fixed,
        "score": round(score, 4),
        "details": {"cwe": canonical_cwe, "compile": compile_detail, "vuln": vuln_detail},
    }


# --------------------------------------------------------------------------
# Selftest: two hardcoded CWE-89 examples, no model needed.
# --------------------------------------------------------------------------

_VULNERABLE_EXAMPLE = """\
import java.sql.*;

public class UserDao {
    public ResultSet findUser(Connection conn, String userId) throws SQLException {
        Statement stmt = conn.createStatement();
        String query = "SELECT * FROM users WHERE id = '" + userId + "'";
        return stmt.executeQuery(query);
    }
}
"""

# Wrapped in a markdown fence on purpose: exercises code-block extraction.
_FIXED_EXAMPLE = """\
Here is the patched code:

```java
import java.sql.*;

public class UserDao {
    public ResultSet findUser(Connection conn, String userId) throws SQLException {
        PreparedStatement ps = conn.prepareStatement("SELECT * FROM users WHERE id = ?");
        ps.setString(1, userId);
        return ps.executeQuery();
    }
}
```
"""


def selftest() -> int:
    """Score one vulnerable and one fixed example; verify expected ordering."""
    print("== harness selftest (CWE-89) ==")
    failures: list[str] = []
    results = {}
    for name, payload in [("vulnerable", _VULNERABLE_EXAMPLE), ("fixed", _FIXED_EXAMPLE)]:
        result = score_patch(payload, "CWE-89")
        results[name] = result
        print(f"\n[{name}]")
        print(json.dumps(result, indent=2))

    if not results["fixed"]["format_ok"]:
        failures.append("fixed example: format_ok should be True")
    for name, expected in [("vulnerable", False), ("fixed", True)]:
        actual = results[name]["vuln_fixed"]
        if actual is not None and actual is not expected:
            failures.append(f"{name} example: vuln_fixed expected {expected}, got {actual}")
    for name in ("vulnerable", "fixed"):
        actual = results[name]["compiles"]
        if actual is not None and actual is not True:
            failures.append(f"{name} example: should compile, got {actual}")
    if results["fixed"]["score"] <= results["vulnerable"]["score"] and \
            results["fixed"]["vuln_fixed"] is not None:
        failures.append("fixed example should outscore vulnerable example")

    print()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("selftest OK (checks with None were skipped: tool not installed)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="run 2 hardcoded examples and exit")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    parser.print_help()


if __name__ == "__main__":
    main()
