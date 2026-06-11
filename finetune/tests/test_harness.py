"""Tests for eval/harness.py. All CPU-only; tool-dependent tests skip cleanly."""

from __future__ import annotations

import shutil

import pytest

from finetune.eval import harness
from finetune.eval.harness import (
    HarnessConfig,
    extract_java_code,
    normalize_cwe,
    score_patch,
)

HAVE_JAVAC = shutil.which("javac") is not None

GOOD_CLASS = """\
public class Hello {
    public static String greet(String name) {
        return "hello " + name;
    }
}
"""

BROKEN_CLASS = """\
public class Hello {
    public static String greet(String name) {
        return "hello " + name   // missing semicolon and brace
}
"""


# ---------------------------------------------------------------- normalize_cwe

@pytest.mark.parametrize("raw", ["CWE-89", "cwe-89", "cwe89", "89", " CWE-89 "])
def test_normalize_cwe_variants(raw: str) -> None:
    assert normalize_cwe(raw) == "CWE-89"


@pytest.mark.parametrize("raw", ["CWE-79", "1234", "not-a-cwe"])
def test_normalize_cwe_rejects_out_of_scope(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_cwe(raw)


# ----------------------------------------------------------- extract_java_code

def test_extract_fenced_java_block() -> None:
    text = f"Sure, here is the fix:\n```java\n{GOOD_CLASS}```\nHope that helps."
    assert extract_java_code(text) == GOOD_CLASS.strip()


def test_extract_anonymous_fence_with_java_content() -> None:
    text = f"```\n{GOOD_CLASS}```"
    assert extract_java_code(text) == GOOD_CLASS.strip()


def test_extract_raw_java_without_fence() -> None:
    assert extract_java_code(GOOD_CLASS) == GOOD_CLASS.strip()


@pytest.mark.parametrize("text", ["", "   ", "no code here", "```\njust prose\n```"])
def test_extract_rejects_non_java(text: str) -> None:
    assert extract_java_code(text) is None


def test_extract_prefers_java_block_over_prose_fence() -> None:
    text = f"```\nsome shell output\n```\n```java\n{GOOD_CLASS}```"
    assert extract_java_code(text) == GOOD_CLASS.strip()


# ------------------------------------------------------------------ compile check

@pytest.mark.skipif(not HAVE_JAVAC, reason="javac not installed")
def test_compiles_good_class() -> None:
    result, detail = harness.check_compiles(GOOD_CLASS, HarnessConfig())
    assert result is True, detail


@pytest.mark.skipif(not HAVE_JAVAC, reason="javac not installed")
def test_compile_fails_broken_class() -> None:
    result, _ = harness.check_compiles(BROKEN_CLASS, HarnessConfig())
    assert result is False


def test_compile_none_when_javac_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    result, detail = harness.check_compiles(GOOD_CLASS, HarnessConfig())
    assert result is None
    assert "javac" in detail


def test_vuln_none_when_semgrep_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    result, detail = harness.check_vuln_fixed(GOOD_CLASS, "CWE-89", HarnessConfig())
    assert result is None
    assert "semgrep" in detail


# ------------------------------------------------------------------- score_patch

def test_score_patch_no_code_hard_fails() -> None:
    result = score_patch("I cannot help with that.", "CWE-89")
    assert result["format_ok"] is False
    assert result["compiles"] is False
    assert result["vuln_fixed"] is False
    assert result["score"] == 0.0


def test_score_renormalizes_when_tools_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    result = score_patch(GOOD_CLASS, "CWE-89")
    assert result["format_ok"] is True
    assert result["compiles"] is None
    assert result["vuln_fixed"] is None
    # only the format weight remains, and it passed
    assert result["score"] == 1.0


def test_score_weighting_all_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "check_compiles", lambda *_: (True, "ok"))
    monkeypatch.setattr(harness, "check_vuln_fixed", lambda *_: (False, "1 finding"))
    result = score_patch(GOOD_CLASS, "cwe-78")
    # format 0.2 + compile 0.4 pass, vuln 0.4 fails -> 0.6
    assert result["score"] == pytest.approx(0.6)
    assert result["details"]["cwe"] == "CWE-78"


def test_score_patch_rejects_unsupported_cwe() -> None:
    with pytest.raises(ValueError):
        score_patch(GOOD_CLASS, "CWE-79")


# ---------------------------------------------------------------------- selftest

def test_selftest_passes() -> None:
    assert harness.selftest() == 0
