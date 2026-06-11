"""Structural checks for the bundled semgrep rules (full validation needs
semgrep itself: run `semgrep scan --validate --config finetune/eval/rules` on
the pod)."""

from __future__ import annotations

import yaml

from finetune.eval.harness import CWE_RULES, RULES_DIR


def test_every_supported_cwe_has_a_rule_file() -> None:
    for cwe, filename in CWE_RULES.items():
        path = RULES_DIR / filename
        assert path.exists(), f"missing rule file for {cwe}: {path}"


def test_rule_files_are_valid_semgrep_shaped_yaml() -> None:
    for filename in CWE_RULES.values():
        doc = yaml.safe_load((RULES_DIR / filename).read_text(encoding="utf-8"))
        assert isinstance(doc.get("rules"), list) and doc["rules"], filename
        for rule in doc["rules"]:
            assert rule.get("id"), f"{filename}: rule missing id"
            assert rule.get("languages") == ["java"], f"{filename}: {rule['id']} must target java"
            assert rule.get("severity") in {"ERROR", "WARNING"}, f"{filename}: {rule['id']}"
            assert rule.get("message"), f"{filename}: {rule['id']} missing message"
            assert any(k in rule for k in ("pattern", "pattern-either", "patterns")), (
                f"{filename}: {rule['id']} has no pattern clause"
            )


def test_rule_metadata_matches_filename_cwe() -> None:
    for cwe, filename in CWE_RULES.items():
        doc = yaml.safe_load((RULES_DIR / filename).read_text(encoding="utf-8"))
        for rule in doc["rules"]:
            assert rule.get("metadata", {}).get("cwe") == cwe, (
                f"{filename}: {rule['id']} metadata.cwe should be {cwe}"
            )
