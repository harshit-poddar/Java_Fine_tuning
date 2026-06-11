"""Tests for data/prepare_dataset.py. CPU-only; uses tmp dirs, never data/raw."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from finetune.data import prepare_dataset as prep
from finetune.data.prepare_dataset import (
    PrepConfig,
    RawPair,
    clean_pairs,
    make_demo_data,
    parse_csv_sources,
    parse_juliet_sources,
    prepare,
    split_pairs,
)

HAVE_JAVAC = shutil.which("javac") is not None


def _pair(vuln: str = "class A { int x; }", fixed: str = "class A { int y; }",
          cwe: str = "CWE-89", source_id: str = "t:0") -> RawPair:
    return RawPair(vulnerable_code=vuln, fixed_code=fixed, cwe=cwe, source_id=source_id)


def _no_compile_cfg(**kwargs) -> PrepConfig:
    return PrepConfig(compile_filter=False, **kwargs)


# ------------------------------------------------------------------- csv adapter

def test_csv_adapter_reads_pairs(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["vulnerable_code", "fixed_code", "cwe"])
        writer.writerow(["class V {}", "class F {}", "CWE-89"])
    pairs = parse_csv_sources(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].cwe == "CWE-89"
    assert pairs[0].source_id == "x.csv:0"


def test_csv_adapter_skips_malformed_file(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    assert parse_csv_sources(tmp_path) == []
    assert "missing columns" in capsys.readouterr().err


# ---------------------------------------------------------------- juliet adapter

JULIET_FILE = """\
package testcases.CWE89_SQL_Injection.s01;
import testcasesupport.*;
import java.sql.*;

public class CWE89_SQL_Injection__database_01 extends AbstractTestCase {
    public void bad() throws Throwable {
        String data = "user";
        IO.writeLine(data);
    }

    public void goodG2B() throws Throwable {
        String data = "constant";
        IO.writeLine(data);
    }
}
"""


def test_juliet_adapter_extracts_bad_good_pair(tmp_path: Path) -> None:
    (tmp_path / "CWE89_SQL_Injection__database_01.java").write_text(JULIET_FILE, encoding="utf-8")
    pairs = parse_juliet_sources(tmp_path)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.cwe == "CWE-89"
    assert "void bad()" in pair.vulnerable_code
    assert "goodG2B" not in pair.fixed_code  # renamed so both sides look alike
    assert "System.out.println" in pair.vulnerable_code  # IO.writeLine stripped
    assert "import java.sql.*;" in pair.vulnerable_code
    assert "testcasesupport" not in pair.vulnerable_code
    assert "AbstractTestCase" not in pair.vulnerable_code


def test_juliet_adapter_maps_cwe23_to_cwe22(tmp_path: Path) -> None:
    content = JULIET_FILE.replace("CWE89_SQL_Injection", "CWE23_Relative_Path_Traversal")
    (tmp_path / "CWE23_Relative_Path_Traversal__file_01.java").write_text(content, encoding="utf-8")
    pairs = parse_juliet_sources(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].cwe == "CWE-22"


def test_juliet_adapter_ignores_out_of_scope_cwe(tmp_path: Path) -> None:
    (tmp_path / "CWE79_XSS__basic_01.java").write_text(
        JULIET_FILE.replace("CWE89_SQL_Injection", "CWE79_XSS"), encoding="utf-8"
    )
    assert parse_juliet_sources(tmp_path) == []


# -------------------------------------------------------------------- cleaning

def test_clean_filters_out_of_scope_cwe() -> None:
    kept, drops = clean_pairs([_pair(cwe="CWE-79")], _no_compile_cfg())
    assert kept == []
    assert drops["cwe_out_of_scope"] == 1


def test_clean_normalizes_cwe_labels() -> None:
    kept, _ = clean_pairs([_pair(cwe="89")], _no_compile_cfg())
    assert kept[0].cwe == "CWE-89"


def test_clean_dedups_whitespace_variants() -> None:
    a = _pair(vuln="class A {\n int x; }")
    b = _pair(vuln="class  A  { int x; }", source_id="t:1")
    kept, drops = clean_pairs([a, b], _no_compile_cfg())
    assert len(kept) == 1
    assert drops["duplicate"] == 1


def test_clean_drops_too_long() -> None:
    kept, drops = clean_pairs([_pair(vuln="class A { String s = \"" + "x" * 9000 + "\"; }")],
                              _no_compile_cfg())
    assert kept == []
    assert drops["too_long"] == 1


@pytest.mark.skipif(not HAVE_JAVAC, reason="javac not installed")
def test_clean_drops_non_compiling_fixed_code() -> None:
    good = _pair(fixed="public class F { int y; }")
    broken = _pair(vuln="class B { int z; }", fixed="public class F { broken", source_id="t:1")
    kept, drops = clean_pairs([good, broken], PrepConfig())
    assert len(kept) == 1
    assert drops["fixed_fails_javac"] == 1


# -------------------------------------------------------------------- splitting

def _distinct_pair(i: int) -> RawPair:
    body = " ".join(f"field_{i}_{j} method_{i}_{j} value_{i}_{j}" for j in range(12))
    return _pair(vuln=f"class C{i} {{ /* {body} */ }}", source_id=f"t:{i}")


def test_split_proportions_and_coverage() -> None:
    pairs = [_distinct_pair(i) for i in range(50)]
    splits = split_pairs(pairs, _no_compile_cfg())
    total = sum(len(s) for s in splits.values())
    assert total == 50
    assert len(splits["train"]) >= 30
    assert len(splits["val"]) >= 1
    assert len(splits["test"]) >= 1
    ids = [p.source_id for s in splits.values() for p in s]
    assert sorted(ids) == sorted(p.source_id for p in pairs)  # nothing lost or duplicated


def test_split_keeps_near_duplicates_together() -> None:
    base = "class Repo { void find(String v) { query(\"SELECT a,b,c FROM t WHERE k = \" + v); } }"
    near_a = _pair(vuln=base, source_id="dup:a")
    near_b = _pair(vuln=base.replace("Repo", "Repo2"), source_id="dup:b")
    others = [_distinct_pair(i) for i in range(20)]
    splits = split_pairs([near_a, near_b, *others], _no_compile_cfg())
    located = {p.source_id: name for name, ps in splits.items() for p in ps}
    assert located["dup:a"] == located["dup:b"]


def test_split_deterministic_for_seed() -> None:
    pairs = [_distinct_pair(i) for i in range(30)]
    one = split_pairs(pairs, _no_compile_cfg())
    two = split_pairs(pairs, _no_compile_cfg())
    assert {k: [p.source_id for p in v] for k, v in one.items()} == \
           {k: [p.source_id for p in v] for k, v in two.items()}


# ------------------------------------------------------------------ end to end

def test_end_to_end_with_demo_data(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    make_demo_data(raw)
    config = PrepConfig(compile_filter=HAVE_JAVAC)
    stats = prepare(raw, out, config)

    for name in ("train", "val", "test"):
        path = out / f"{name}.jsonl"
        assert path.exists()
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert set(record) == {"system", "input", "output", "cwe", "vulnerable_code", "source_id"}
            assert record["cwe"] in prep.TARGET_CWES
            assert record["output"].startswith("```java\n")
    assert stats["drops"]["duplicate"] == 1  # the deliberate duplicate was caught
    assert (out / "stats.json").exists()
    total = sum(row["total"] for row in stats["splits"].values())
    assert total == 18
