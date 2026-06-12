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
    synthesize_fixed_code,
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


JULIET_BOTH_VARIANTS = """\
package testcases.CWE89_SQL_Injection.s01;
import testcasesupport.*;

public class CWE89_SQL_Injection__database_02 extends AbstractTestCase {
    public void bad() throws Throwable {
        String data = getUserInput();
        runSink(data);
    }

    public void goodG2B() throws Throwable {
        String data = "safe-constant";
        runSink(data);
    }

    public void goodB2G() throws Throwable {
        String data = getUserInput();
        runSafeSink(data);
    }
}
"""


def test_juliet_adapter_prefers_b2g_fix_variant(tmp_path: Path) -> None:
    (tmp_path / "CWE89_SQL_Injection__database_02.java").write_text(
        JULIET_BOTH_VARIANTS, encoding="utf-8"
    )
    pairs = parse_juliet_sources(tmp_path)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.source_id.endswith(":B2G")
    assert "runSafeSink" in pair.fixed_code        # the B2G body was chosen
    assert "safe-constant" not in pair.fixed_code  # not the G2B constant swap
    assert "goodB2G" not in pair.fixed_code        # renamed to bad


def test_juliet_adapter_falls_back_to_g2b(tmp_path: Path) -> None:
    (tmp_path / "CWE89_SQL_Injection__database_01.java").write_text(JULIET_FILE, encoding="utf-8")
    pairs = parse_juliet_sources(tmp_path)
    assert pairs[0].source_id.endswith(":G2B")


JULIET_CWE23_FILE = """\
package testcases.CWE23_Relative_Path_Traversal.s01;
import testcasesupport.*;
import java.io.*;

public class CWE23_Relative_Path_Traversal__file_01 extends AbstractTestCase {
    public void bad() throws Throwable {
        String root = "C:\\\\uploads\\\\";
        String data = "user.txt";
        File file = new File(root + data);
        FileInputStream stream = new FileInputStream(file);
        stream.close();
    }

    public void goodG2B() throws Throwable {
        String root = "C:\\\\uploads\\\\";
        String data = "safe.txt";
        File file = new File(root + data);
        FileInputStream stream = new FileInputStream(file);
        stream.close();
    }
}
"""


def test_juliet_adapter_maps_cwe23_to_cwe22_and_synthesizes(tmp_path: Path) -> None:
    (tmp_path / "CWE23_Relative_Path_Traversal__file_01.java").write_text(
        JULIET_CWE23_FILE, encoding="utf-8"
    )
    pairs = parse_juliet_sources(tmp_path)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.cwe == "CWE-22"
    assert pair.source_id.endswith(":synth")
    # the G2B constant swap must NOT be the fix; the synthesized guard must be
    assert "safe.txt" not in pair.fixed_code
    assert "new File(allowedDir, data)" in pair.fixed_code
    assert ".normalize()" in pair.fixed_code
    assert "new File(root + data)" not in pair.fixed_code
    # vulnerable side keeps the original sink
    assert "new File(root + data)" in pair.vulnerable_code


JULIET_CWE78_FILE = """\
package testcases.CWE78_OS_Command_Injection.s01;
import testcasesupport.*;
import java.io.*;

public class CWE78_OS_Command_Injection__env_01 extends AbstractTestCase {
    public void bad() throws Throwable {
        String osCommand = "/bin/ls ";
        String data = System.getenv("ADD");
        Process process = Runtime.getRuntime().exec(osCommand + data);
        process.waitFor();
    }

    public void goodG2B() throws Throwable {
        String osCommand = "/bin/ls ";
        String data = "fixed";
        Process process = Runtime.getRuntime().exec(osCommand + data);
        process.waitFor();
    }
}
"""


def test_juliet_adapter_synthesizes_cwe78_fix(tmp_path: Path) -> None:
    (tmp_path / "CWE78_OS_Command_Injection__env_01.java").write_text(
        JULIET_CWE78_FILE, encoding="utf-8"
    )
    pairs = parse_juliet_sources(tmp_path)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.cwe == "CWE-78"
    assert pair.source_id.endswith(":synth")
    assert ".exec(" not in pair.fixed_code          # vulnerable sink is gone
    assert "new ProcessBuilder(commandTokens)" in pair.fixed_code
    assert "data.matches(" in pair.fixed_code        # allow-list validation
    # the process variable keeps its name so downstream code still compiles
    assert "Process process = new ProcessBuilder" in pair.fixed_code
    assert "process.waitFor();" in pair.fixed_code
    assert ".exec(osCommand + data)" in pair.vulnerable_code


def test_synthesize_returns_none_for_unknown_sink() -> None:
    assert synthesize_fixed_code("CWE-78", "public class X { void bad() {} }") is None
    assert synthesize_fixed_code("CWE-22", "public class X { void bad() {} }") is None
    assert synthesize_fixed_code("CWE-89", "anything") is None  # 89 never synthesizes


def test_synthesize_cwe22_direct_file_uses_fixed_base() -> None:
    code = 'public class X {\n    void bad() throws Exception {\n        String data = "x";\n        File file = new File(data);\n    }\n}'
    fixed = synthesize_fixed_code("CWE-22", code)
    assert fixed is not None
    assert 'new File(System.getProperty("user.dir"))' in fixed
    assert "new File(allowedDir, data)" in fixed
    assert "new File(data)" not in fixed


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


def test_semgrep_filter_skips_gracefully_without_semgrep(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda _: None)
    kept, drops = clean_pairs([_pair()], PrepConfig(compile_filter=False, semgrep_filter=True))
    assert len(kept) == 1  # nothing dropped, filter skipped
    assert "semgrep filter SKIPPED" in capsys.readouterr().err


def test_semgrep_filter_drops_mislabeled_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    # fake semgrep verdicts: "SAFE" marker means no finding
    def fake_check(code: str, cwe: str, cfg) -> tuple[bool, str]:
        return ("SAFE" in code, "fake")

    monkeypatch.setattr(prep, "check_vuln_fixed", fake_check)
    good = _pair(vuln="class A { int VULN; }", fixed="class A { int SAFE; }")
    g2b_like = _pair(vuln="class B { int VULN; }", fixed="class B { int VULN2; }",
                     source_id="t:1")  # "fixed" still triggers the rule
    not_vuln = _pair(vuln="class C { int SAFE; }", fixed="class C { int SAFE2; }",
                     source_id="t:2")  # "vulnerable" side never triggers
    kept, drops = clean_pairs([good, g2b_like, not_vuln],
                              PrepConfig(compile_filter=False, semgrep_filter=True))
    assert [p.source_id for p in kept] == ["t:0"]
    assert drops["fixed_still_flagged"] == 1
    assert drops["vuln_not_flagged"] == 1


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
            assert set(record) == {"system", "input", "output", "cwe",
                                   "vulnerable_code", "source_id", "kind"}
            assert record["cwe"] in prep.TARGET_CWES
            assert record["output"].startswith("```java\n")
            assert record["kind"] in ("fix", "no_change")
    assert stats["drops"]["duplicate"] == 1  # the deliberate duplicate was caught
    assert (out / "stats.json").exists()
    total = sum(row["total"] for row in stats["splits"].values())
    assert total == 18 + stats["negatives_added"]
    assert stats["negatives_added"] >= 3  # at least one per non-empty split


# --------------------------------------------------------- negatives & replay

def test_make_negative_uses_fixed_code_both_sides() -> None:
    pair = _pair(vuln="class V { int bad; }", fixed="class V { int good; }")
    negative = prep.make_negative(pair)
    assert negative.vulnerable_code == pair.fixed_code
    assert negative.fixed_code == pair.fixed_code
    assert negative.kind == "no_change"
    assert negative.source_id.endswith("#neg")


def test_negative_record_expects_unchanged_output() -> None:
    pair = _pair(fixed="class F { int y; }")
    record = prep.to_record(prep.make_negative(pair))
    assert "class F { int y; }" in record["input"]
    assert record["output"] == "```java\nclass F { int y; }\n```"


def test_negatives_stay_in_source_split(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    make_demo_data(raw)
    prepare(raw, out, PrepConfig(compile_filter=False))
    for name in ("train", "val", "test"):
        records = [json.loads(l) for l in (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        in_split = {r["source_id"] for r in records}
        for record in records:
            if record["kind"] == "no_change":
                assert record["source_id"][: -len("#neg")] in in_split


def test_replay_mixes_into_train_only_and_is_capped(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    make_demo_data(raw)
    replay_path = tmp_path / "general.jsonl"
    replay_records = [
        {"system": "You are helpful.", "input": f"Question {i}?", "output": f"Answer {i}."}
        for i in range(100)
    ]
    replay_path.write_text(
        "\n".join(json.dumps(r) for r in replay_records) + "\n", encoding="utf-8"
    )
    config = PrepConfig(compile_filter=False, replay_frac=0.2)
    stats = prepare(raw, out, config, replay_file=replay_path)

    assert 0 < stats["replay_added"] <= 100
    train = [json.loads(l) for l in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    task_train = [r for r in train if r["kind"] != "replay"]
    replay_train = [r for r in train if r["kind"] == "replay"]
    assert len(replay_train) == int(len(task_train) * 0.2)  # capped by replay_frac
    for name in ("val", "test"):
        records = [json.loads(l) for l in (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(r["kind"] != "replay" for r in records)


def test_replay_file_rejects_bad_records(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"system": "s", "input": "i"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="output"):
        prep.load_replay_records(bad)
