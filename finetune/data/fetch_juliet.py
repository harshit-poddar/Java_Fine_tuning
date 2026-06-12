"""Download Juliet Java 1.3 and extract the target-CWE test files.

Pulls the suite zip and extracts only the Java files for our CWE families
(89 SQLi, 78 cmd injection, 23/36 path traversal -> CWE-22) into
data/raw/juliet/. Then run prepare_dataset.py.

Source: the find-sec-bugs GitHub mirror of Juliet Java 1.3 (full 112-CWE
suite). NIST SARD returns HTTP 403/500 to programmatic downloads, so we pull
the archive from codeload, which is reliable. Override with --url if needed.

    python fetch_juliet.py
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

JULIET_URL = (
    "https://codeload.github.com/find-sec-bugs/juliet-test-suite/"
    "zip/refs/heads/master"
)
TARGET_FILE_RE = re.compile(r"(CWE89_|CWE78_|CWE23_|CWE36_).*\.java$")


def fetch(url: str, out_dir: Path) -> int:
    """Download the suite and extract target-CWE .java files. Returns file count."""
    print(f"downloading {url} ...", flush=True)
    # urllib's default User-Agent gets 403'd by some CDNs; send a browser UA.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except OSError as error:
        raise SystemExit(
            f"Download failed ({error}). Clone the mirror manually:\n"
            "  git clone --depth 1 https://github.com/find-sec-bugs/juliet-test-suite\n"
            f"then copy its CWE89/78/23/36 *.java files into {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            if TARGET_FILE_RE.search(info.filename):
                (out_dir / Path(info.filename).name).write_bytes(archive.read(info))
                count += 1
    if count == 0:
        raise SystemExit("Zip downloaded but no target-CWE files found - layout changed?")
    print(f"extracted {count} java files to {out_dir}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=JULIET_URL)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "raw" / "juliet")
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.glob("*.java")):
        print(f"{args.out_dir} already has java files - skipping download", file=sys.stderr)
        return
    fetch(args.url, args.out_dir)


if __name__ == "__main__":
    main()
