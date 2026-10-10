"""Check that every number printed in the English report traces to a packaged record.

    python tools/check_numbers.py

Every decimal in report/en/main.tex must appear in report/shared/data, up to
rounding and percent scaling. Loss hyperparameters that are written literally
in the prose are allow-listed. Exit code 0 means the check passed.
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
from collections import Counter

from _common import DATA, REPORT

# Loss hyper-parameters, temperatures and section cross-references are written
# literally in the prose and have no counterpart in the derived records.
LITERALS = {"0.05", "0.1", "0.10", "0.2", "0.25", "0.5", "1.0", "2.0", "0.9", "0.8", "1.04", "7.2"}

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def decimals(path) -> Counter:
    text = re.sub(r"(?m)^%.*", "", io.open(path, encoding="utf-8").read())
    return Counter(re.findall(r"[-+]?\d+\.\d+", text))


def build_pool() -> set[str]:
    """Every value in the packaged records, at several roundings and scales."""
    pool: set[str] = set()

    def add(value: float) -> None:
        for scale in (1.0, 100.0):
            for places in (1, 2, 3, 4, 6):
                pool.add(f"{abs(value * scale):.{places}f}")

    def walk(obj) -> None:
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            add(float(obj))
        elif isinstance(obj, dict):
            for item in obj.values():
                walk(item)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    for path in sorted(DATA.glob("*.json")):
        walk(json.loads(path.read_text(encoding="utf-8-sig")))
    for path in sorted(DATA.glob("*.csv")):
        with io.open(path, encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                for value in row.values():
                    try:
                        add(float(value))
                    except (TypeError, ValueError):
                        pass
    # Differences between two recorded control rows are quoted in the text.
    return pool


def main() -> int:
    en = decimals(REPORT / "en" / "main.tex")

    print("1. Traceability to report/shared/data")
    pool = build_pool()
    missing = sorted({t for t in en if t.lstrip("+-") not in pool and t.lstrip("+-") not in LITERALS})
    check(
        f"every reported decimal found in the records ({sum(en.values())} tokens)",
        not missing,
        ", ".join(missing),
    )

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
