#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_golden_baseline_24.py — READ-ONLY verification of the 24-point golden baseline.

What it does
------------
1. Walks each ``<date>/`` directory under the golden-baseline root.
2. Reads the recorded hash index ``golden_baseline_hashes.json`` (generated at
   freeze time) and verifies that every recorded file still exists and that its
   SHA-256 matches the frozen value.
3. Validates the structure of ``submission_ready.csv`` when it is part of the
   baseline (24 rows, expected columns).
4. Reports which model legs were present / missing at freeze time.

What it does NOT do
-------------------
* It never trains a model, never runs the production pipeline, and never writes
  to ``outputs/runs``, ``outputs/ledger`` or any production output path.
* It only reads the frozen baseline directory.

Exit codes
----------
0  all checked baselines pass
1  at least one baseline has a missing / hash-mismatched file
2  baseline root or no date directory found

Usage
-----
    python scripts/check_golden_baseline_24.py [--root outputs/golden_baseline_24]
    python scripts/check_golden_baseline_24.py --root <path> [--strict-legs]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_submission(date_dir: Path, meta: dict) -> bool:
    """Validate submission_ready.csv structure if recorded."""
    rel = meta.get("path")
    if not rel:
        return True
    p = date_dir / rel
    if not p.exists():
        print(f"  FAIL: submission_ready missing ({rel})")
        return False
    # Lazy import csv only when needed (stdlib, always available).
    import csv

    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = sum(1 for _ in reader)

    ok = True
    exp_cols = meta.get("columns")
    if exp_cols:
        missing = [c for c in exp_cols if c not in header]
        if missing:
            print(f"  FAIL: submission_ready columns missing {missing} (have {header})")
            ok = False
    exp_rows = meta.get("rows")
    if exp_rows is not None and rows != exp_rows:
        print(f"  FAIL: submission_ready row count {rows} != expected {exp_rows}")
        ok = False
    if ok:
        print(f"  OK   submission_ready ({rows} rows, cols={header})")
    return ok


def check_date(date_dir: Path, strict_legs: bool) -> bool:
    print(f"\n### {date_dir.name} ###")
    idx = date_dir / "golden_baseline_hashes.json"
    if not idx.exists():
        print("  FAIL: golden_baseline_hashes.json not found")
        return False

    data = json.loads(idx.read_text(encoding="utf-8"))
    ok = True

    # Hash/index verification of every recorded file.
    files = data.get("files", {})
    for rel, meta in files.items():
        p = date_dir / rel
        if not p.exists():
            print(f"  FAIL: missing file {rel}")
            ok = False
            continue
        exp = meta.get("sha256")
        if exp:
            got = sha256_of(p)
            if got != exp:
                print(f"  FAIL: hash mismatch {rel}\n        got={got}\n        exp={exp}")
                ok = False
                continue
        rows = meta.get("rows")
        if rows is not None:
            print(f"  OK   {rel} (sha256 ok, {rows} rows)")
        else:
            print(f"  OK   {rel} (sha256 ok)")

    # Structural check for submission_ready.csv.
    sr = data.get("submission_ready")
    if sr:
        ok = _check_submission(date_dir, sr) and ok

    # Leg presence reporting.
    present = data.get("legs_present", [])
    missing = data.get("legs_missing", [])
    print(f"  legs present ({len(present)}): {', '.join(present) if present else 'NONE'}")
    if missing:
        tag = "FAIL" if strict_legs else "WARN"
        print(f"  {tag}: legs missing at freeze ({len(missing)}): {', '.join(missing)}")
        if strict_legs:
            ok = False

    delivery = data.get("delivery_status", "UNKNOWN")
    print(f"  delivery_status: {delivery} | exit_code: {data.get('exit_code')}")
    print(f"  {'PASS' if ok else 'FAIL'}: {date_dir.name}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only verification of the 24-point golden baseline.")
    ap.add_argument(
        "--root",
        default="outputs/golden_baseline_24",
        help="Golden baseline root directory (default: outputs/golden_baseline_24).",
    )
    ap.add_argument(
        "--strict-legs",
        action="store_true",
        help="Treat missing model legs as a hard failure instead of a warning.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"ERROR: baseline root not found: {root}")
        return 2

    dates = sorted([d for d in root.iterdir() if d.is_dir()])
    if not dates:
        print(f"ERROR: no baseline date directories under {root}")
        return 2

    overall = True
    for d in dates:
        overall = check_date(d, args.strict_legs) and overall

    print("\n=== SUMMARY ===")
    print("ALL BASELINES PASS" if overall else "BASELINE VERIFICATION FAILED")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
