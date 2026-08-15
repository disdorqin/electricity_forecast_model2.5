#!/usr/bin/env python
"""可复现性检查工具。

用相同种子运行两次 ledger_smoke，比较两次输出的
all_model_predictions_long.csv 是否一致。

用法:
    python scripts/check_reproducibility.py 2026-02-24
    python scripts/check_reproducibility.py 2026-02-24 --seed 42 --deterministic
"""
from __future__ import annotations

import argparse
import hashlib
import json as json_lib
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _file_sha256(path: Path) -> str:
    """Return SHA-256 hex digest of file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare_csv_files(path_a: Path, path_b: Path, label: str) -> list[str]:
    """Compare two CSV files and return list of differences."""
    issues: list[str] = []
    if not path_a.exists() and not path_b.exists():
        return []
    if not path_a.exists():
        issues.append(f"[{label}] Missing in run A")
        return issues
    if not path_b.exists():
        issues.append(f"[{label}] Missing in run B")
        return issues

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    if len(df_a) != len(df_b):
        issues.append(f"[{label}] Row count: A={len(df_a)} vs B={len(df_b)}")
        return issues

    # Sort by common key columns before comparing
    key_cols = [
        c for c in ["task", "model_name", "forecast_date", "target_day", "business_day", "hour_business"]
        if c in df_a.columns and c in df_b.columns
    ]
    if key_cols:
        df_a = df_a.sort_values(key_cols).reset_index(drop=True)
        df_b = df_b.sort_values(key_cols).reset_index(drop=True)
        if not df_a[key_cols].equals(df_b[key_cols]):
            issues.append(f"[{label}] key columns differ")

    # Check y_pred column with tolerance
    for col in ["y_pred"]:
        if col in df_a.columns and col in df_b.columns:
            a_vals = df_a[col].to_numpy(float)
            b_vals = df_b[col].to_numpy(float)
            if not np.allclose(a_vals, b_vals, atol=1e-6, rtol=1e-6, equal_nan=True):
                mismatches = (~np.isclose(a_vals, b_vals, atol=1e-6, rtol=1e-6, equal_nan=True)).sum()
                max_diff = np.max(np.abs(a_vals - b_vals))
                issues.append(
                    f"[{label}] {col}: {mismatches}/{len(df_a)} rows differ, "
                    f"max_abs_diff={max_diff:.6f}"
                )

    # Compare hashes only when no value differences found
    if not issues:
        sha_a = _file_sha256(path_a)
        sha_b = _file_sha256(path_b)
        if sha_a != sha_b:
            issues.append(
                f"[{label}] SHA256 mismatch (content differs in non-y_pred columns)"
            )

    return issues


def check_reproducibility(
    date: str,
    seed: int = 42,
    deterministic: bool = False,
    keep_tmp: bool = False,
    epf_v1_root: str | None = None,
    allow_v2_fallback: bool = False,
) -> bool:
    """Run ledger_smoke twice and compare outputs."""
    print(f"=== Reproducibility Check: date={date}, seed={seed}, deterministic={deterministic}, epf_v1_root={epf_v1_root} ===\n")

    tmp_a = Path(tempfile.mkdtemp(prefix="repro_a_"))
    tmp_b = Path(tempfile.mkdtemp(prefix="repro_b_"))

    project_root = Path(__file__).resolve().parent.parent
    results_root = project_root / "outputs" / "repro_check"
    results_root.mkdir(parents=True, exist_ok=True)

    # Clean up any existing output for the target date
    for tmp in [tmp_a, tmp_b]:
        ledger_dir = tmp / "ledger"
        runs_dir = tmp / "runs"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)

    def _run(test_dir: Path, tag: str) -> tuple[bool, Path]:
        """Run ledger_smoke once."""
        cmd = [
            sys.executable, "main.py",
            "--pipeline", "ledger_smoke",
            "--ledger-root", str(test_dir / "ledger"),
            "--runs-root", str(test_dir / "runs"),
            "--seed", str(seed),
            "--date", date,
        ]
        if deterministic:
            cmd.append("--deterministic")
        if epf_v1_root:
            cmd.extend(["--epf-v1-root", str(epf_v1_root)])
        if allow_v2_fallback:
            cmd.append("--allow-v2-fallback")

        print(f">>> Run {tag}: {' '.join(cmd)}")
        t0 = time.time()
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        elapsed = time.time() - t0
        print(f"    Exit: {result.returncode}, Elapsed: {elapsed:.1f}s")
        if result.returncode != 0:
            print(f"    STDOUT: {result.stdout[-500:]}")
            print(f"    STDERR: {result.stderr[-500:]}")
        return result.returncode == 0, test_dir

    # Run A
    ok_a, _ = _run(tmp_a, "A")
    # Run B
    ok_b, _ = _run(tmp_b, "B")

    if not ok_a or not ok_b:
        print("\n[FAIL] One or both runs failed")
        if not ok_a:
            print("  Run A: FAILED")
        if not ok_b:
            print("  Run B: FAILED")
        return False

    # Collect all comparison files
    all_ok = True
    compared_files = 0

    # Compare model-level CSV files
    for task in ["dayahead", "realtime"]:
        pred_dir_a = tmp_a / "runs" / date / task / "prediction"
        pred_dir_b = tmp_b / "runs" / date / task / "prediction"

        if not pred_dir_a.exists() or not pred_dir_b.exists():
            print(f"\n  [FAIL] {task}: prediction directory missing")
            all_ok = False
            continue

        # Compare all_model_predictions_long.csv
        long_a = pred_dir_a / "all_model_predictions_long.csv"
        long_b = pred_dir_b / "all_model_predictions_long.csv"
        issues = _compare_csv_files(long_a, long_b, f"{task}/long")
        for issue in issues:
            print(f"  {issue}")
            all_ok = False
        if long_a.exists() and long_b.exists():
            compared_files += 1

        # Compare per-model files
        csv_files_a = sorted(pred_dir_a.glob("*_predictions.csv"))
        for f_a in csv_files_a:
            if f_a.name == "all_model_predictions_long.csv":
                continue
            f_b = pred_dir_b / f_a.name
            model_label = f"{task}/{f_a.stem.replace('_predictions', '')}"
            issues = _compare_csv_files(f_a, f_b, model_label)
            for issue in issues:
                print(f"  {issue}")
                all_ok = False
            if f_b.exists():
                compared_files += 1

    # Compare run manifests
    manifest_a = tmp_a / "runs" / date / "run_manifest.json"
    manifest_b = tmp_b / "runs" / date / "run_manifest.json"
    if manifest_a.exists() and manifest_b.exists():
        sha_a = _file_sha256(manifest_a)
        sha_b = _file_sha256(manifest_b)
        if sha_a != sha_b:
            # Manifest timestamps/paths/timing will differ — strip those fields
            def _strip_manifest_noise(d: dict) -> dict:
                """Remove timing, path, and other non-structural fields."""
                noise_keys = {"started_at", "completed_at", "elapsed_seconds",
                              "output_path", "csv_path", "parquet_path", "source_file"}
                out = {}
                for k, v in d.items():
                    if k in noise_keys:
                        continue
                    if isinstance(v, dict):
                        nested = _strip_manifest_noise(v)
                        if nested:
                            out[k] = nested
                    elif isinstance(v, list):
                        cleaned = [
                            _strip_manifest_noise(item) if isinstance(item, dict) else item
                            for item in v
                        ]
                        if cleaned:
                            out[k] = cleaned
                    else:
                        out[k] = v
                return out

            with open(manifest_a) as f:
                m_a = _strip_manifest_noise(json_lib.load(f))
            with open(manifest_b) as f:
                m_b = _strip_manifest_noise(json_lib.load(f))
            if m_a != m_b:
                print(f"  [WARN] run_manifest.json: structural differences (non-timestamp)")
                import difflib
                a_str = json_lib.dumps(m_a, sort_keys=True, indent=2, default=str)
                b_str = json_lib.dumps(m_b, sort_keys=True, indent=2, default=str)
                for line in difflib.unified_diff(a_str.splitlines(), b_str.splitlines(),
                                                  fromfile='A', tofile='B', lineterm='')[:30]:
                    print(f"    {line}")
                all_ok = False
            else:
                print(f"  run_manifest.json: structural match OK (timing/paths excluded)")

    print(f"\n  compared files: {compared_files}")

    if compared_files == 0:
        print("[FAIL] No prediction CSV files were compared.")
        all_ok = False

    print()
    if all_ok:
        print("=== Result: PASS (all outputs identical) ===")
    else:
        print("=== Result: FAIL (outputs differ between runs) ===")

    # Copy results for inspection
    shutil.copytree(str(tmp_a), str(results_root / f"{date}_A"), dirs_exist_ok=True)
    shutil.copytree(str(tmp_b), str(results_root / f"{date}_B"), dirs_exist_ok=True)
    print(f"  Results saved to: {results_root}/{date}_A/ and {results_root}/{date}_B/")

    if not keep_tmp:
        shutil.rmtree(str(tmp_a))
        shutil.rmtree(str(tmp_b))
        print(f"  Temp dirs cleaned up")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Check ledger_smoke reproducibility")
    parser.add_argument("date", help="Target date YYYY-MM-DD")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic mode")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temp output directories")
    parser.add_argument("--epf-v1-root", default=None, help="Path to EPF v1.0 repository root")
    parser.add_argument("--allow-v2-fallback", action="store_true", default=False, help="Allow fallback to 2.0 implementations")
    args = parser.parse_args()

    ok = check_reproducibility(
        args.date,
        seed=args.seed,
        deterministic=args.deterministic,
        keep_tmp=args.keep_tmp,
        epf_v1_root=args.epf_v1_root,
        allow_v2_fallback=args.allow_v2_fallback,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
