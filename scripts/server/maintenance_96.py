from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "96"

SUCCESS_INTERMEDIATE_DIRS = (
    "dayahead/prediction",
    "dayahead/weight",
    "dayahead/fuse",
    "realtime/prediction",
    "realtime/weight",
    "realtime/fuse",
)

PERSISTENT_ASSETS = (
    "ledger",
    "cache/classifier",
    "runs/*/final",
    "runs/*/run_manifest.json",
)


def _age_days(path: Path, now_ts: float) -> float:
    return max(0.0, (now_ts - path.stat().st_mtime) / 86400.0)


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += int(item.stat().st_size)
        except OSError:
            continue
    return total


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _decision_snapshot_complete(manifest: dict[str, Any]) -> bool:
    """Return True only when both final task decisions are durably embedded."""
    tasks = (manifest.get("decision_snapshot") or {}).get("tasks") or {}
    for task in ("dayahead", "realtime"):
        item = tasks.get(task) or {}
        if not item.get("weights") or not item.get("model_quality_gate"):
            return False
    return True


def build_retention_plan(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    now: datetime | None = None,
    normal_days: int = 30,
    failed_days: int = 90,
    runtime_hours: int = 24,
) -> dict[str, Any]:
    """Build a non-destructive retention plan for the 96-point production tree."""
    now = now or datetime.now(timezone.utc)
    now_ts = now.timestamp()
    candidates: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    runs_root = output_root / "runs"
    if runs_root.exists():
        for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            if run_dir.name.startswith("range_"):
                range_manifest = {}
                for name in ("prediction_range_manifest.json", "range_manifest.json"):
                    path = run_dir / name
                    if not path.exists():
                        continue
                    try:
                        range_manifest = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        range_manifest = {}
                    break
                status = str(range_manifest.get("status") or "").upper()
                success = status in {"COMPLETE", "COMPLETE_WITH_WARNINGS", "NORMAL"}
                logs_dir = run_dir / "logs"
                if logs_dir.exists():
                    age = _age_days(logs_dir, now_ts)
                    threshold = normal_days if success else failed_days
                    if age >= threshold:
                        candidates.append({
                            "path": str(logs_dir),
                            "kind": "range_normal_log" if success else "range_failed_diagnostic",
                            "age_days": round(age, 2),
                            "size_bytes": _size_bytes(logs_dir),
                            "retention_days": threshold,
                        })
                continue
            manifest = _read_manifest(run_dir)
            status = str(
                manifest.get("delivery_status")
                or manifest.get("status")
                or ""
            ).upper()
            success = status in {
                "NORMAL",
                "COMPLETE",
                "COMPLETE_WITH_WARNINGS",
                "DEGRADED_DELIVERED",
            }

            logs_dir = run_dir / "logs"
            if logs_dir.exists():
                age = _age_days(logs_dir, now_ts)
                threshold = normal_days if success else failed_days
                if age >= threshold:
                    candidates.append({
                        "path": str(logs_dir),
                        "kind": "normal_log" if success else "failed_diagnostic",
                        "age_days": round(age, 2),
                        "size_bytes": _size_bytes(logs_dir),
                        "retention_days": threshold,
                    })

            if success:
                snapshot_ok = _decision_snapshot_complete(manifest)
                for rel in SUCCESS_INTERMEDIATE_DIRS:
                    path = run_dir / rel
                    if not path.exists():
                        continue
                    age = _age_days(path, now_ts)
                    if age < normal_days:
                        continue
                    item = {
                        "path": str(path),
                        "kind": "successful_run_intermediate",
                        "age_days": round(age, 2),
                        "size_bytes": _size_bytes(path),
                        "retention_days": normal_days,
                    }
                    if snapshot_ok:
                        candidates.append(item)
                    else:
                        item["kind"] = "blocked_missing_decision_snapshot"
                        item["reason"] = (
                            "run_manifest.decision_snapshot must contain non-empty "
                            "dayahead/realtime weights and model_quality_gate before "
                            "successful intermediates become retention-eligible"
                        )
                        blocked.append(item)

    runtime_root = output_root / "runtime"
    if runtime_root.exists():
        threshold_days = float(runtime_hours) / 24.0
        for path in sorted(runtime_root.iterdir()):
            try:
                age = _age_days(path, now_ts)
            except OSError:
                continue
            if age >= threshold_days:
                candidates.append({
                    "path": str(path),
                    "kind": "stale_runtime_scratch",
                    "age_days": round(age, 2),
                    "size_bytes": _size_bytes(path),
                    "retention_days": round(threshold_days, 4),
                })

    return {
        "status": "DRY_RUN_ONLY",
        "generated_at": now.isoformat(),
        "output_root": str(output_root),
        "policy": {
            "normal_logs_days": int(normal_days),
            "failed_diagnostics_days": int(failed_days),
            "successful_intermediates_days": int(normal_days),
            "runtime_stale_hours": int(runtime_hours),
            "persistent_assets": list(PERSISTENT_ASSETS),
            "experiments": "manual_research_archive_only",
            "destructive_cleanup_enabled": False,
        },
        "candidate_count": len(candidates),
        "candidate_bytes": int(sum(int(item["size_bytes"]) for item in candidates)),
        "blocked_count": len(blocked),
        "blocked_bytes": int(sum(int(item["size_bytes"]) for item in blocked)),
        "candidates": candidates,
        "blocked": blocked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan bounded 96-point output retention without deleting files."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--normal-days", type=int, default=30)
    parser.add_argument("--failed-days", type=int, default=90)
    parser.add_argument("--runtime-hours", type=int, default=24)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Reserved for post-acceptance cleanup; currently always rejected.",
    )
    args = parser.parse_args(argv)

    if args.apply:
        raise SystemExit(
            "DESTRUCTIVE_CLEANUP_DISABLED: server acceptance is required before --apply can exist."
        )

    plan = build_retention_plan(
        args.output_root,
        normal_days=args.normal_days,
        failed_days=args.failed_days,
        runtime_hours=args.runtime_hours,
    )
    payload = json.dumps(plan, ensure_ascii=False, indent=2)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.report.with_suffix(args.report.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(args.report)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
