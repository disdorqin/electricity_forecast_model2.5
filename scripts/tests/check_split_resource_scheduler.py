"""Contract test for the isolated CPU/GPU split-process scheduler."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.resource_scheduler import ResourceScheduler, ScheduleTask
from runtime.scheduler_test_helpers import fail_task, write_marker_task


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="efm3-scheduler-") as tmp:
        root = Path(tmp)
        cpu_marker = root / "cpu_markers.jsonl"
        gpu_marker = root / "gpu_markers.jsonl"
        cpu_a = root / "cpu_a.ok"
        cpu_b = root / "cpu_b.ok"
        cpu_c = root / "cpu_c.ok"
        cpu_failed_child = root / "cpu_failed_child.ok"
        gpu_a = root / "gpu_a.ok"
        gpu_b = root / "gpu_b.ok"
        log_path = root / "pipeline.log"
        tasks = [
            ScheduleTask(
                "lightgbm", "dayahead", "2026-01-01", write_marker_task,
                {"output_path": str(cpu_a), "marker_path": str(cpu_marker), "sleep_seconds": 0.20}, "cpu",
                node_id="lightgbm/dayahead",
            ),
            ScheduleTask(
                "timesfm", "realtime", "2026-01-01", write_marker_task,
                {"output_path": str(cpu_b), "marker_path": str(cpu_marker)}, "cpu",
                node_id="timesfm/realtime", depends_on=("lightgbm/dayahead",),
            ),
            ScheduleTask(
                "sgdfnet", "anchor_prepare", "2026-01-01", write_marker_task,
                {"output_path": str(cpu_c), "marker_path": str(cpu_marker), "sleep_seconds": 0.20}, "cpu",
                node_id="sgdfnet/anchor_prepare",
            ),
            ScheduleTask(
                "timemixer", "dayahead", "2026-01-01", write_marker_task,
                {"output_path": str(gpu_a), "marker_path": str(gpu_marker), "sleep_seconds": 0.05}, "gpu",
                node_id="timemixer/dayahead",
            ),
            ScheduleTask(
                "rt916", "realtime", "2026-01-01", write_marker_task,
                {"output_path": str(gpu_b), "marker_path": str(gpu_marker)}, "gpu",
                node_id="rt916/realtime", depends_on=("timemixer/dayahead",),
            ),
        ]
        started = time.perf_counter()
        results = ResourceScheduler(
            max_cpu_workers=2,
            max_gpu_workers=1,
            resource_mode="split_process",
            log_path=str(log_path),
        ).run(tasks)
        elapsed = time.perf_counter() - started

        assert len(results) == 5, results
        assert all(r.success for r in results), results
        assert all(p.exists() for p in (cpu_a, cpu_b, cpu_c, gpu_a, gpu_b))
        cpu_records = [json.loads(line) for line in cpu_marker.read_text().splitlines()]
        gpu_records = [json.loads(line) for line in gpu_marker.read_text().splitlines()]
        def event(rows, name, kind):
            return next(r["time"] for r in rows if r["name"] == name and r["event"] == kind)
        assert event(cpu_records, "cpu_a.ok", "end") <= event(cpu_records, "cpu_b.ok", "start"), cpu_records
        assert event(gpu_records, "gpu_a.ok", "end") <= event(gpu_records, "gpu_b.ok", "start")
        # CPU independent nodes overlap with two workers; GPU remains serial.
        assert abs(event(cpu_records, "cpu_a.ok", "start") - event(cpu_records, "cpu_c.ok", "start")) < 0.15
        first_cpu = event(cpu_records, "cpu_a.ok", "start")
        first_gpu = event(gpu_records, "gpu_a.ok", "start")
        assert abs(first_cpu - first_gpu) < 1.0, (first_cpu, first_gpu)
        # Process creation dominates this tiny fake workload on Windows.  This
        # is a spawn-watchdog bound, not a performance KPI; the timestamp
        # overlap assertion above remains the actual concurrency gate.
        assert elapsed < 35.0, elapsed
        log_text = log_path.read_text(encoding="utf-8")
        assert "scheduler marker task starting: cpu_a.ok" in log_text, log_text
        assert "scheduler marker task starting: gpu_a.ok" in log_text, log_text

        # Failure isolation: only the dependent node is blocked; an unrelated
        # CPU node still completes.
        independent = root / "independent.ok"
        isolated = ResourceScheduler(max_cpu_workers=2, max_gpu_workers=1).run([
            ScheduleTask("lightgbm", "dayahead", "2026-01-01", fail_task,
                         node_id="broken"),
            ScheduleTask("timesfm", "realtime", "2026-01-01", write_marker_task,
                         {"output_path": str(cpu_failed_child), "marker_path": str(cpu_marker)},
                         node_id="blocked", depends_on=("broken",)),
            ScheduleTask("sgdfnet", "realtime", "2026-01-01", write_marker_task,
                         {"output_path": str(independent), "marker_path": str(cpu_marker)},
                         node_id="independent"),
        ])
        assert independent.exists()
        assert {r.success for r in isolated} == {True, False}
        assert not cpu_failed_child.exists()

    print("check_split_resource_scheduler: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
