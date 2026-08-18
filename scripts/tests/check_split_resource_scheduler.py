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
from runtime.scheduler_test_helpers import write_marker_task


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="efm3-scheduler-") as tmp:
        root = Path(tmp)
        marker = root / "markers.jsonl"
        cpu_a = root / "cpu_a.ok"
        cpu_b = root / "cpu_b.ok"
        gpu_a = root / "gpu_a.ok"
        gpu_b = root / "gpu_b.ok"
        tasks = [
            ScheduleTask(
                "lightgbm", "dayahead", "2026-01-01", write_marker_task,
                {"output_path": str(cpu_a), "marker_path": str(marker)}, "cpu",
            ),
            ScheduleTask(
                "timesfm", "realtime", "2026-01-01", write_marker_task,
                {"output_path": str(cpu_b), "marker_path": str(marker)}, "cpu",
                dependencies=(str(cpu_a),),
            ),
            ScheduleTask(
                "timemixer", "dayahead", "2026-01-01", write_marker_task,
                {"output_path": str(gpu_a), "marker_path": str(marker)}, "gpu",
            ),
            ScheduleTask(
                "rt916", "realtime", "2026-01-01", write_marker_task,
                {"output_path": str(gpu_b), "marker_path": str(marker)}, "gpu",
                dependencies=(str(gpu_a),),
            ),
        ]
        started = time.perf_counter()
        results = ResourceScheduler(
            max_cpu_workers=1, max_gpu_workers=1, resource_mode="split_process"
        ).run(tasks)
        elapsed = time.perf_counter() - started

        assert len(results) == 4, results
        assert all(r.success for r in results), results
        assert all(p.exists() for p in (cpu_a, cpu_b, gpu_a, gpu_b))
        records = [json.loads(line) for line in marker.read_text().splitlines()]
        names = [r["name"] for r in records]
        assert names.index("cpu_a.ok") < names.index("cpu_b.ok")
        assert names.index("gpu_a.ok") < names.index("gpu_b.ok")
        first_cpu = next(r["time"] for r in records if r["name"] == "cpu_a.ok")
        first_gpu = next(r["time"] for r in records if r["name"] == "gpu_a.ok")
        assert abs(first_cpu - first_gpu) < 1.0, (first_cpu, first_gpu)
        # Process creation dominates this tiny fake workload on Windows; the
        # timestamp overlap assertion above is the actual concurrency gate.
        assert elapsed < 10.0, elapsed

    print("check_split_resource_scheduler: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
