"""Fast contract checks for the formal 96-point model DAG (no model training)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pipelines.ledger_predict as ledger_predict
from runtime.resource_scheduler import ScheduleResult


class _RecordingScheduler:
    created: list["_RecordingScheduler"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tasks = []
        self.__class__.created.append(self)

    def run(self, tasks):
        self.tasks = list(tasks)
        return [
            ScheduleResult(t.model_name, t.task_name, t.target_date, True)
            for t in self.tasks
        ]


def _plan(da: tuple[str, ...], rt: tuple[str, ...], *, force: bool = True, cache_timesfm_da: bool = False):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        if cache_timesfm_da:
            cache_dir = root / "dayahead" / "prediction"
            cache_dir.mkdir(parents=True)
            periods = range(1, 97)
            cached = pd.DataFrame({
                "business_day": "2026-01-01",
                "business_period": periods,
                "ds": pd.date_range("2026-01-01 00:15", periods=96, freq="15min"),
                "y_pred": 1.0,
            })
            cached.to_csv(cache_dir / "timesfm_predictions.csv", index=False)
        original = ledger_predict.ResourceScheduler
        ledger_predict.ResourceScheduler = _RecordingScheduler
        try:
            result = ledger_predict._run_unified_model_plan(
                target_date="2026-01-01", selected_da_models=da, selected_rt_models=rt,
                common_kwargs={"data_path": __file__, "resolution": "15min"},
                feature_view_paths={}, da_cutoff_date="2025-12-31",
                rt_cutoff_date="2025-12-31", run_dir=root, max_cpu=99, max_gpu=99,
                force=force,
            )
            scheduler = _RecordingScheduler.created[-1]
            return result, scheduler
        finally:
            ledger_predict.ResourceScheduler = original


def main() -> int:
    da_result, da_scheduler = _plan(("lightgbm", "timesfm", "timemixer"), ())
    assert {(t.model_name, t.task_name) for t in da_scheduler.tasks} == {
        ("lightgbm", "dayahead"), ("timesfm", "dayahead"), ("timemixer", "dayahead"),
    }
    assert set(da_result["dayahead"]) == {"lightgbm", "timesfm", "timemixer"}
    assert not da_result["realtime"]

    rt_result, rt_scheduler = _plan((), ("timesfm", "sgdfnet", "timemixer", "rt916"))
    nodes = {(t.model_name, t.task_name) for t in rt_scheduler.tasks}
    assert nodes == {
        ("timesfm", "dayahead"), ("timesfm", "realtime"),
        ("sgdfnet", "anchor_prepare"), ("sgdfnet", "realtime"),
        ("timemixer", "dayahead"), ("timemixer", "realtime"), ("rt916", "realtime"),
    }
    assert set(rt_result["realtime"]) == {"timesfm", "sgdfnet", "timemixer", "rt916"}
    assert not rt_result["dayahead"]
    deps = {t.node_id: t.depends_on for t in rt_scheduler.tasks}
    assert deps["timesfm/realtime"] == ("timesfm/dayahead",)
    assert deps["timemixer/realtime"] == ("timemixer/dayahead",)
    assert deps["sgdfnet/realtime"] == ("sgdfnet/anchor_prepare",)
    assert deps["rt916/realtime"] == ()
    assert rt_scheduler.kwargs["max_cpu_workers"] == 2
    assert rt_scheduler.kwargs["max_gpu_workers"] == 1
    assert rt_scheduler.kwargs["resource_mode"] == "split_process"
    assert Path(rt_scheduler.kwargs["log_path"]).name == "pipeline.log"
    assert Path(rt_scheduler.kwargs["log_path"]).parent.name == "logs"

    # A cached DA prerequisite is a satisfied graph node, not an unknown
    # dependency for a fresh realtime child.
    _, cached_scheduler = _plan(
        (), ("timesfm",), force=False, cache_timesfm_da=True
    )
    cached_deps = {t.node_id: t.depends_on for t in cached_scheduler.tasks}
    assert cached_deps["timesfm/realtime"] == ()
    print("check_96_production_dag: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
