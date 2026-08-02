"""
Resource scheduler for ledger pipeline.

Manages CPU and GPU task queues with controlled concurrency.

CPU queue:  LightGBM, SGDFNet, TimesFM (fixed CPU), data processing
GPU queue:  TimeMixer, RT916

Default concurrency:
  max_cpu_workers = 2
  max_gpu_workers = 1

CPU and GPU queues run concurrently.
GPU models are serialized to avoid CUDA OOM.
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Model → device classification
# TimesFM is fixed CPU in the ledger pipeline (not GPU)
CPU_MODELS = {"lightgbm", "sgdfnet", "timesfm"}
GPU_MODELS = {"timemixer", "rt916"}


@dataclass
class ScheduleTask:
    """A single model prediction task for the scheduler."""

    model_name: str
    task_name: str  # "dayahead" or "realtime"
    target_date: str
    fn: Callable[..., Any]
    kwargs: dict = field(default_factory=dict)
    device: str = "auto"  # "cpu", "gpu", or "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "gpu" if self.model_name in GPU_MODELS else "cpu"


@dataclass
class ScheduleResult:
    """Result of a scheduled task."""

    model_name: str
    task_name: str
    target_date: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


class ResourceScheduler:
    """
    Schedules model prediction tasks across CPU and GPU queues.

    Usage:
        scheduler = ResourceScheduler(max_cpu_workers=2, max_gpu_workers=1)
        tasks = [
            ScheduleTask("lightgbm", "dayahead", "2026-02-24", predict_fn),
            ScheduleTask("timemixer", "dayahead", "2026-02-24", predict_fn),
        ]
        results = scheduler.run(tasks)
    """

    def __init__(
        self,
        max_cpu_workers: int = 2,
        max_gpu_workers: int = 1,
        use_process_pool: bool = True,
    ):
        self.max_cpu_workers = max_cpu_workers
        self.max_gpu_workers = max_gpu_workers
        self.use_process_pool = use_process_pool

    def run(self, tasks: list[ScheduleTask]) -> list[ScheduleResult]:
        """
        Execute all tasks with CPU/GPU queue management.

        CPU tasks run in parallel (up to max_cpu_workers).
        GPU tasks run sequentially (max_gpu_workers=1).

        CPU and GPU queues run concurrently.
        """
        cpu_tasks = [t for t in tasks if t.device == "cpu"]
        gpu_tasks = [t for t in tasks if t.device == "gpu"]

        logger.info(
            f"Scheduler: {len(cpu_tasks)} CPU tasks, "
            f"{len(gpu_tasks)} GPU tasks | "
            f"CPU workers={self.max_cpu_workers}, "
            f"GPU workers={self.max_gpu_workers}"
        )

        results: list[ScheduleResult] = []

        # Run CPU and GPU queues concurrently using threads
        cpu_futures: list[Future] = []
        gpu_futures: list[Future] = []

        with ThreadPoolExecutor(max_workers=2) as pool:
            if cpu_tasks:
                cpu_future = pool.submit(
                    self._run_queue, cpu_tasks, self.max_cpu_workers, "CPU"
                )
                cpu_futures.append(("CPU", cpu_future))

            if gpu_tasks:
                gpu_future = pool.submit(
                    self._run_queue, gpu_tasks, self.max_gpu_workers, "GPU"
                )
                gpu_futures.append(("GPU", gpu_future))

            # Collect results
            for label, future in cpu_futures + gpu_futures:
                try:
                    queue_results = future.result()
                    results.extend(queue_results)
                except Exception as e:
                    logger.error(f"{label} queue failed: {e}")

        # Report
        succeeded = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        logger.info(f"Scheduler done: {succeeded} OK, {failed} FAIL")

        return results

    def _run_queue(
        self,
        tasks: list[ScheduleTask],
        max_workers: int,
        queue_name: str,
    ) -> list[ScheduleResult]:
        """Run a queue of tasks with the specified concurrency."""
        results: list[ScheduleResult] = []

        if max_workers <= 1 or len(tasks) <= 1:
            # Sequential execution
            for task in tasks:
                result = self._execute_task(task)
                results.append(result)
        else:
            # Parallel execution — 用线程池而非进程池：GPU 模型共享主进程 CUDA 上下文，
            # 避免多进程各自 init CUDA 导致 CUDA error: initialization error。
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_map: dict[Future, ScheduleTask] = {}
                for task in tasks:
                    future = pool.submit(
                        self._execute_task,
                        task,
                    )
                    future_map[future] = task

                for future in as_completed(future_map):
                    task = future_map[future]
                    try:
                        output = future.result()
                        result = ScheduleResult(
                            model_name=task.model_name,
                            task_name=task.task_name,
                            target_date=task.target_date,
                            success=True,
                            output=output,
                            elapsed_seconds=0.0,
                        )
                    except Exception as e:
                        result = ScheduleResult(
                            model_name=task.model_name,
                            task_name=task.task_name,
                            target_date=task.target_date,
                            success=False,
                            error=f"{type(e).__name__}: {e}",
                        )
                        logger.error(
                            f"{queue_name} [{task.model_name}/{task.task_name}] "
                            f"FAILED: {e}\n{traceback.format_exc()}"
                        )
                    results.append(result)

        return results

    def _execute_task(self, task: ScheduleTask) -> ScheduleResult:
        """Execute a single task in the current process."""
        # Reproducibility: set seed in the executing thread/process
        from utils.reproducibility import set_global_seed

        set_global_seed(
            int(task.kwargs.get("seed", 42)),
            bool(task.kwargs.get("deterministic", False)),
        )
        logger.info(
            f"[{task.device.upper()}] {task.model_name}/{task.task_name} "
            f"on {task.target_date} starting..."
        )
        t0 = time.perf_counter()
        try:
            output = task.fn(**task.kwargs)
            elapsed = time.perf_counter() - t0
            logger.info(
                f"[{task.device.upper()}] {task.model_name}/{task.task_name} "
                f"done in {elapsed:.1f}s"
            )
            return ScheduleResult(
                model_name=task.model_name,
                task_name=task.task_name,
                target_date=task.target_date,
                success=True,
                output=output,
                elapsed_seconds=elapsed,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error(
                f"[{task.device.upper()}] {task.model_name}/{task.task_name} "
                f"FAILED in {elapsed:.1f}s: {e}"
            )
            return ScheduleResult(
                model_name=task.model_name,
                task_name=task.task_name,
                target_date=task.target_date,
                success=False,
                error=f"{type(e).__name__}: {e}",
                elapsed_seconds=elapsed,
            )


def _execute_in_subprocess(fn: Callable, kwargs: dict) -> Any:
    """Wrapper for ProcessPoolExecutor — function must be picklable."""
    from utils.reproducibility import set_global_seed

    set_global_seed(
        int(kwargs.get("seed", 42)),
        bool(kwargs.get("deterministic", False)),
    )
    return fn(**kwargs)


def classify_model_device(model_name: str) -> str:
    """Return "cpu" or "gpu" for a given model name."""
    if model_name in GPU_MODELS:
        return "gpu"
    return "cpu"
