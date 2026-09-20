"""
Resource scheduler for ledger pipeline.

Manages CPU and GPU task queues with controlled concurrency.

CPU queue:  LightGBM, SGDFNet, TimesFM (fixed CPU), data processing
GPU queue:  TimeMixer, RT916

Default concurrency:
  max_cpu_workers = 2 (DAG-aware in production split-process mode)
  max_gpu_workers = 1

The 96-point production candidate uses ``resource_mode=split_process``:
one DAG-aware CPU child (at most two workers) and one strict-serial GPU child
start together.
GPU models are serialized to avoid CUDA OOM.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
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
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    # Stable graph identity.  A model may legitimately appear in more than
    # one task, so model_name alone is never a dependency identity.
    node_id: str | None = None
    depends_on: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.device == "auto":
            self.device = "gpu" if self.model_name in GPU_MODELS else "cpu"
        if self.node_id is None:
            self.node_id = f"{self.model_name}/{self.task_name}"


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
        resource_mode: str = "legacy",
        log_path: str | None = None,
    ):
        self.max_cpu_workers = max_cpu_workers
        self.max_gpu_workers = max_gpu_workers
        self.use_process_pool = use_process_pool
        self.resource_mode = resource_mode
        self.log_path = str(log_path) if log_path else None

    def run(self, tasks: list[ScheduleTask]) -> list[ScheduleResult]:
        """
        Execute all tasks with CPU/GPU queue management.

        Legacy mode preserves the existing scheduler behavior. The
        split_process mode uses one independent CPU child and one independent
        GPU child, both started before the parent waits.
        """
        cpu_tasks = [t for t in tasks if t.device == "cpu"]
        gpu_tasks = [t for t in tasks if t.device == "gpu"]

        logger.info(
            f"Scheduler: {len(cpu_tasks)} CPU tasks, "
            f"{len(gpu_tasks)} GPU tasks | "
            f"CPU workers={self.max_cpu_workers}, "
            f"GPU workers={self.max_gpu_workers} | mode={self.resource_mode}"
        )

        results: list[ScheduleResult] = []

        if self.resource_mode == "split_process":
            return self._run_split_process(cpu_tasks, gpu_tasks)

        # PyTorch's deterministic/debug algorithm policy is shared across the
        # process while adapters run in threads. TimeMixer GPU training
        # requires non-deterministic CUDA upsample backward, so overlapping
        # CPU adapters can change the policy mid-backward. Stable delivery
        # serializes the queues by default; overlap is benchmark-only opt-in.
        allow_overlap = os.getenv("EFM3_ALLOW_GPU_CPU_OVERLAP", "0") == "1"
        if cpu_tasks and gpu_tasks and not allow_overlap:
            logger.info("Scheduler: serializing CPU/GPU queues for Torch policy isolation")
            results.extend(self._run_queue(cpu_tasks, self.max_cpu_workers, "CPU"))
            results.extend(self._run_queue(gpu_tasks, self.max_gpu_workers, "GPU"))
            succeeded = sum(1 for r in results if r.success)
            failed = sum(1 for r in results if not r.success)
            logger.info(f"Scheduler done: {succeeded} OK, {failed} FAIL")
            return results

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

    def _run_split_process(
        self,
        cpu_tasks: list[ScheduleTask],
        gpu_tasks: list[ScheduleTask],
    ) -> list[ScheduleResult]:
        """Run one strict-serial CPU child and one strict-serial GPU child.

        The two children deliberately do not share a Torch/JAX process state.
        This is the production-safe replacement for the old thread-based
        overlap switch: CPU adapters cannot mutate CUDA deterministic policy,
        and the GPU child owns exactly one CUDA context.
        """
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        children: list[tuple[str, mp.Process]] = []
        results: list[ScheduleResult] = []

        for queue_name, queue_tasks in (("CPU", cpu_tasks), ("GPU", gpu_tasks)):
            if not queue_tasks:
                continue
            child = ctx.Process(
                target=_split_queue_entry,
                args=(queue_name, queue_tasks, result_queue,
                      self.max_cpu_workers if queue_name == "CPU" else 1,
                      self.log_path),
                name=f"efm3-{queue_name.lower()}-queue",
            )
            children.append((queue_name, child))

        # Start both resource queues before waiting on either one.
        for _, child in children:
            child.start()

        payloads: dict[str, dict] = {}
        for queue_name, child in children:
            child.join()
            if child.exitcode != 0:
                logger.error(
                    "Scheduler %s child exited with code %s",
                    queue_name,
                    child.exitcode,
                )

        # A multiprocessing.Queue has a feeder thread; allow it to flush
        # after join instead of relying on a racy get_nowait().
        expected_queues = {name for name, _ in children}
        deadline = time.monotonic() + 10.0
        while expected_queues - payloads.keys() and time.monotonic() < deadline:
            try:
                payload = result_queue.get(timeout=0.25)
            except Exception:
                continue
            payloads[payload["queue"]] = payload

        for queue_name, queue_tasks in (("CPU", cpu_tasks), ("GPU", gpu_tasks)):
            payload = payloads.get(queue_name)
            if payload is None:
                for task in queue_tasks:
                    results.append(
                        ScheduleResult(
                            model_name=task.model_name,
                            task_name=task.task_name,
                            target_date=task.target_date,
                            success=False,
                            error=f"{queue_name} child produced no result manifest",
                        )
                    )
                continue
            for item in payload.get("results", []):
                results.append(ScheduleResult(**item))

        succeeded = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        logger.info("Split-process scheduler done: %s OK, %s FAIL", succeeded, failed)
        return results

    def _run_queue(
        self,
        tasks: list[ScheduleTask],
        max_workers: int,
        queue_name: str,
    ) -> list[ScheduleResult]:
        """Run only dependency-ready nodes, isolating failed descendants.

        ``depends_on`` is a graph dependency between node identities; legacy
        ``dependencies`` remains a file-existence gate.  This avoids submitting
        a realtime task before its prerequisite has produced its output.
        """
        pending = {task.node_id: task for task in tasks}
        if len(pending) != len(tasks):
            raise ValueError(f"{queue_name} queue has duplicate node_id values")
        unknown = {
            dep for task in tasks for dep in task.depends_on if dep not in pending
        }
        if unknown:
            raise ValueError(f"{queue_name} queue has unknown graph dependencies: {sorted(unknown)}")

        results: list[ScheduleResult] = []
        state: dict[str, bool] = {}
        running: dict[Future, ScheduleTask] = {}
        worker_count = max(1, int(max_workers))

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            while pending or running:
                # A failed prerequisite blocks only its descendants.  Other
                # disconnected nodes remain eligible in the same iteration.
                for node_id, task in list(pending.items()):
                    failed = [dep for dep in task.depends_on if state.get(dep) is False]
                    if failed:
                        results.append(ScheduleResult(
                            model_name=task.model_name, task_name=task.task_name,
                            target_date=task.target_date, success=False,
                            error="failed graph dependencies: " + ", ".join(failed),
                        ))
                        state[node_id] = False
                        del pending[node_id]

                while len(running) < worker_count:
                    ready = next((
                        task for task in pending.values()
                        if all(state.get(dep) is True for dep in task.depends_on)
                    ), None)
                    if ready is None:
                        break
                    del pending[ready.node_id]
                    running[pool.submit(self._execute_task, ready)] = ready

                if not running:
                    if pending:
                        unresolved = {key: task.depends_on for key, task in pending.items()}
                        raise RuntimeError(f"{queue_name} DAG made no progress: {unresolved}")
                    continue

                completed = next(as_completed(running))
                task = running.pop(completed)
                try:
                    result = completed.result()
                except Exception as exc:  # defensive: _execute_task normally captures
                    result = ScheduleResult(
                        model_name=task.model_name, task_name=task.task_name,
                        target_date=task.target_date, success=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                results.append(result)
                state[task.node_id] = result.success
        return results

    def _execute_task(self, task: ScheduleTask) -> ScheduleResult:
        """Execute a single task in the current process."""
        missing_dependencies = [p for p in task.dependencies if not os.path.exists(p)]
        if missing_dependencies:
            return ScheduleResult(
                model_name=task.model_name,
                task_name=task.task_name,
                target_date=task.target_date,
                success=False,
                error=(
                    "missing task dependencies: "
                    + ", ".join(missing_dependencies)
                ),
            )
        # Reproducibility: set seed in the executing thread/process
        from utils.reproducibility import set_global_seed

        set_global_seed(
            int(task.kwargs.get("seed", 42)),
            bool(task.kwargs.get("deterministic", False)),
        )
        # TimeMixer uses CUDA upsample backward, which has no strict
        # deterministic implementation in the project Torch/CUDA baseline.
        # Re-assert its GPU policy at the scheduler boundary because the
        # deterministic switch is shared process/thread state while other
        # model adapters run concurrently.
        if task.model_name == "timemixer":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.use_deterministic_algorithms(False, warn_only=False)
                    if hasattr(torch, "set_deterministic_debug_mode"):
                        torch.set_deterministic_debug_mode("default")
                    torch.backends.cudnn.deterministic = False
            except Exception:
                logger.debug("Could not reset TimeMixer CUDA algorithm policy", exc_info=True)
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


def _install_process_log_handler(log_path: str | None) -> None:
    """Attach split-process model logs to the parent run's pipeline.log.

    Spawned workers do not inherit the parent's FileHandler on Windows (or
    under explicit spawn elsewhere).  Adding the same append-only run log in
    each child keeps model-module logging in the canonical daily log instead of
    forcing models to create their own files.
    """
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if str(Path(handler.baseFilename).resolve()) == resolved:
                    return
            except (OSError, TypeError, ValueError):
                continue
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)


def _split_queue_entry(
    queue_name: str,
    tasks: list[ScheduleTask],
    result_queue: Any,
    max_workers: int,
    log_path: str | None = None,
) -> None:
    """Child entrypoint for the split-process scheduler.

    Environment variables are set before the first model adapter is imported.
    Results intentionally omit the in-memory model output because adapters
    write validated prediction files themselves; this keeps IPC small.
    """
    _install_process_log_handler(log_path)

    if queue_name == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TIMESFM_DEVICE"] = "cpu"
        os.environ["JAX_PLATFORMS"] = "cpu"
        thread_budget = os.environ.get("EFM3_CPU_THREAD_BUDGET", "24")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            os.environ.get("EFM3_GPU_DEVICE")
            or os.environ.get("CUDA_VISIBLE_DEVICES")
            or "0"
        )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # These are small, launch-bound models.  Letting a 256-vCPU host
        # create 128 Torch threads per child causes severe oversubscription.
        thread_budget = os.environ.get("EFM3_GPU_THREAD_BUDGET", "8")

    # Set BLAS/OpenMP limits before importing any model adapter.  The explicit
    # Torch limits below cover builds that ignore the environment variables.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = thread_budget
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch
        torch.set_num_threads(int(thread_budget))
        torch.set_num_interop_threads(1)
    except Exception:
        logger.debug("Could not apply queue thread budget", exc_info=True)

    scheduler = ResourceScheduler(
        max_cpu_workers=max_workers if queue_name == "CPU" else 1,
        max_gpu_workers=1,
        use_process_pool=False,
        resource_mode="legacy",
    )
    results = scheduler._run_queue(tasks, max_workers if queue_name == "CPU" else 1, queue_name)
    payload = {
        "queue": queue_name,
        "results": [
            {
                "model_name": result.model_name,
                "task_name": result.task_name,
                "target_date": result.target_date,
                "success": result.success,
                "output": None,
                "error": result.error,
                "elapsed_seconds": result.elapsed_seconds,
            }
            for result in results
        ],
    }
    result_queue.put(payload)


def classify_model_device(model_name: str) -> str:
    """Return "cpu" or "gpu" for a given model name."""
    if model_name in GPU_MODELS:
        return "gpu"
    return "cpu"
