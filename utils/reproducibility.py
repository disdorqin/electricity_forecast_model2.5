"""全局随机种子与可复现性工具。

用法:
    from utils.reproducibility import set_global_seed
    set_global_seed(42, deterministic=True)
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set project-level random seeds for reproducible model runs."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)

    # 历史上这里无条件 `cudnn.benchmark=False` + `float32_matmul_precision("highest")`
    # 会把 TF32 二次关掉（GPU matmul 变慢）。实测对小模型开 benchmark 反而有搜索开销，
    # 因此这里只放开 TF32，benchmark 交由 perf_knobs/模型自行决定（默认不开）。
    # deterministic=True 时保持完全可复现（关 TF32 + 关 benchmark）。
    if bool(deterministic):
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")  # 关 TF32，保证逐位可复现
    else:
        from optim.perf_knobs import _env_bool

        if _env_bool("OPTIM_TF32", "1") and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
