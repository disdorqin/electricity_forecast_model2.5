from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of one immutable input/artifact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON for manifests and configuration identity."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_dict_hash(model: torch.nn.Module) -> str:
    """Hash state tensors in sorted key order, independent of object identity."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def source_tree_hash(root: str | Path, *, suffixes: tuple[str, ...] = (".py", ".json")) -> str:
    """Hash source/config files in deterministic relative-path order."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def git_identity(root: str | Path) -> dict[str, Any]:
    """Record git identity without requiring the cycle itself to be tracked."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short", "--", str(root)], cwd=root, text=True)
        tracked = bool(subprocess.check_output(["git", "ls-files", "--", str(root)], cwd=root, text=True).strip())
        return {"commit": commit, "cycle_status": status.splitlines(), "cycle_has_tracked_files": tracked}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "cycle_status": [], "cycle_has_tracked_files": False, "error": str(exc)}


def environment_identity(*, device: torch.device, deterministic: bool, seed: int) -> dict[str, Any]:
    """Return the runtime identity needed to reproduce a training process."""
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "float32",
        "amp": False,
        "deterministic_algorithms": bool(deterministic),
        "seed": int(seed),
        "resolution": "hourly",
        "forecast_origin": "D-1 14:00",
    }


def artifact_reuse_audit(
    run_dir: str | Path,
    *,
    config: Any,
    source_path: str | Path,
    source_code_hash: str,
    device: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Verify identity before reusing a previously materialized target run."""
    run_dir = Path(run_dir)
    provenance_path = run_dir / "provenance.json"
    manifest_path = run_dir / "manifest.json"
    if not provenance_path.exists() or not manifest_path.exists():
        return [{"field": "artifact_exists", "status": "FAIL"}]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "config_sha256": sha256_json(config),
        "source_data_sha256": sha256_file(source_path),
        "source_code_sha256": source_code_hash,
        "device": str(device),
        "seed": int(seed),
    }
    actual = {
        "config_sha256": provenance.get("config_sha256"),
        "source_data_sha256": provenance.get("source_data_sha256"),
        "source_code_sha256": provenance.get("source_code_sha256"),
        "device": provenance.get("device", {}).get("device"),
        "seed": provenance.get("device", {}).get("seed"),
    }
    rows = [
        {"field": key, "expected": value, "actual": actual[key], "status": "MATCH" if actual[key] == value else "FAIL"}
        for key, value in expected.items()
    ]
    rows.append({
        "field": "target_day_sample_count",
        "expected": 24,
        "actual": manifest.get("target_day_sample_count"),
        "status": "MATCH" if manifest.get("target_day_sample_count") == 24 else "FAIL",
    })
    return rows
