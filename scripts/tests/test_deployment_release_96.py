from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.server.build_predictor_release import (
    PROJECT_ROOT,
    build_release_manifest,
    iter_release_files,
    materialize_release,
)
from scripts.server.doctor_96_deployment import (
    RELEASE_CONTRACT,
    check_layout,
    check_release_manifest,
)
from TimesFMBackend.price_forecast_copy_分时段预测 import (
    _resolve_timesfm_model_dir,
)


def test_release_plan_is_whitelist_and_secret_free():
    files = {path.as_posix() for path in iter_release_files(PROJECT_ROOT)}

    assert "main.py" in files
    assert "requirements.txt" in files
    assert ".env.example" in files
    assert "models/LightGBM/best_model_日前电价.pkl" in files
    assert "models/timesFM/model.safetensors" in files
    assert "models/timesFM/config.json" in files

    forbidden_prefixes = (
        "outputs/",
        "data/",
        "deliverables/",
        "fixtures/",
        "_archive/",
        "dist/",
        "build/",
        "scripts/tests/",
        "scripts/experiments/",
        "scripts/crawler/",
        "ExtremPriceClf/",
        "TimeMixer/outputs_v2/",
    )
    assert not any(
        path.startswith(prefix)
        for path in files
        for prefix in forbidden_prefixes
    )
    assert ".env" not in files
    assert "dist/crawler/config.json" not in files
    assert "dist/crawler/db_config.json" not in files


def test_release_manifest_dry_plan_marks_state_and_crawler_excluded():
    manifest = build_release_manifest(PROJECT_ROOT, hash_files=False)

    assert manifest["contract"] == RELEASE_CONTRACT
    assert manifest["file_count"] > 0
    assert manifest["total_bytes"] > 0
    assert manifest["includes_models"] is True
    assert manifest["mutable_state_included"] is False
    assert manifest["crawler_included"] is False
    assert ".env.example" in {item["path"] for item in manifest["files"]}
    assert all("sha256" not in item for item in manifest["files"])


def test_materialize_release_refuses_nonempty_destination(tmp_path: Path):
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError):
        materialize_release(
            destination,
            {"files": []},
            project_root=PROJECT_ROOT,
        )


def test_release_manifest_hash_verification_roundtrip(tmp_path: Path):
    payload_file = tmp_path / "main.py"
    payload_file.write_text("print('ok')\n", encoding="utf-8")

    import hashlib

    sha = hashlib.sha256(payload_file.read_bytes()).hexdigest()
    manifest = {
        "contract": RELEASE_CONTRACT,
        "files": [
            {
                "path": "main.py",
                "size_bytes": payload_file.stat().st_size,
                "sha256": sha,
            }
        ],
    }
    manifest_path = tmp_path / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    checks = check_release_manifest(tmp_path, manifest_path)
    assert len(checks) == 1
    assert checks[0].status == "PASS"

    payload_file.write_text("tampered\n", encoding="utf-8")
    checks = check_release_manifest(tmp_path, manifest_path)
    assert checks[0].status == "FAIL"


def test_timesfm_model_dir_ignores_generic_project_root(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path / "wrong_checkout"))
    monkeypatch.delenv("TIMESFM_MODEL_DIR", raising=False)
    resolved = _resolve_timesfm_model_dir()
    assert resolved == PROJECT_ROOT / "models" / "timesFM"

    explicit = tmp_path / "external_timesfm"
    monkeypatch.setenv("TIMESFM_MODEL_DIR", str(explicit))
    assert _resolve_timesfm_model_dir() == explicit


def test_source_checkout_has_formal_paths_but_is_not_a_minimal_release():
    checks = check_layout(PROJECT_ROOT, strict_release=True)
    by_name = {item.name: item for item in checks}

    assert by_name["release_layout"].status == "PASS"
    assert by_name["release_hygiene"].status == "FAIL"
