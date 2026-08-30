import json
from pathlib import Path

from nbeatsx_spread.training.provenance import artifact_reuse_audit, sha256_file, sha256_json, source_tree_hash


def test_artifact_reuse_hash_gate_accepts_matching_identity():
    cycle = Path(__file__).parents[1]
    project_root = next(p for p in cycle.parents if (p / "data").exists())
    data_path = project_root / "data/24/canonical/shandong_pmos_hourly.csv"
    config = json.loads((cycle / "configs/business_strict34_core.json").read_text(encoding="utf-8"))
    code_hash = source_tree_hash(cycle / "src")
    run_dir = cycle / "runs/test_artifact_reuse_hash_gate/2026-06-01"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(json.dumps({
        "config_sha256": sha256_json(config),
        "source_data_sha256": sha256_file(data_path),
        "source_code_sha256": code_hash,
        "device": {"device": "cpu", "seed": 42},
    }), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({"target_day_sample_count": 24}), encoding="utf-8")
    rows = artifact_reuse_audit(
        run_dir, config=config, source_path=data_path, source_code_hash=code_hash, device="cpu", seed=42
    )
    assert all(row["status"] == "MATCH" for row in rows)
