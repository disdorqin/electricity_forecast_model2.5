from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_compact_feature_study import FEATURE_PROFILES, candidate_config, load_matrix, target_days


ROOT = Path(__file__).resolve().parents[1]


def test_compact_runner_accepts_only_frozen_f0_f1_f2_and_dev14() -> None:
    matrix = load_matrix()
    assert set(FEATURE_PROFILES) == {"F0_A2_CORE5", "F1_A2_PHYSICAL_SHAPE", "F2_A2_CAUSAL_PRICE_STATE"}
    assert len(target_days(matrix)) == 14
    assert matrix["development_panel"] == "DEV14"


def test_compact_runner_changes_only_feature_profile_from_a2_config() -> None:
    base = json.loads((ROOT / "configs/business_strict34_core.json").read_text(encoding="utf-8"))
    for candidate_id, profile in FEATURE_PROFILES.items():
        config = candidate_config(base, candidate_id)
        assert config["training_history_months"] == 36
        assert config["validation_history_days"] == 28
        assert config["feature_package"] == profile
        assert config["training"]["loss"] == "MAE"
        assert config["training"]["seed"] == 42
        assert len(config["feature_profile"]["temporal_covariates"]) in {9, 17, 19}
