from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import SGDFNet.pipeline as sgpipe


def test_sgdfnet_wrapper_preserves_anchor_metadata(tmp_path, monkeypatch):
    target = "2026-08-16"
    run_dir = tmp_path / "core_run"
    run_dir.mkdir(parents=True)

    rows = []
    start = pd.Timestamp(target)
    for period in range(1, 97):
        ts = start + pd.Timedelta(minutes=15 * period)
        rows.append(
            {
                "timestamp": ts,
                "business_day": target,
                "rt_hat": 100.0 + period,
                "anchor_source_day": "2026-08-15",
                "anchor_source_type": "decision_day_da",
                "anchor_rows": 96,
                "fallback_used": False,
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "predictions.csv", index=False)

    monkeypatch.setattr(
        sgpipe,
        "run_protocol_b_cutoff_experiment",
        lambda _config_path: run_dir,
    )

    pipeline = sgpipe.ModelPipeline()
    result = pipeline.predict_range(
        target="realtime",
        predict_date=target,
        start=target,
        end=target,
        resolution="15min",
        realtime_cutoff_hour=15,
        data_path="unused-by-mocked-core.parquet",
        output_root=tmp_path / "runtime",
    )

    frame = result.frame
    assert len(frame) == 96
    assert frame["时刻"].notna().all()
    assert set(frame["anchor_source_day"].astype(str).str[:10]) == {"2026-08-15"}
    assert set(frame["anchor_source_type"]) == {"decision_day_da"}
    assert set(pd.to_numeric(frame["anchor_rows"]).astype(int)) == {96}
    assert not frame["fallback_used"].astype(bool).any()


def test_sgdfnet_dynamic_config_uses_short_runtime_path(tmp_path):
    pipeline = sgpipe.ModelPipeline()
    config_path = pipeline._build_temp_config(
        data_path="dummy.parquet",
        start_day="2026-09-20",
        end_day="2026-09-20",
        output_root=str(tmp_path / "sgdfnet" / "realtime"),
        decision_hour=15,
        resolution=96,
        seed=42,
        deterministic=True,
        dynamic_serving=True,
    )
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    assert payload["dynamic_serving"] is True
    assert payload["experiment_name"] == "dyn"
    assert Path(payload["output_root"]).name == "realtime"
