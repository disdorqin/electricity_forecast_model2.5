from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ExtremPriceClf.merge_model.core.range_runner import (  # noqa: E402
    ClassifierRangeSpec,
    prepare_classifier_cache,
)
from utils.asof_view_96 import (  # noqa: E402
    build_asof_view_96,
    cleanup_transient_asof_96,
    transient_asof_path_96,
)
from utils.data_layout import DATA  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate 96-point classifier cache migration under the formal as-of contract."
    )
    parser.add_argument("--date", default="2026-08-15")
    parser.add_argument("--cutoff-hour", type=int, default=15)
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA.model_96_full_parquet,
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "96",
    )
    parser.add_argument(
        "--expected-p1-through",
        default="2026-08-14 23:00:00",
        help="Minimum expected cached timestamp after legacy adoption.",
    )
    args = parser.parse_args(argv)

    asof_path = transient_asof_path_96(args.date)
    try:
        _, audit = build_asof_view_96(
            source_path=args.source,
            target_day=args.date,
            cutoff_hour=args.cutoff_hour,
            output_path=asof_path,
        )
        spec = ClassifierRangeSpec(
            start_date=args.date,
            end_date=args.date,
            resolution="15min",
            task="realtime",
        )
        _, layout, feature_cache_hit = prepare_classifier_cache(
            project_root=PROJECT_ROOT,
            source=asof_path,
            spec=spec,
            feature_store_root=args.cache_root,
        )
        manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
        reuse = manifest.get("p1_cache_reuse", {})
        if not layout.p1_cache.exists():
            raise RuntimeError("classifier p1 cache was not prepared")

        p1 = pd.read_parquet(layout.p1_cache)
        if "时刻" not in p1.columns or p1.empty:
            raise RuntimeError("classifier p1 cache is empty or missing 时刻")
        p1_max = pd.to_datetime(p1["时刻"], errors="coerce").max()
        expected = pd.Timestamp(args.expected_p1_through)
        if pd.isna(p1_max) or p1_max < expected:
            raise RuntimeError(
                f"classifier p1 cache ends too early: {p1_max}; expected >= {expected}"
            )
        if reuse.get("status") not in {"adopted", "semantic_prefix_reuse", "same_source_reuse"}:
            raise RuntimeError(f"classifier p1 cache was not safely reused: {reuse}")

        result = {
            "status": "PASS",
            "target_date": args.date,
            "asof": {
                "decision_day_da_visible": audit["decision_day_da_visible"],
                "decision_day_rt_visible": audit["decision_day_rt_visible"],
                "target_realized_nonnull": audit["target_realized_nonnull"],
                "target_forecast_nonnull": audit["target_forecast_nonnull"],
            },
            "cache_dir": str(layout.root),
            "feature_cache_hit": bool(feature_cache_hit),
            "p1_reuse": reuse,
            "p1_rows": int(len(p1)),
            "p1_cached_through": str(p1_max),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    finally:
        cleanup_transient_asof_96(asof_path)


if __name__ == "__main__":
    raise SystemExit(main())
