#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Experiment-only range runner for the reusable 24/96 classifier.

This command deliberately writes only to ``outputs/experiments``.  It is the
parity/performance proving ground; production ``ledger_classifier`` is not
changed until its output matches the reference implementation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ExtremPriceClf.merge_model.core.range_runner import (  # noqa: E402
    ClassifierRangeSpec,
    run_classifier_range,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reusable 24/96 classifier range replay in experiment area")
    parser.add_argument("--source", required=True, help="classifier input: xlsx/csv/parquet")
    parser.add_argument("--start", required=True, help="target start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="target end date YYYY-MM-DD")
    parser.add_argument("--resolution", default="hourly", choices=["hourly", "15min"])
    parser.add_argument("--task", default="realtime", choices=["realtime", "dayahead"])
    parser.add_argument("--output-dir", help="experiment output directory")
    parser.add_argument("--no-cache", action="store_true", help="use isolated cache namespace")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "outputs" / "experiments" / "classifier_range" / f"{args.start}_to_{args.end}"
    )
    spec = ClassifierRangeSpec(
        start_date=args.start,
        end_date=args.end,
        resolution=args.resolution,
        task=args.task,
    )
    result = run_classifier_range(
        project_root=PROJECT_ROOT,
        source=source,
        spec=spec,
        output_dir=output_dir,
        reuse_cache=not args.no_cache,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

