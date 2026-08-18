from __future__ import annotations

import argparse
from datetime import datetime


def _parse_yyyy_mm_dd(value: str, parser: argparse.ArgumentParser, field_name: str) -> str:
    """Validate and return a YYYY-MM-DD date string. Raises parser.error on failure."""
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
        return value
    except ValueError:
        parser.error(
            f"Invalid date for {field_name}: '{value}'. "
            f"Expected YYYY-MM-DD format (e.g. 2026-02-24)."
        )


def normalize_date_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """
    Normalize date-related arguments after parse_args().

    Handles:
      - Single positional  → args.date = value, pipeline = ledger_full
      - Two positionals    → args.start/end = values, pipeline = ledger_full_range
      - --start/--end      → auto-switch to ledger_full_range if not explicitly set
      - Conflict detection  → parser.error(...)
      - Date validation    → ensures YYYY-MM-DD format
    """
    # Validate date format for any provided date values
    if args.pos_date is not None:
        args.pos_date = _parse_yyyy_mm_dd(args.pos_date, parser, "pos_date")
    if args.pos_end is not None:
        args.pos_end = _parse_yyyy_mm_dd(args.pos_end, parser, "pos_end")
    if args.date is not None:
        args.date = _parse_yyyy_mm_dd(args.date, parser, "--date")
    if args.start is not None:
        args.start = _parse_yyyy_mm_dd(args.start, parser, "--start")
    if args.end is not None:
        args.end = _parse_yyyy_mm_dd(args.end, parser, "--end")

    # --- Conflict detection ---
    has_range_args = args.start is not None or args.end is not None

    if args.pos_date is not None and args.pos_end is not None:
        # Two positionals
        if args.date is not None:
            parser.error("Cannot use both positional dates and --date")
        if has_range_args:
            parser.error("Cannot use both positional dates and --start/--end")
        args.start = args.pos_date
        args.end = args.pos_end
        args.pipeline = "ledger_full_range"

    elif args.pos_date is not None:
        # Single positional
        if args.date is not None:
            parser.error("Cannot use both positional date and --date")
        if has_range_args:
            parser.error("Cannot use positional date together with --start/--end")
        args.date = args.pos_date

    # Explicit --date conflicts with --start/--end
    if args.date is not None and has_range_args:
        parser.error("Cannot use --date together with --start/--end")

    # Auto-switch to range mode when --start/--end are provided with default pipeline
    if has_range_args:
        if not args.start or not args.end:
            parser.error("Range mode requires both --start and --end")
        if args.pipeline == "ledger_full":
            args.pipeline = "ledger_full_range"

    # --- Pipeline-specific validations ---
    if args.pipeline == "ledger_full_range":
        if not args.start or not args.end:
            parser.error("ledger_full_range requires --start and --end (or two positional dates)")
        if getattr(args, "predict_only", False) and getattr(args, "replay_only", False):
            parser.error("--predict-only and --replay-only are mutually exclusive")
        # Validate start <= end using parsed dates
        if datetime.strptime(args.start, "%Y-%m-%d") > datetime.strptime(args.end, "%Y-%m-%d"):
            parser.error(f"--start ({args.start}) must be <= --end ({args.end})")
    elif args.pipeline in ("ledger_full", "ledger_predict", "ledger_weight",
                           "ledger_fuse", "ledger_classifier", "ledger_smoke"):
        if getattr(args, "predict_only", False) or getattr(args, "replay_only", False):
            parser.error("--predict-only/--replay-only require ledger_full_range")
        if not args.date:
            parser.error(f"--pipeline {args.pipeline} requires --date (or positional date)")
    elif args.pipeline == "ledger_backfill":
        if not args.start or not args.end:
            parser.error("ledger_backfill requires --start and --end")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified electricity forecast entrypoint")
    parser.add_argument(
        "pos_date", nargs="?", default=None,
        help="Target date (YYYY-MM-DD). Shortcut for --date with the default pipeline.",
    )
    parser.add_argument(
        "pos_end", nargs="?", default=None,
        help="Range end date (YYYY-MM-DD). If provided with pos_date, activates range mode.",
    )
    parser.add_argument(
        "--pipeline",
        default="ledger_full",
        choices=[
            "evaluate",
            "sync_dataset",
            # Ledger production pipelines
            "ledger_predict",
            "ledger_backfill",
            "ledger_weight",
            "ledger_fuse",
            "ledger_classifier",
            "ledger_full",
            "ledger_full_range",
            "ledger_smoke",
        ],
    )
    parser.add_argument("--target", default="both", choices=["dayahead", "realtime", "both"])
    parser.add_argument("--models", default="all", help="Comma-separated model names or all")
    parser.add_argument("--date", default=None, help="Single target day, YYYY-MM-DD")
    parser.add_argument("--start", default=None, help="Range start, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Range end, YYYY-MM-DD")
    from utils.data_layout import data_path
    parser.add_argument("--data-path", default=str(data_path("hourly")))
    parser.add_argument(
        "--actual-data-path", default=None,
        help=(
            "Optional authoritative actual-price table used only to populate "
            "actual ledgers. When omitted, --data-path is used for backward compatibility."
        ),
    )
    parser.add_argument("--max-cpu-workers", type=int, default=2)
    parser.add_argument("--max-gpu-workers", type=int, default=1)
    parser.add_argument(
        "--resource-mode",
        choices=["legacy", "split_process"],
        default="legacy",
        help=(
            "Model resource execution mode. legacy preserves the existing "
            "scheduler; split_process starts independent CPU/GPU child "
            "processes and is currently intended for the 96-point feature_store chain."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--force", action="store_true", default=False, help="Force rerun even if cached outputs exist")

    # Ledger pipeline parameters
    parser.add_argument("--epf-v1-root", default=None, help="[Optional legacy compatibility] External EPF v1.0 root. Not required for normal ledger_full runs; local lightGBM/ and TimesFMBackend/ are used by default.")
    # Hidden compatibility knob: LightGBM/TimesFM are always run through bundled 1.0-compatible adapters.
    # Kept only so old scripts that pass this flag do not break; deployment users should not see or set it.
    parser.add_argument("--epf-v1-mode", default="exact", choices=["exact", "cutoff_safe"], help=argparse.SUPPRESS)
    parser.add_argument("--allow-v2-fallback", action="store_true", default=False, help="Allow LightGBM/TimesFM to fall back to 2.0")
    parser.add_argument("--allow-missing-models", action="store_true", default=False, help="Continue even if some models fail")
    parser.add_argument("--allow-equal-weight-fallback", action="store_true", default=False, help="Use equal weights when no period weights available")
    parser.add_argument("--strict-classifier", action="store_true", default=False, help="Fail ledger_full if classifier fails")
    parser.add_argument(
        "--training-months", type=int, default=12,
        help="Model training window in months; use a small value only for smoke validation.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-profile",
        choices=["legacy", "feature_store", "domain"],
        default="legacy",
        help=(
            "Output chain profile. legacy preserves the existing ledger/runs; "
            "feature_store isolates candidate outputs under "
            "outputs/{24,96}/... (default: legacy compatibility roots)."
        ),
    )
    parser.add_argument("--ledger-root", default=None, help="Override ledger storage root")
    parser.add_argument("--runs-root", default=None, help="Override daily run output root")
    parser.add_argument("--feature-store-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--feature-store-mode",
        choices=["off", "raw", "materialized"],
        default="off",
        help=(
            "FeatureStore input mode. off keeps the legacy source path; raw "
            "materializes/uses the raw.parquet cache; materialized also "
            "builds the resolution-aware base/view manifest before prediction."
        ),
    )
    parser.add_argument("--realtime-cutoff-hour", type=int, default=14, help="Cutoff hour for realtime models on D-1")
    parser.add_argument("--recent-week-boost", dest="recent_week_boost", action="store_true", default=True, help="Enable recent-week boost in day_gate weighting")
    parser.add_argument("--no-recent-week-boost", dest="recent_week_boost", action="store_false", help="Disable recent-week boost")
    parser.add_argument("--recent-week-max-gate", type=float, default=0.85, help="Maximum day_gate with recent-week boost")
    parser.add_argument("--weight-max-lookback-days", type=int, default=90, help="Maximum calendar days to look back when selecting complete realtime training days (default 90)")
    parser.add_argument("--validation-days", type=int, default=30, help="Number of complete historical days used by ledger_weight (default 30; champion_short requires 14)")
    parser.add_argument("--weight-learner", choices=["nnls", "bgew", "smape_reg", "champion_short"], default="nnls",
                        help="Fusion weight learner: nnls (默认, 稀疏非负最小二乘) / bgew (旧算法) / smape_reg (SLSQP软门控) / champion_short (实验：14日冠军门控)")
    parser.add_argument("--weight-granularity", choices=["period", "hour", "point"], default="period",
                        help="Weight learning granularity: period (3段, 默认, 实证最优) / hour (24组) / point (96组). 小时/点粒度因样本稀释降级, 仅实验用")
    parser.add_argument("--weight-prune-threshold", type=float, default=0.05,
                        help="Exclude models whose learned weight is below this threshold per task/period; 0 disables pruning")
    parser.add_argument("--weight-min-active-models", type=int, default=1,
                        help="Safety minimum number of active models after weight pruning")

    # TimeMixer tuning
    parser.add_argument("--timemixer-epochs", type=int, default=80)
    parser.add_argument("--timemixer-patience", type=int, default=15)
    parser.add_argument("--timemixer-batch-size", type=int, default=16)
    parser.add_argument("--timemixer-full-refit", action="store_true", default=True)
    parser.add_argument("--timemixer-seeds", type=int, default=42)

    # --- Data sync parameters ---
    parser.add_argument(
        "--sync-data-before-run",
        action="store_true",
        default=False,
        help="Run sync_dataset before ledger_full / ledger_full_range.",
    )
    parser.add_argument(
        "--sync-source",
        default="auto",
        choices=["auto", "db", "http", "local"],
        help="Data sync source. auto = db first, then http/local fallback.",
    )
    parser.add_argument(
        "--resolution",
        default="hourly",
        choices=["hourly", "15min"],
        help=(
            "Temporal resolution for sync_dataset. "
            "hourly = legacy 24-point canonical dataset (default); "
            "15min = native 96-point mirror from the remote database "
            "(epf_market_data_96 + epf_unit_data_96). Omitting --resolution "
            "retains the existing hourly behavior."
        ),
    )
    parser.add_argument(
        "--sync-mode",
        default="full",
        choices=["full", "incremental"],
        help=(
            "96-point sync mode. full = download the complete available "
            "history; incremental = re-pull recent days (overlap window) and "
            "merge. Only applies when --resolution 15min."
        ),
    )
    parser.add_argument(
        "--sync-overlap-days",
        type=int,
        default=7,
        help="Incremental 96-point sync overlap window in days (default 7).",
    )
    parser.add_argument(
        "--include-extended",
        action="store_true",
        default=False,
        help=(
            "96-point sync: also download optional_extended 96-point tables "
            "(congestion, tie-line). Off by default to keep the core mirror lean."
        ),
    )
    parser.add_argument(
        "--force-sync",
        action="store_true",
        default=False,
        help="Refresh canonical dataset even if local data exists.",
    )
    parser.add_argument(
        "--require-fresh-data",
        action="store_true",
        default=False,
        help="Fail if synced/local dataset is not fresh enough for the requested target date.",
    )
    parser.add_argument(
        "--max-data-lag-hours",
        type=int,
        default=36,
        help="Maximum allowed lag between target decision time and latest available data.",
    )

    # Smoke pipeline params
    parser.add_argument("--smoke-training-months", type=int, default=3)
    parser.add_argument("--smoke-timemixer-epochs", type=int, default=3)
    parser.add_argument("--smoke-timemixer-patience", type=int, default=1)

    # Range pipeline params
    parser.add_argument("--continue-on-error", action="store_true", default=False,
        help="Continue range pipeline even if a single day fails")
    range_mode = parser.add_mutually_exclusive_group()
    range_mode.add_argument(
        "--predict-only", action="store_true", default=False,
        help=(
            "Range mode: run only ledger_predict for each day and build the "
            "prediction/actual ledgers; skip weight, fuse, classifier, and final output stages."
        ),
    )
    range_mode.add_argument(
        "--replay-only", action="store_true", default=False,
        help=(
            "Range mode: reuse existing prediction/actual ledgers and run "
            "weight, fuse, classifier, and final output stages without model prediction."
        ),
    )
    parser.add_argument("--skip-existing-final", action="store_true", default=False,
        help="Skip days with verified submission_ready.csv already present")
    parser.add_argument("--range-preflight", dest="range_preflight", action="store_true", default=True,
        help="Run preflight checks before starting range pipeline")
    parser.add_argument("--no-range-preflight", dest="range_preflight", action="store_false",
        help="Skip preflight checks")
    return parser
