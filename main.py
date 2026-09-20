from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from cli.parser import build_parser, normalize_date_args
from pipelines.evaluate_pipeline import run_evaluate_pipeline
from pipelines.sync_dataset_pipeline import run_sync_dataset_pipeline

# Ledger production pipelines
from pipelines.ledger_predict import run_ledger_predict
from pipelines.ledger_backfill import run_ledger_backfill
from pipelines.ledger_weight import run_ledger_weight
from pipelines.ledger_fuse import run_ledger_fuse
from pipelines.ledger_classifier import run_ledger_classifier
from pipelines.ledger_full import run_ledger_full
from pipelines.ledger_full_range import run_ledger_full_range
from pipelines.ledger_smoke import run_ledger_smoke


def _delivery_exit_code(delivery_status: str, default: int = 1) -> int:
    """Map delivery status to exit code.

    NORMAL           -> 0
    DEGRADED_DELIVERED -> 2
    FAILED_NO_DELIVERY -> 1 (also default)
    """
    if delivery_status == "NORMAL":
        return 0
    elif delivery_status == "DEGRADED_DELIVERED":
        return 2
    elif delivery_status == "FAILED_NO_DELIVERY":
        return 1
    return default


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Normalize date arguments (handles positional <-> --date/--start/--end mapping)
    normalize_date_args(args, parser)

    if getattr(args, "facade_96", False):
        # Production façade owns these values; ambient shell overrides cannot
        # silently switch the 96 chain back to legacy scheduling/configuration.
        args.resolution = "15min"
        args.output_profile = "production"
        args.resource_mode = "split_process"
        args.max_cpu_workers = 2
        args.max_gpu_workers = 1
        args.weight_learner = "smape_reg"
        args.weight_granularity = "period"
        args.validation_days = 30
        args.weight_max_lookback_days = 90
        args.weight_prune_threshold = 0.05
        args.rt916_train_steps = 24
        args.realtime_cutoff_hour = 15
        # Dynamic-v1 owns the serving input via SnapshotBuilder +
        # FeatureViewBuilder.  Never let an explicit/ambient FeatureStore
        # mode replace that view on the formal façade; FeatureStore remains a
        # legacy/shadow compatibility path only.
        args.feature_store_mode = "off"
        args.production_mode = True

    # ``argparse`` cannot make a default depend on --resolution.  Only replace
    # the parser's hourly default; an explicitly supplied --data-path always
    # wins.  This prevents a 96-point run from silently reading the 24-point
    # source table.
    from utils.data_layout import data_path
    if args.data_path == str(data_path("hourly")) and args.resolution == "15min":
        args.data_path = str(data_path("15min"))
    if args.resolution == "15min" and getattr(args, "actual_data_path", None) is None:
        args.actual_data_path = str(data_path("15min", "authoritative"))

    # Resolve output roots before any pipeline runs. Production is the default
    # domain-scoped profile; legacy/FeatureStore remain explicit compatibility
    # modes so new state cannot silently grow under old roots.
    from utils.output_layout import apply_output_layout

    output_layout = apply_output_layout(args)
    logging.getLogger(__name__).info(
        "Output profile=%s resolution=%s ledger=%s runs=%s feature_store=%s",
        output_layout.profile,
        output_layout.resolution,
        args.ledger_root,
        args.runs_root,
        args.feature_store_root,
    )

    # Global reproducibility: seed must be set before any model code runs
    from utils.reproducibility import set_global_seed

    set_global_seed(args.seed, args.deterministic)

    # --- Formal96 Dynamic-v1 sync gate ---------------------------------
    # A production --96 full/predict invocation must refresh the DB mirror
    # before SnapshotBuilder runs.  ``--finish`` is the sole exception: it
    # reuses the Stage1 snapshot/provenance and never syncs or re-snapshots.
    formal96_facade = bool(getattr(args, "facade_96", False))
    formal96_finish = formal96_facade and bool(getattr(args, "replay_only", False))
    # Formal96 keeps one source-of-truth rule for full/predict runs:
    # refresh the authoritative DB mirror first, then let the snapshot router
    # decide LIVE vs stored-LIVE replay vs historical proxy from data facts.
    # --finish is the sole exception because it must replay the exact Stage1
    # provenance without mutating data or creating a newer snapshot.
    formal96_sync = formal96_facade and not formal96_finish
    # --finish is provenance replay: even an explicitly supplied legacy
    # --sync-data-before-run flag must not mutate data or create a newer
    # snapshot before weight/fuse/final reuse Stage1 predictions.
    sync_before = False if formal96_finish else (
        bool(getattr(args, "sync_data_before_run", False)) or formal96_sync
    )
    if sync_before and args.pipeline in ("ledger_full", "ledger_full_range", "ledger_predict"):
        if formal96_sync:
            # No local/stale fallback is permitted for the formal façade.
            args.sync_source = "db"
            args.force_sync = True
        try:
            sync_result = run_sync_dataset_pipeline(args)
        except Exception as exc:
            if formal96_sync:
                print(
                    f"DATABASE_SYNC_FAILED source=epf_pmos_96_full detail={exc} models_started=false",
                    flush=True,
                )
                return 1
            raise
        status = sync_result.get("status", "failed")
        if status != "ok":
            sync_errors = sync_result.get("errors", ["sync_dataset failed"])
            if formal96_sync:
                print(
                    "DATABASE_SYNC_FAILED source=epf_pmos_96_full "
                    f"detail={'; '.join(sync_errors)} models_started=false",
                    flush=True,
                )
            else:
                print(f"ERROR: --sync-data-before-run: sync_dataset failed: {'; '.join(sync_errors)}", flush=True)
            return 1
        # Point downstream pipelines at the freshly materialized model store.
        if args.resolution == "15min":
            synced_model = (
                sync_result.get("model_inputs", {}).get("full_parquet")
                or sync_result.get("paths", {}).get("model_input_full_parquet")
            )
            if not synced_model:
                print("ERROR: 96 sync completed without model_input_full_parquet", flush=True)
                return 1
            latest_closed_day = sync_result.get("latest_closed_day")
            if formal96_sync and not latest_closed_day:
                print(
                    "DATABASE_SYNC_FAILED source=epf_pmos_96_full "
                    "detail=latest_closed_day missing models_started=false",
                    flush=True,
                )
                return 1
            args._formal96_latest_closed_day = latest_closed_day
            args.data_path = synced_model
            args.actual_data_path = sync_result.get("paths", {}).get(
                "authoritative_csv", args.actual_data_path
            )
        else:
            synced_xlsx = sync_result.get("output_xlsx")
            if synced_xlsx:
                args.data_path = synced_xlsx
        print(
            f"sync_dataset: OK (source={sync_result.get('source_table', sync_result.get('source', '?'))}, "
            f"rows={sync_result.get('authoritative_rows', sync_result.get('rows', 0))})",
            flush=True,
        )

    if args.pipeline == "evaluate":
        output_path = run_evaluate_pipeline(args)
        print(output_path)
        return 0
    if args.pipeline == "sync_dataset":
        output_path = run_sync_dataset_pipeline(args)
        status = output_path.get("status", "failed") if isinstance(output_path, dict) else "ok"
        print(json.dumps(output_path, indent=2, ensure_ascii=False) if isinstance(output_path, dict) else output_path)
        return 0 if status == "ok" or status == "skipped" else 1
    # --- Ledger production pipelines ---
    if args.pipeline == "ledger_predict":
        result = run_ledger_predict(args)
        transient = getattr(args, "_transient_asof_path", None)
        if transient:
            from utils.asof_view_96 import cleanup_transient_asof_96
            cleanup_transient_asof_96(transient)
            result["runtime_input_cleanup"] = {
                "path": str(transient),
                "persistent": False,
            }
        print(f"ledger_predict complete: {result}")
        return 0 if result.get("status") in {"complete", "complete_with_warnings"} else 1
    if args.pipeline == "ledger_backfill":
        result = run_ledger_backfill(args)
        print(f"ledger_backfill complete: {result}")
        return 0
    if args.pipeline == "ledger_weight":
        result = run_ledger_weight(args)
        print(f"ledger_weight complete: {result}")
        return 0
    if args.pipeline == "ledger_fuse":
        result = run_ledger_fuse(args)
        print(f"ledger_fuse complete: {result}")
        return 0
    if args.pipeline == "ledger_classifier":
        result = run_ledger_classifier(args)
        transient = getattr(args, "_transient_asof_path", None)
        if transient:
            from utils.asof_view_96 import cleanup_transient_asof_96
            cleanup_transient_asof_96(transient)
            result["runtime_input_cleanup"] = {
                "path": str(transient),
                "persistent": False,
            }
        print(f"ledger_classifier complete: {result}")
        return 0
    if args.pipeline == "ledger_full":
        result = run_ledger_full(args)
        ds = result.get("delivery_status", "UNKNOWN")
        exit_code = _delivery_exit_code(ds, default=1)
        print(f"ledger_full complete: delivery_status={ds}, exit_code={exit_code}")
        return exit_code
    if args.pipeline == "ledger_full_range":
        result = run_ledger_full_range(args)
        ds = result.get("delivery_status", "UNKNOWN")
        # Range logic: NORMAL/complete -> 0, DEGRADED -> 2, else -> 1
        range_status = result.get("status", "")
        if ds in ("NORMAL", "PREDICTIONS_READY"):
            exit_code = 0
        elif ds == "DEGRADED_DELIVERED":
            exit_code = 2
        elif range_status in ("complete", "all_skipped") and ds != "FAILED_NO_DELIVERY":
            # complete/all_skipped without degraded delivery is normal
            exit_code = 0
        else:
            # partial / failed / preflight_failed / interrupted / FAILED_NO_DELIVERY
            exit_code = 1
        print(f"ledger_full_range complete: status={range_status}, "
              f"delivery_status={ds}, exit_code={exit_code}")
        return exit_code
    if args.pipeline == "ledger_smoke":
        result = run_ledger_smoke(args)
        print(f"ledger_smoke complete: {result}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
