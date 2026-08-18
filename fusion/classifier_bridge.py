from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def classifier_data_covers_range(clf_data_path: Path, start_date: str, end_date: str) -> tuple[bool, str]:
    if not clf_data_path.exists():
        return False, f"classifier data file not found: {clf_data_path}"
    try:
        from utils.data_loader import load_table

        df = load_table(clf_data_path)
        if "时刻" in df.columns:
            df = df[["时刻"]]
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to read classifier data range: {exc}"
    if "时刻" not in df.columns:
        return False, "classifier data missing 时刻 column"
    ts = pd.to_datetime(df["时刻"], errors="coerce").dropna()
    if ts.empty:
        return False, "classifier data has no valid 时刻 values"
    required_start = pd.Timestamp(start_date) + pd.Timedelta(hours=1)
    required_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    data_start = ts.min()
    data_end = ts.max()
    if data_start <= required_start and data_end >= required_end:
        return True, ""
    return (
        False,
        f"classifier data covers {data_start} ~ {data_end}, required {required_start} ~ {required_end}",
    )


def run_extreme_price_classifier(
    *,
    project_root: Path,
    start_date: str,
    end_date: str,
    clf_data_path: Path,
    output_dir: Path,
    resolution: str = "hourly",
    feature_store_root: Path | None = None,
) -> Path:
    """Run the production classifier through the reusable range runner.

    The old subprocess entry point remains available as a compatibility tool,
    but production must use the cache-aware implementation so the feature
    engineering and historical p1 warm-up are materialized once per
    resolution/task/source namespace.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    from ExtremPriceClf.merge_model.core.range_runner import (
        ClassifierRangeSpec,
        run_classifier_range,
    )

    spec = ClassifierRangeSpec(
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
        task="realtime",
    )
    run_result = run_classifier_range(
        project_root=project_root,
        source=clf_data_path.resolve(),
        spec=spec,
        output_dir=output_dir.resolve(),
        feature_store_root=feature_store_root.resolve() if feature_store_root else None,
        reuse_cache=True,
    )
    parquet_result = Path(run_result["result_path"])
    if not parquet_result.exists():
        raise FileNotFoundError(f"Classifier ledger not found: {parquet_result}")

    # Keep the canonical classifier output in parquet.  XLSX remains an
    # optional external compatibility export, never part of the hot path.
    result_path = output_dir / f"{start_date}_{end_date}_clf.parquet"
    if parquet_result.resolve() != result_path.resolve():
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_tmp = result_path.with_suffix(".parquet.tmp")
        result_tmp.write_bytes(parquet_result.read_bytes())
        result_tmp.replace(result_path)
    if not result_path.exists():
        raise FileNotFoundError(f"Classifier result not found: {result_path}")
    return result_path


def merge_clf_results(fused_csv_path: Path, clf_result_path: Path, output_path: Path) -> pd.DataFrame:
    fused = pd.read_csv(fused_csv_path)
    from utils.data_loader import load_table
    clf = load_table(clf_result_path)
    clf = clf.rename(columns={"时刻": "ds"})
    if "final_pred" not in clf.columns:
        raise ValueError(
            f"Classifier result {clf_result_path} is missing 'final_pred' column. "
            f"Available columns: {list(clf.columns)}"
        )
    clf["ds"] = pd.to_datetime(clf["ds"], errors="coerce")
    fused["ds"] = pd.to_datetime(fused["ds"], errors="coerce")

    # 分类器是小时级模型（输出小时粒度 时刻/final_pred），
    # fused 可能是 96 点（15min）或 24 点。做统一对齐：
    #   - 96 点：把 fused ds 归到所属业务小时 → 与分类器小时 final_pred join → 广播到该小时 4 个刻度
    #   - 24 点：直接按 ds 精确 join
    is_96 = "business_period" in fused.columns or (len(fused) > 24)
    if is_96 and len(fused) != len(clf):
        clf_hour = clf["ds"].dt.floor("h")
        clf_map = dict(zip(clf_hour, clf["final_pred"]))
        fused["_hour"] = fused["ds"].dt.floor("h")
        merged = fused.copy()
        merged["final_pred"] = merged["_hour"].map(clf_map)
        merged = merged.drop(columns=["_hour"])
    else:
        merged = fused.merge(clf[["ds", "final_pred"]], on="ds", how="left")

    merged["y_fused_corrected"] = merged["y_fused"]
    mask = (merged["final_pred"] == 1) & (merged["y_fused"] <= 100)
    merged.loc[mask, "y_fused_corrected"] = -80.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return merged


def run_classifier_pipeline(
    *,
    fusion_work_dir: Path,
    project_root: Path,
    start_date: str,
    end_date: str,
    clf_data_path: Path,
    feature_store_root: Path | None = None,
) -> dict:
    rt_fused = fusion_work_dir / "realtime" / "fused_predictions.csv"
    if not rt_fused.exists():
        return {"status": "skipped", "reason": "missing_rt_fused"}
    covered, reason = classifier_data_covers_range(clf_data_path, start_date, end_date)
    if not covered:
        return {"status": "skipped", "reason": reason, "clf_data_path": str(clf_data_path)}
    clf_dir = fusion_work_dir / "classifier"
    clf_dir.mkdir(parents=True, exist_ok=True)
    # 自动检测 fused 分辨率：96 点（15min）→ 分类器入口按小时聚合；24 点（hourly）→ 原样。
    _probe = pd.read_csv(rt_fused)
    _is_96 = "business_period" in _probe.columns or len(_probe) > 24
    resolution = "15min" if _is_96 else "hourly"
    clf_result = run_extreme_price_classifier(
        project_root=project_root,
        start_date=start_date,
        end_date=end_date,
        clf_data_path=clf_data_path,
        output_dir=clf_dir,
        resolution=resolution,
        feature_store_root=feature_store_root,
    )
    corrected = fusion_work_dir / "realtime" / "fused_predictions_corrected.csv"
    merged = merge_clf_results(rt_fused, clf_result, corrected)
    return {
        "status": "completed",
        "method": "range_runner_feature_cache",
        "resolution": resolution,
        "corrected_hours": int((merged["final_pred"] == 1).sum()),
        "output_path": str(corrected),
        "clf_result_path": str(clf_result),
    }
