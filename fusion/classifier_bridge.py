from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def classifier_data_covers_range(clf_data_path: Path, start_date: str, end_date: str) -> tuple[bool, str]:
    if not clf_data_path.exists():
        return False, f"classifier data file not found: {clf_data_path}"
    try:
        df = pd.read_excel(clf_data_path, usecols=["时刻"], engine="openpyxl")
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
) -> Path:
    script_path = project_root / "ExtremPriceClf" / "merge_model_scripts" / "run_daily.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Classifier script not found: {script_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Always pass absolute paths — run_daily.py resolves relative paths
    # against its own project_root (ExtremPriceClf/), not our cwd.
    abs_output_dir = output_dir.resolve()
    abs_data_path = clf_data_path.resolve()
    cmd = [
        sys.executable,
        str(script_path),
        start_date,
        end_date,
        "--output",
        str(abs_output_dir),
        "--data",
        str(abs_data_path),
        "--resolution",
        resolution,
    ]
    try:
        subprocess.run(cmd, check=True, cwd=script_path.parent.parent, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr_snippet = (exc.stderr or "")[:500]
        raise RuntimeError(
            f"Extreme price classifier failed (exit code {exc.returncode}). "
            f"Command: {' '.join(cmd)}. "
            f"Stderr: {stderr_snippet}"
        ) from exc
    result_path = output_dir / f"{start_date}_{end_date}_clf.xlsx"
    if not result_path.exists():
        raise FileNotFoundError(f"Classifier result not found: {result_path}")
    return result_path


def merge_clf_results(fused_csv_path: Path, clf_result_path: Path, output_path: Path) -> pd.DataFrame:
    fused = pd.read_csv(fused_csv_path)
    clf = pd.read_excel(clf_result_path, engine="openpyxl")
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
    )
    corrected = fusion_work_dir / "realtime" / "fused_predictions_corrected.csv"
    merged = merge_clf_results(rt_fused, clf_result, corrected)
    return {
        "status": "completed",
        "corrected_hours": int((merged["final_pred"] == 1).sum()),
        "output_path": str(corrected),
        "clf_result_path": str(clf_result),
    }
