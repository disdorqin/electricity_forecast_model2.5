#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FeatureStore 全模型 + SLSQP 软门控融合 — 端到端实验
==================================================
目标：
  1. 用 FeatureStore raw parquet（提速 226x）跑 96 点全模型预测
  2. 融合学习器用 SLSQP 软门控（--weight-learner smape_reg）
  3. 记录时间与结果输出，评估可跑通性

用法（在项目根）：
  python scripts/experiments/feature_store_ab/run_pipeline_slsqp.py \
      --date 2026-01-01 --runs-root outputs/experiments/04_pipeline_audits/feature_store_ab/runs
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
PY = sys.executable

MAIN = PROJECT / "main.py"
# FeatureStore raw parquet（先由 feature_store.ensure() 生成）
FEATURE_STORE_DIR = PROJECT / "outputs" / "feature_store"


def find_raw_parquet() -> Path:
    for p in sorted(FEATURE_STORE_DIR.rglob("raw.parquet")):
        return p
    raise FileNotFoundError("未找到 raw.parquet，请先运行 FeatureStore.load_raw()")


def run_cmd(label: str, cmd: list[str]) -> float:
    print(f"\n=== {label} ===", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    out = (r.stdout or "") + (r.stderr or "")
    # 打印关键行
    for line in out.splitlines():
        if any(k in line.lower() for k in ["status': 'ok'", "status': 'failed'", "fused", "complete", "error", "elapsed"]):
            if len(line) < 300:
                print("  " + line, flush=True)
    print(f"  [{label}] 耗时 {dt:.1f}s, exit={r.returncode}", flush=True)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-01-01")
    ap.add_argument("--runs-root", default=str(PROJECT / "outputs/experiments/04_pipeline_audits/feature_store_ab/runs"))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--skip-predict", action="store_true")
    args = ap.parse_args()

    raw_pq = find_raw_parquet()
    print(f"FeatureStore raw: {raw_pq}")

    # 先确保 raw parquet 存在（若已存在则跳过）
    from utils.feature_store import FeatureStore
    fs = FeatureStore(resolution="15min", source=PROJECT / "data/shandong_pmos_96_full_v2.xlsx")
    df = fs.load_raw()
    print(f"raw parquet: {len(df)} 行")

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    # 隔离 ledger：复制正式 ledger_96 到实验目录，避免污染生产账本
    exp_ledger = runs_root / "ledger_96"
    src_ledger = PROJECT / "outputs/ledger_96"
    if not exp_ledger.exists():
        import shutil
        shutil.copytree(src_ledger, exp_ledger)
    ledger_root = exp_ledger

    base = [PY, str(MAIN), "--date", args.date, "--resolution", "15min",
            "--runs-root", str(runs_root), "--ledger-root", str(ledger_root),
            "--data-path", str(raw_pq)]

    timings = {}

    if not args.skip_predict:
        m = args.models or ["lightgbm", "timesfm", "sgdfnet", "timemixer", "rt916"]
        timings["predict"] = run_cmd("ledger_predict 全模型", base + ["--pipeline", "ledger_predict", "--models", ",".join(m), "--force"])

    timings["weight"] = run_cmd("ledger_weight (SLSQP 软门控)", base + ["--pipeline", "ledger_weight", "--weight-learner", "smape_reg"])
    timings["fuse"] = run_cmd("ledger_fuse", base + ["--pipeline", "ledger_fuse"])

    print("\n=== 计时汇总 ===")
    for k, v in timings.items():
        print(f"  {k}: {v:.1f}s")

    # 保存计时
    report = {"date": args.date, "timings": timings}
    (runs_root / "timing_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n计时报告: {runs_root / 'timing_report.json'}")


if __name__ == "__main__":
    main()
