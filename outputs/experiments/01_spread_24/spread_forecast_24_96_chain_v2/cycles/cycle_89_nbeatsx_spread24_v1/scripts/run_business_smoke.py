from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve(); CYCLE = HERE.parents[1]; ROOT = next(p for p in HERE.parents if (p / "utils" / "resolution.py").exists())
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(CYCLE / "src"))
from nbeatsx_spread.audits import audit_covariate_availability, audit_holdout_registry, audit_horizon, audit_origin, audit_training_cutoff, load_holdout_registry, run_counterfactual_audit
from nbeatsx_spread.contracts import latest_complete_label_day
from nbeatsx_spread.data.business_dataset import build_business_split
from nbeatsx_spread.data.canonical_source import CanonicalHourlySource
from nbeatsx_spread.data.origin_index import build_origin_window
from nbeatsx_spread.evaluation.metrics import compute_metrics, metric_by_forecast_offset
from nbeatsx_spread.losses.paper_mae import PaperMAE
from nbeatsx_spread.model.factory import build_model
from nbeatsx_spread.training.trainer import Trainer, TrainingConfig
from nbeatsx_spread.training.reproducibility import seed_everything


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--target-day", action="append", default=None); ap.add_argument("--data", type=Path, default=ROOT / "data/24/canonical/shandong_pmos_hourly.csv"); ap.add_argument("--run-dir", type=Path, default=CYCLE / "runs/smoke_h34_mae"); ap.add_argument("--steps", type=int, default=3); args = ap.parse_args()
    target_days = (args.target_day or ["2026-06-01"])[:3]
    source = CanonicalHourlySource.from_csv(args.data); config = json.loads((CYCLE / "configs/business_strict34_core.json").read_text(encoding="utf-8")); registry = load_holdout_registry(CYCLE / "configs/holdout_registry.json"); all_results = []
    for day in target_days:
        w = build_origin_window(day); audits = [audit_origin(day, w), audit_horizon(w), audit_covariate_availability(source, day), *run_counterfactual_audit(source, day), audit_holdout_registry([day], registry)]
        train, val, split = build_business_split(source, day)
        audits.append(audit_training_cutoff(day, split["train_days"] + split["validation_days"]))
        if not all(a.passed for a in audits): raise RuntimeError("INVALID-LEAKAGE; training is blocked")
        run_dir = args.run_dir / day; seed_everything(42); model = build_model(config); trainer = Trainer(model, train, val, PaperMAE(), run_dir, TrainingConfig(max_steps=args.steps, min_steps=1, eval_every=1, patience_checks=2, batch_size=32, seed=42), device="cpu"); train_info = trainer.fit()
        model.eval(); preds=[]; targets=[]; masks=[]
        with torch.no_grad():
            for i in range(len(val)):
                b=val[i]; p=model(b["y_backcast"].unsqueeze(0),b["x_backcast"].unsqueeze(0),b["x_future"].unsqueeze(0)).squeeze(0).numpy()*split["target_scale"]["scale"]; preds.append(p); targets.append(b["y_future"].numpy()*split["target_scale"]["scale"]); masks.append(b["score_mask"].numpy())
        pred=np.stack(preds); target=np.stack(targets); mask=np.stack(masks); headline=compute_metrics(pred[:,10:],target[:,10:],mask[:,10:]); bridge=compute_metrics(pred[:,:10],target[:,:10]);
        day_dir=run_dir; (day_dir/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8"); (day_dir/"environment.json").write_text(json.dumps({"python":sys.version,"torch":torch.__version__,"device":"cpu","seed":42,"resolution":"hourly","forecast_origin":"D-1 14:00"},ensure_ascii=False,indent=2),encoding="utf-8"); (day_dir/"leakage_audit.json").write_text(json.dumps({"leakage_status":"STRICT/PASS","audits":[a.as_dict() for a in audits]},ensure_ascii=False,indent=2),encoding="utf-8"); (day_dir/"split_manifest.json").write_text(json.dumps(split,ensure_ascii=False,indent=2,default=str),encoding="utf-8"); (day_dir/"headline_metrics.json").write_text(json.dumps(headline,ensure_ascii=False,indent=2),encoding="utf-8"); (day_dir/"bridge_metrics.json").write_text(json.dumps(bridge,ensure_ascii=False,indent=2),encoding="utf-8");
        with (day_dir/"predictions.csv").open("w",newline="",encoding="utf-8") as f: csv.writer(f).writerows([["sample","offset","prediction","target"]]+[[i,j+1,float(pred[i,j]),float(target[i,j])] for i in range(len(pred)) for j in range(pred.shape[1])])
        with (day_dir/"metric_by_forecast_offset.csv").open("w",newline="",encoding="utf-8") as f: writer=csv.DictWriter(f,fieldnames=list(metric_by_forecast_offset(pred,target)[0])); writer.writeheader(); writer.writerows(metric_by_forecast_offset(pred,target))
        manifest={"target_day":day,"forecast_origin":"D-1 14:00","training_last_day":latest_complete_label_day(day),"target_day_actual_as_feature":False,"target_day_DA_as_feature":False,"d1_post14_spread_as_feature":False,"final_holdout_touched":False,"leakage_status":"STRICT/PASS","training":train_info}; (day_dir/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"); all_results.append(manifest)
    print(json.dumps({"status":"SMOKE_PASS","runs":all_results},ensure_ascii=False,indent=2)); return 0


if __name__ == "__main__": sys.exit(main())
