from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve(); CYCLE = HERE.parents[1]
sys.path.insert(0, str(CYCLE / "src"))
from nbeatsx_spread.evaluation.decomposition import assert_decomposition_sum
from nbeatsx_spread.model.nbeatsx import NBEATSx
from nbeatsx_spread.training.reproducibility import seed_everything


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",type=Path,default=CYCLE/"runs/paper_repro/tiny"); args=ap.parse_args(); seed_everything(42)
    model=NBEATSx(168,24,1,("identity","exogenous_tcn"),(1,1),256,2,8,3,"softplus",0.0,False,"orthogonal"); y=torch.randn(2,168); xb=torch.randn(2,168,1); xf=torch.randn(2,24,1); out=model(y,xb,xf,True); assert out.forecast.shape==(2,24); assert_decomposition_sum(out.initial_level,out.block_forecasts,out.forecast)
    args.run_dir.mkdir(parents=True,exist_ok=True); (args.run_dir/"reproduction_delta.json").write_text(json.dumps({"status":"P1_STRUCTURAL_PASS","shape":list(out.forecast.shape),"decomposition_sum":True,"paper_profile":{"input_size":168,"horizon":24,"loss":"MAE","business_adaptation":False}},indent=2),encoding="utf-8"); print("PAPER_REPRO_TINY PASS"); return 0


if __name__ == "__main__": sys.exit(main())
