from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "utils" / "resolution.py").exists(): return parent
    raise RuntimeError("project root not found")


ROOT = project_root(); CYCLE = ROOT / "outputs/experiments/01_spread_24/spread_forecast_24_96_chain_v2/cycles/cycle_89_nbeatsx_spread24_v1"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(CYCLE / "src"))


def main() -> int:
    path = ROOT / "data/24/canonical/shandong_pmos_hourly.csv"
    checks = {"canonical_exists": path.exists()}
    if path.exists():
        from nbeatsx_spread.data.canonical_source import CanonicalHourlySource
        from nbeatsx_spread.data.business_dataset import complete_business_days
        source = CanonicalHourlySource.from_csv(path)
        checks.update({"rows_nonzero": len(source.frame) > 0, "has_prices": {"日前电价", "实时电价"}.issubset(source.frame.columns), "complete_days": len(complete_business_days(source)) > 300})
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    ok = all(checks.values())
    print("PRECHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__": sys.exit(main())
