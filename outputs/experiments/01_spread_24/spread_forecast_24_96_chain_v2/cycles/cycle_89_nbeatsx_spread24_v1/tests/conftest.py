from pathlib import Path
import sys

CYCLE = Path(__file__).resolve().parents[1]
ROOT = next(p for p in CYCLE.parents if (p / "utils" / "resolution.py").exists())
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(CYCLE / "src"))
