from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sgdfnet.protocol_b import run_protocol_b_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SGDFNet Protocol B rolling-month experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    run_dir = run_protocol_b_experiment(args.config)
    print(run_dir)


if __name__ == "__main__":
    main()
