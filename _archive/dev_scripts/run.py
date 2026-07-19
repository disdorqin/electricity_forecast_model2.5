from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rt916_spikefusionnet.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
