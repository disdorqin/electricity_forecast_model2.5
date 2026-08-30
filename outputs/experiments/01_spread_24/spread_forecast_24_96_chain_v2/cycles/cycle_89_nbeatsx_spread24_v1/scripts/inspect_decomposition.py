from __future__ import annotations

import argparse
import torch


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("checkpoint"); args=ap.parse_args(); payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False); print({"step":payload.get("step"),"metadata":payload.get("metadata")}); return 0


if __name__ == "__main__": raise SystemExit(main())
