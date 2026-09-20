"""CLI contract smoke for the formal 96-point façade."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cli.parser import build_parser, normalize_date_args


def parse(argv: list[str]):
    parser = build_parser()
    args = parser.parse_args(argv)
    normalize_date_args(args, parser)
    return args


def main() -> int:
    cases = {
        ("--96", "2026-09-18"): ("ledger_full", "both", False),
        ("--96", "2026-09-18", "--predict", "dayahead"): ("ledger_predict", "dayahead", False),
        ("--96", "2026-09-18", "--predict", "realtime"): ("ledger_predict", "realtime", False),
        ("--96", "2026-09-18", "--predict", "both"): ("ledger_predict", "both", False),
        ("--96", "2026-09-18", "--finish"): ("ledger_full", "both", True),
    }
    for argv, expected in cases.items():
        args = parse(list(argv))
        got = (args.pipeline, args.target, bool(getattr(args, "replay_only", False)))
        assert got == expected, (argv, got, expected)
        assert args.resolution == "15min"
        assert args.output_profile == "production"
        assert args.resource_mode == "split_process"
        assert args.realtime_cutoff_hour == 15
        assert args.max_cpu_workers == 2 and args.max_gpu_workers == 1
        assert args.weight_learner == "smape_reg"
        assert args.weight_granularity == "period"
        assert args.validation_days == 30 and args.weight_max_lookback_days == 90
        assert args.weight_prune_threshold == 0.05

    # Formal façade may use a dedicated deployment root, but it must never be
    # redirected into known legacy/compatibility state by mistake.
    custom = parse([
        "--96", "2026-09-18",
        "--ledger-root", "D:/formal96_state/ledger",
        "--runs-root", "D:/formal96_state/runs",
    ])
    assert custom.ledger_root == "D:/formal96_state/ledger"
    assert custom.runs_root == "D:/formal96_state/runs"

    rejected = [
        ["--96", "2026-09-18", "--ledger-root", "outputs/ledger_96"],
        ["--96", "2026-09-18", "--runs-root", "outputs/runs_96"],
        ["--96", "2026-09-18", "--feature-store-root", "outputs/cache"],
        ["--96", "2026-09-18", "--feature-store-root", "outputs/96/feature_store/cache"],
    ]
    for argv in rejected:
        try:
            parse(argv)
        except SystemExit as exc:
            assert exc.code == 2, (argv, exc.code)
        else:
            raise AssertionError(f"formal --96 accepted legacy root: {argv}")

    print("check_96_facade_cli: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
