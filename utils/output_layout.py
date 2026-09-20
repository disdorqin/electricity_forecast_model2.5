"""Canonical output roots for production and compatibility chains.

96-point production is domain-scoped under ``outputs/96``. The validated
24-point chain temporarily keeps its established ``outputs/ledger`` +
``outputs/runs`` state until a separate migration is explicitly verified.
Legacy and FeatureStore profiles remain compatibility/history choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputLayout:
    """Output roots belonging to one chain profile and one resolution."""

    profile: str
    resolution: str
    ledger_root: Path
    runs_root: Path
    feature_store_root: Path


def resolve_output_layout(profile: str = "production", resolution: str = "hourly") -> OutputLayout:
    """Return isolated output roots for ``profile`` and ``resolution``.

    ``production`` writes 96-point state to the bounded domain tree while
    retaining the validated 24-point state roots until their dedicated
    migration. ``legacy`` preserves historical roots. ``feature_store`` and
    ``domain`` remain explicit compatibility profiles for old experiments.
    """
    if profile not in {"production", "legacy", "feature_store", "domain"}:
        raise ValueError(f"Unknown output profile: {profile!r}")
    if resolution not in {"hourly", "15min"}:
        raise ValueError(f"Unknown resolution: {resolution!r}")

    is_96 = resolution == "15min"
    if profile == "production":
        if is_96:
            root = Path("outputs") / "96"
            return OutputLayout(
                profile=profile,
                resolution=resolution,
                ledger_root=root / "ledger",
                runs_root=root / "runs",
                feature_store_root=root / "cache",
            )
        return OutputLayout(
            profile=profile,
            resolution=resolution,
            ledger_root=Path("outputs/ledger"),
            runs_root=Path("outputs/runs"),
            feature_store_root=Path("outputs/24/cache"),
        )

    if profile == "legacy":
        return OutputLayout(
            profile=profile,
            resolution=resolution,
            ledger_root=Path("outputs/ledger_96" if is_96 else "outputs/ledger"),
            runs_root=Path("outputs/runs_96" if is_96 else "outputs/runs"),
            feature_store_root=Path("outputs/feature_store"),
        )

    domain = "96" if is_96 else "24"
    chain = "feature_store" if profile == "feature_store" else "legacy"
    root = Path("outputs") / domain / chain
    return OutputLayout(
        profile=profile,
        resolution=resolution,
        ledger_root=root / "ledger",
        runs_root=root / "runs",
        feature_store_root=root / "cache",
    )


def apply_output_layout(args) -> OutputLayout:
    """Resolve CLI output roots once and attach them to ``args``.

    Explicit ``--ledger-root``/``--runs-root`` values always win.  The parser
    uses ``None`` defaults so selecting a profile does not overwrite a custom
    deployment directory.
    """
    profile = getattr(args, "output_profile", "production")
    resolution = getattr(args, "resolution", "hourly")
    layout = resolve_output_layout(profile, resolution)

    if getattr(args, "ledger_root", None) is None:
        args.ledger_root = str(layout.ledger_root)
    if getattr(args, "runs_root", None) is None:
        args.runs_root = str(layout.runs_root)
    if getattr(args, "feature_store_root", None) is None:
        args.feature_store_root = str(layout.feature_store_root)
    return layout
