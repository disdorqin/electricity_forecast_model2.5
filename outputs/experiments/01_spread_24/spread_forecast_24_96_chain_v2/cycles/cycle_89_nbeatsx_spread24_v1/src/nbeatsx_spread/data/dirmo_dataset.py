"""Strict daily-origin datasets for the frozen C3 DIRMO block partition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .business_dataset import build_business_split
from .canonical_source import CanonicalHourlySource
from .normalization import RobustArrayScaler, fit_target_scale
from .strategy_dataset import StrategyDataset, StrategyView


@dataclass(frozen=True)
class DirmoBlock:
    """One immutable direct-output block in H34 coordinates."""

    block_id: str
    start: int
    stop: int
    business_scope: str
    headline: bool

    @property
    def horizon(self) -> int:
        """Return the number of outputs in this block."""
        return self.stop - self.start


DIRMO_BLOCKS: tuple[DirmoBlock, ...] = (
    DirmoBlock("B0_BRIDGE10", 0, 10, "D-1 h15..h24", False),
    DirmoBlock("B1_DDAY_FIRST12", 10, 22, "D h1..h12", True),
    DirmoBlock("B2_DDAY_LAST12", 22, 34, "D h13..h24", True),
)


def get_dirmo_block(block_id: str) -> DirmoBlock:
    """Return a frozen C3 block or fail closed for an unknown block."""
    for block in DIRMO_BLOCKS:
        if block.block_id == block_id:
            return block
    raise ValueError(f"unknown C3 DIRMO block: {block_id}")


def validate_dirmo_partition() -> None:
    """Assert that C3 covers H34 exactly once without feedback."""
    spans = [(block.start, block.stop) for block in DIRMO_BLOCKS]
    if spans != [(0, 10), (10, 22), (22, 34)]:
        raise AssertionError(f"C3 partition changed: {spans}")
    if sum(block.horizon for block in DIRMO_BLOCKS) != 34:
        raise AssertionError("C3 partition does not cover H34")


def _views(samples: list[Any], block: DirmoBlock) -> list[StrategyView]:
    """Project legal H34 samples into one independent block view."""
    return [StrategyView(sample, slice(block.start, block.stop)) for sample in samples]


def build_dirmo_split(
    source: CanonicalHourlySource,
    target_day: str,
    block_id: str,
    *,
    validation_days: int = 28,
    training_months: int = 36,
) -> tuple[StrategyDataset, StrategyDataset, dict[str, Any]]:
    """Build one C3 block using train-only scaling and strict D-2 labels.

    The returned datasets all retain the same daily origins as the frozen C0
    chassis.  No block prediction is included in either dataset, so this API
    cannot silently implement recursive feedback.
    """
    validate_dirmo_partition()
    block = get_dirmo_block(block_id)
    base_train, base_val, base_manifest = build_business_split(
        source,
        target_day,
        validation_days=validation_days,
        training_months=training_months,
        feature_profile="CORE5_RAW",
    )
    train_views = _views(base_train.samples, block)
    val_views = _views(base_val.samples, block)
    target_scale = fit_target_scale([view.y_future for view in train_views])
    x_scaler = RobustArrayScaler.fit(
        [view.sample.x_backcast for view in train_views]
        + [view.x_future for view in train_views],
        numeric_channels=tuple(range(5)),
    )
    train = StrategyDataset(train_views, target_scale, x_scaler)
    val = StrategyDataset(val_views, target_scale, x_scaler)
    manifest = dict(base_manifest)
    manifest.update(
        {
            "strategy": "C3_DIRMO_10_12_12",
            "block_id": block.block_id,
            "block_h34_offsets": {"start": block.start + 1, "stop": block.stop},
            "business_scope": block.business_scope,
            "headline": block.headline,
            "recursive_feedback": False,
            "target_slice": {"start": block.start, "stop": block.stop},
            "target_scale": {"scale": target_scale.scale, "floor": target_scale.floor},
            "x_scaler": x_scaler.to_dict(),
            "input_feature_count": train.n_features,
            "horizon": train.horizon,
            "label_source": "historical_samples_only; target-day labels excluded from builder",
        }
    )
    return train, val, manifest
