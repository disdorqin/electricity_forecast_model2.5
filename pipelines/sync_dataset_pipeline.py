from __future__ import annotations

import logging
from typing import Any

from sync_data import sync_dataset

logger = logging.getLogger(__name__)


def run_sync_dataset_pipeline(args: Any = None) -> dict:
    """Run the sync_dataset pipeline with optional CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace or None
        If given, the following attributes are read:
          - sync_source (str): default "auto"
          - force_sync (bool): default False
          - require_fresh_data (bool): default False
          - date or start (str): target date for freshness check
          - data_path (str): custom data path
          - max_data_lag_hours (int): default 36
          - resolution (str): "hourly" (default) or "15min"
          - sync_mode (str): "full" (default) or "incremental"  [15min only]
          - sync_overlap_days (int): default 7             [15min only]
          - include_extended (bool): default False          [15min only]

    Resolution routing:
      * hourly (default) -> legacy 24-point canonical dataset via sync_data.
      * 15min            -> native 96-point local mirror via sync_data_96_core.

    Returns
    -------
    dict — the sync manifest (hourly) or 96-point manifest (15min).
    """
    if args is None:
        return sync_dataset()

    resolution = getattr(args, "resolution", "hourly")

    # ------------------------------------------------------------------
    # 15-minute (96-point) resolution -> native local mirror
    # ------------------------------------------------------------------
    if resolution == "15min":
        from sync_data_96_core import sync_96
        logger.info("sync_dataset: resolution=15min source=%s mode=%s",
                     getattr(args, "sync_source", "db"),
                     getattr(args, "sync_mode", "full"))
        return sync_96(args)

    # ------------------------------------------------------------------
    # Hourly (default) -> legacy behavior, unchanged
    # ------------------------------------------------------------------
    source = getattr(args, "sync_source", "auto")
    force = getattr(args, "force_sync", False)
    require_fresh = getattr(args, "require_fresh_data", False)
    max_lag = getattr(args, "max_data_lag_hours", 36)

    # Determine target_date — prefer --date, then --start, then None
    target_date = getattr(args, "date", None)
    if target_date is None:
        target_date = getattr(args, "start", None)

    data_path = getattr(args, "data_path", None)

    logger.info(
        "sync_dataset: source=%s force=%s require_fresh=%s target=%s",
        source, force, require_fresh, target_date,
    )

    return sync_dataset(
        data_path=data_path,
        source=source,
        force=force,
        require_fresh=require_fresh,
        target_date=target_date,
        max_data_lag_hours=max_lag,
    )
