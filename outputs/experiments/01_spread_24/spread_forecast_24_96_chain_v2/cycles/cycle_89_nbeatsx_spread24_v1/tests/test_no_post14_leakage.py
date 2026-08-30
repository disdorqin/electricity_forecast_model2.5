from pathlib import Path

from nbeatsx_spread.audits.holdout import audit_holdout
from nbeatsx_spread.audits.training_cutoff import audit_training_cutoff


def test_training_cutoff_rejects_d_minus_one():
    assert not audit_training_cutoff("2026-08-27", ["2026-08-26"]).passed


def test_holdout_default_is_untouched():
    assert audit_holdout(False).passed
