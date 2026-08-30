from nbeatsx_spread.audits.holdout import audit_holdout_registry


def test_holdout_registry_rejects_overlap():
    registry = {"status": "ACTIVE", "lockboxes": [{"start_date": "2026-09-01", "end_date": "2026-09-30"}]}
    assert audit_holdout_registry(["2026-08-31"], registry).passed
    assert not audit_holdout_registry(["2026-09-01"], registry).passed
