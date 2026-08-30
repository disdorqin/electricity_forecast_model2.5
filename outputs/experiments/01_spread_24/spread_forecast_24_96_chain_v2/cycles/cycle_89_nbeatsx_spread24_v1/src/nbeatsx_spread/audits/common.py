from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AuditResult:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "status": "PASS" if self.passed else "FAIL", "detail": self.detail}


def result(name: str, passed: bool, detail: str) -> AuditResult:
    return AuditResult(name, bool(passed), detail)
