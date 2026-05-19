"""Shared return type for every diagnostic check.

The review UI renders each diagnostic identically — a colored pill from
`status`, the one-line `message`, and the `evidence` dict expanded for
detail-on-demand. Status semantics:

  - "pass" — the check found no evidence of the failure mode.
  - "warn" — borderline; surface in the UI but don't block a ship decision.
  - "fail" — strong evidence of the failure mode; ship-blocking by default.

Thresholds are method-specific and exposed as parameters on each check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DiagnosticStatus = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class DiagnosticResult:
    name: str
    status: DiagnosticStatus
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
