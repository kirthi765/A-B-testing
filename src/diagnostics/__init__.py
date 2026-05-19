"""Diagnostics — every check returns a `DiagnosticResult(status, message, evidence)`."""

from ._result import DiagnosticResult, DiagnosticStatus
from .aa_test import check_aa
from .guardrails import GuardrailConfig, evaluate_guardrail
from .multiple_comparisons import FDRResult, benjamini_hochberg
from .novelty import check_novelty
from .simpsons import check_simpsons
from .srm import check_srm

__all__ = [
    "DiagnosticResult",
    "DiagnosticStatus",
    "check_srm",
    "check_aa",
    "check_novelty",
    "check_simpsons",
    "benjamini_hochberg",
    "FDRResult",
    "GuardrailConfig",
    "evaluate_guardrail",
]
