"""Stats engine — every method returns a `TestResult` on the same scale."""

from ._result import TestResult
from .bootstrap import bootstrap_diff, bootstrap_diff_of_means
from .cuped import cuped_adjust, cuped_t_test, cuped_theta
from .delta_method import delta_method_ratio
from .frequentist import two_proportion_z_test, welch_t_test
from .power import (
    SampleSize,
    mde_continuous,
    mde_proportions,
    sample_size_continuous,
    sample_size_proportions,
)
from .sequential import msprt, simulate_type_i_error_under_peeking

__all__ = [
    "TestResult",
    "SampleSize",
    "welch_t_test",
    "two_proportion_z_test",
    "bootstrap_diff",
    "bootstrap_diff_of_means",
    "delta_method_ratio",
    "cuped_theta",
    "cuped_adjust",
    "cuped_t_test",
    "msprt",
    "simulate_type_i_error_under_peeking",
    "sample_size_proportions",
    "sample_size_continuous",
    "mde_proportions",
    "mde_continuous",
]
