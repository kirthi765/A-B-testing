"""Heterogeneous Treatment Effects — CATE estimation via econml causal forest."""

from .causal_trees import (
    build_user_level_dataset,
    cate_by_decile,
    cate_by_segment,
    fit_causal_forest,
    one_hot_segments,
    predict_cate,
)

__all__ = [
    "fit_causal_forest",
    "predict_cate",
    "cate_by_decile",
    "cate_by_segment",
    "build_user_level_dataset",
    "one_hot_segments",
]
