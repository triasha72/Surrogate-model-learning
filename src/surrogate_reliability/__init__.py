"""Reliability primitives for engineering surrogate models."""

from .reliability import (
    NearestNeighborDomainGuard,
    conformal_coverage,
    conformal_quantiles,
    normalized_conformal_quantiles,
    normalized_interval_half_width,
    summarize_operating_shift,
)

__all__ = [
    "NearestNeighborDomainGuard",
    "summarize_operating_shift",
    "conformal_coverage",
    "conformal_quantiles",
    "normalized_conformal_quantiles",
    "normalized_interval_half_width",
]
