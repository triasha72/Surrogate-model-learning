"""Distribution-free intervals and transparent extrapolation warnings."""

from __future__ import annotations

import math

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def conformal_quantiles(target: np.ndarray, prediction: np.ndarray, coverage: float = 0.9):
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must have matching [rows, outputs] shapes")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    residual = np.abs(target - prediction)
    level = min(1.0, math.ceil((len(target) + 1) * coverage) / len(target))
    return np.quantile(residual, level, axis=0, method="higher")


def conformal_coverage(target: np.ndarray, prediction: np.ndarray, quantiles: np.ndarray):
    covered = np.abs(np.asarray(target) - np.asarray(prediction)) <= np.asarray(quantiles)
    return np.mean(covered, axis=0)


def normalized_conformal_quantiles(
    target: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray,
    coverage: float = 0.9,
    minimum_scale: float = 1e-6,
) -> np.ndarray:
    """Calibrate residuals after normalizing by a model-derived local scale."""

    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if target.shape != prediction.shape or target.shape != scale.shape or target.ndim != 2:
        raise ValueError("target, prediction, and scale must share [rows, outputs] shape")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    if minimum_scale <= 0:
        raise ValueError("minimum_scale must be positive")
    score = np.abs(target - prediction) / np.maximum(scale, minimum_scale)
    level = min(1.0, math.ceil((len(target) + 1) * coverage) / len(target))
    return np.quantile(score, level, axis=0, method="higher")


def normalized_interval_half_width(
    scale: np.ndarray, quantiles: np.ndarray, minimum_scale: float = 1e-6
) -> np.ndarray:
    scale = np.asarray(scale, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if scale.ndim != 2 or quantiles.shape != (scale.shape[1],):
        raise ValueError("scale must be [rows, outputs] and quantiles must match outputs")
    return np.maximum(scale, minimum_scale) * quantiles


class NearestNeighborDomainGuard:
    """Flag points farther from training data than held-in training points."""

    def __init__(self, percentile: float = 95.0):
        if not 50 <= percentile < 100:
            raise ValueError("percentile must be in [50, 100)")
        self.percentile = percentile

    def fit(self, features: np.ndarray):
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or len(values) < 3:
            raise ValueError("features must contain at least three rows")
        self.scaler = StandardScaler().fit(values)
        transformed = self.scaler.transform(values)
        self.neighbors = NearestNeighbors(n_neighbors=2).fit(transformed)
        held_in_distance = self.neighbors.kneighbors(transformed)[0][:, 1]
        self.threshold = float(np.percentile(held_in_distance, self.percentile))
        return self

    def distances(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "threshold"):
            raise RuntimeError("domain guard must be fitted before use")
        transformed = self.scaler.transform(np.asarray(features, dtype=float))
        return self.neighbors.kneighbors(transformed, n_neighbors=1)[0][:, 0]

    def outside_domain(self, features: np.ndarray) -> np.ndarray:
        return self.distances(features) > self.threshold


def summarize_operating_shift(
    guard: NearestNeighborDomainGuard,
    reference_features: np.ndarray,
    shifted_features: np.ndarray,
) -> dict[str, float | int]:
    """Compare guard behavior between a reference and shifted operating slice.

    This is a diagnostic, not a calibration procedure: it does not change the
    threshold, fit the guard, or use outcomes from either slice.
    """

    reference = np.asarray(reference_features, dtype=float)
    shifted = np.asarray(shifted_features, dtype=float)
    if reference.ndim != 2 or shifted.ndim != 2 or reference.shape[1] != shifted.shape[1]:
        raise ValueError("reference and shifted features must be aligned two-dimensional arrays")
    if len(reference) == 0 or len(shifted) == 0:
        raise ValueError("reference and shifted features must each contain at least one row")
    reference_distances = guard.distances(reference)
    shifted_distances = guard.distances(shifted)
    return {
        "reference_rows": len(reference),
        "shifted_rows": len(shifted),
        "reference_mean_distance": float(np.mean(reference_distances)),
        "shifted_mean_distance": float(np.mean(shifted_distances)),
        "reference_outside_domain_fraction": float(np.mean(reference_distances > guard.threshold)),
        "shifted_outside_domain_fraction": float(np.mean(shifted_distances > guard.threshold)),
        "guard_threshold": float(guard.threshold),
    }
