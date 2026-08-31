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
