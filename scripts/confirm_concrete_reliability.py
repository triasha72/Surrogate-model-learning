#!/usr/bin/env python3
"""Confirm the frozen reliability method on untouched UCI concrete data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from surrogate_reliability import (
    NearestNeighborDomainGuard,
    conformal_coverage,
    conformal_quantiles,
    normalized_conformal_quantiles,
    normalized_interval_half_width,
)

FEATURES = (
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer",
    "Coarse Aggregate", "Fine Aggregate", "Age",
)
TARGET = "Concrete compressive strength"


def mixture_grouped_split(features: np.ndarray, seed: int = 2026):
    """Keep measurements of one concrete mixture out of multiple partitions."""
    groups = np.asarray(
        ["|".join(format(value, ".12g") for value in row[:-1]) for row in features]
    )
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_calibration, test = next(outer.split(features, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_relative, calibration_relative = next(
        inner.split(features[train_calibration], groups=groups[train_calibration])
    )
    return train_calibration[train_relative], train_calibration[calibration_relative], test


def ensemble_scale(model: ExtraTreesRegressor, features: np.ndarray) -> np.ndarray:
    predictions = np.asarray([tree.predict(features) for tree in model.estimators_])
    return np.std(predictions, axis=0, ddof=1).reshape(-1, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/concrete_reliability_confirmation_v1.json")
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("models/concrete_reliability_v1.joblib")
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    features = frame[list(FEATURES)].to_numpy(dtype=float)
    target = frame[TARGET].to_numpy(dtype=float).reshape(-1, 1)
    train, calibration, test = mixture_grouped_split(features, args.seed)
    model = ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=args.seed
    )
    model.fit(features[train], target[train].ravel())
    calibration_prediction = model.predict(features[calibration]).reshape(-1, 1)
    test_prediction = model.predict(features[test]).reshape(-1, 1)
    plain_quantile = conformal_quantiles(target[calibration], calibration_prediction, 0.9)
    normalized_quantile = normalized_conformal_quantiles(
        target[calibration], calibration_prediction,
        ensemble_scale(model, features[calibration]), 0.9,
    )
    normalized_width = normalized_interval_half_width(
        ensemble_scale(model, features[test]), normalized_quantile
    )
    guard = NearestNeighborDomainGuard().fit(features[train])

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model, "feature_names": FEATURES, "target_name": TARGET,
            "normalized_conformal_quantile": normalized_quantile,
            "nominal_coverage": 0.9, "domain_guard": guard,
        },
        args.model_output,
    )
    payload = {
        "schema_version": "1.0",
        "status": "external confirmation; protocol frozen before test evaluation",
        "dataset": "UCI Concrete Compressive Strength",
        "dataset_doi": "10.24432/C5PK67",
        "dataset_license": "CC BY 4.0",
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "seed": args.seed,
        "split_policy": "grouped by seven-component mixture; ages cannot cross splits",
        "rows": {"train": len(train), "calibration": len(calibration), "test": len(test)},
        "point_prediction": {
            "r2": float(r2_score(target[test], test_prediction)),
            "rmse_mpa": float(mean_squared_error(target[test], test_prediction) ** 0.5),
            "mae_mpa": float(mean_absolute_error(target[test], test_prediction)),
        },
        "reliability": {
            "nominal_coverage": 0.9,
            "plain_conformal": {
                "test_coverage": float(conformal_coverage(
                    target[test], test_prediction, plain_quantile
                )[0]),
                "mean_half_width_mpa": float(plain_quantile[0]),
            },
            "normalized_conformal": {
                "scale": "standard deviation across Extra Trees members",
                "test_coverage": float(np.mean(
                    np.abs(target[test] - test_prediction) <= normalized_width
                )),
                "mean_half_width_mpa": float(np.mean(normalized_width)),
            },
            "nearest_neighbor_domain_guard": {
                "training_distance_percentile": 95,
                "test_outside_domain_fraction": float(np.mean(
                    guard.outside_domain(features[test])
                )),
            },
        },
        "model_artifact": {
            "path": str(args.model_output),
            "sha256": hashlib.sha256(args.model_output.read_bytes()).hexdigest(),
        },
        "contains_synthetic_data": False,
        "limitations": [
            "This confirms one fixed method on one public laboratory dataset.",
            "It does not establish reliability for in-service concrete or arbitrary shifts.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
