#!/usr/bin/env python3
"""Measure Gaussian-process robustness across grouped operating-condition splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    matrix = np.column_stack([data[name] for name in data.dtype.names])
    return matrix[:, :-1], matrix[:, -1]


def grouped_split(features, seed):
    groups = np.array([f"{row[1]:g}|{row[2]:g}|{row[3]:g}" for row in features])
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_validation, test = next(outer.split(features, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=seed)
    train_relative, validation_relative = next(
        inner.split(features[train_validation], groups=groups[train_validation])
    )
    return train_validation[train_relative], train_validation[validation_relative], test


def metrics(expected, predicted):
    return {
        "r2": float(r2_score(expected, predicted)),
        "rmse_db": float(mean_squared_error(expected, predicted) ** 0.5),
        "mae_db": float(mean_absolute_error(expected, predicted)),
    }


def make_model(seed: int):
    return make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(5), nu=1.5)
            + WhiteKernel(0.1),
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=seed,
        ),
    )


def interval(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)),
        "empirical_95_percent_interval": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("results/airfoil_robustness_v1.json")
    )
    args = parser.parse_args()
    features, targets = load_data(args.data)
    runs = []
    for seed in range(args.seeds):
        train, _validation, test = grouped_split(features, seed)
        model = make_model(seed)
        model.fit(features[train], targets[train])
        runs.append(
            {"seed": seed, **metrics(targets[test], model.predict(features[test]))}
        )
    payload = {
        "schema_version": "1.0",
        "dataset": "UCI Airfoil Self-Noise",
        "split_policy": "10 repeated aircraft-condition-grouped holdouts",
        "runs": runs,
        "summary": {
            name: interval([run[name] for run in runs])
            for name in ("r2", "rmse_db", "mae_db")
        },
        "limitations": [
            "The empirical interval summarizes split sensitivity; it is not a population confidence interval.",
            "The model family and preprocessing are fixed before these repeated evaluations.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
