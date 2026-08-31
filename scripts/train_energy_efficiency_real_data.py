#!/usr/bin/env python3
"""Train and evaluate surrogates on measured UCI building-energy data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


def grouped_split(features: np.ndarray, seed: int):
    """Keep the four orientations of a physical design in one partition."""
    group_columns = (0, 1, 2, 3, 4, 6, 7)
    groups = np.asarray(
        [
            "|".join(format(value, ".12g") for value in row[list(group_columns)])
            for row in features
        ]
    )
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_validation, test = next(outer.split(features, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=seed)
    train_relative, validation_relative = next(
        inner.split(features[train_validation], groups=groups[train_validation])
    )
    return train_validation[train_relative], train_validation[validation_relative], test


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, float]]:
    names = ("heating_load", "cooling_load")
    return {
        name: {
            "r2": float(r2_score(target[:, index], prediction[:, index])),
            "rmse": float(
                mean_squared_error(target[:, index], prediction[:, index]) ** 0.5
            ),
            "mae": float(mean_absolute_error(target[:, index], prediction[:, index])),
        }
        for index, name in enumerate(names)
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/energy_efficiency_v1.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_excel(args.data)
    features = frame[[f"X{index}" for index in range(1, 9)]].to_numpy(dtype=float)
    targets = frame[["Y1", "Y2"]].to_numpy(dtype=float)
    train, validation, test = grouped_split(features, args.seed)
    candidates = {
        "random_forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=args.seed
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=args.seed
        ),
    }
    results = {}
    for name, model in candidates.items():
        model.fit(features[train], targets[train])
        results[name] = {
            "validation": metrics(
                targets[validation], model.predict(features[validation])
            )
        }
    selected = max(
        results,
        key=lambda name: np.mean(
            [value["r2"] for value in results[name]["validation"].values()]
        ),
    )
    results[selected]["test"] = metrics(
        targets[test], candidates[selected].predict(features[test])
    )
    payload = {
        "schema_version": "1.0",
        "dataset": "UCI Energy Efficiency",
        "dataset_doi": "10.24432/C51307",
        "dataset_license": "CC BY 4.0",
        "source_sha256": sha256(args.data),
        "seed": args.seed,
        "split_policy": "grouped by physical design; orientations cannot cross splits",
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "selected_on_validation": selected,
        "models": results,
        "contains_synthetic_data": False,
        "limitations": [
            "Small controlled building experiment with 768 rows.",
            "External validity to occupied buildings and other climates is unmeasured.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["models"][selected]["test"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
