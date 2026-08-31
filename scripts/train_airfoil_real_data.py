#!/usr/bin/env python3
"""Compare surrogate models on measured UCI Airfoil Self-Noise data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def load_data(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = data.dtype.names
    matrix = np.column_stack([data[name] for name in names])
    return matrix[:, :-1], matrix[:, -1]


def grouped_split(x, seed):
    groups = np.array([f"{r[1]:g}|{r[2]:g}|{r[3]:g}" for r in x])
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val, test = next(outer.split(x, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=seed)
    train_rel, val_rel = next(inner.split(x[train_val], groups=groups[train_val]))
    return train_val[train_rel], train_val[val_rel], test


def metrics(y, p):
    return {
        "r2": float(r2_score(y, p)),
        "rmse_db": float(mean_squared_error(y, p) ** 0.5),
        "mae_db": float(mean_absolute_error(y, p)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/airfoil_real_data_v1.json"))
    args = parser.parse_args()
    x, y = load_data(args.data)
    train, val, test = grouped_split(x, args.seed)
    candidates = {
        "polynomial_ridge": make_pipeline(
            StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=10)
        ),
        "gaussian_process": make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(5), nu=1.5)
                + WhiteKernel(0.1),
                normalize_y=True,
                n_restarts_optimizer=0,
                random_state=args.seed,
            ),
        ),
    }
    records = {}
    fitted = {}
    for name, model in candidates.items():
        started = time.perf_counter()
        model.fit(x[train], y[train])
        fitted[name] = model
        records[name] = {
            "validation": metrics(y[val], model.predict(x[val])),
            "fit_seconds": time.perf_counter() - started,
        }
    scaler = StandardScaler().fit(x[train])
    started = time.perf_counter()
    rbf = RBFInterpolator(
        scaler.transform(x[train]), y[train], kernel="thin_plate_spline", smoothing=1.0
    )
    records["rbf"] = {
        "validation": metrics(y[val], rbf(scaler.transform(x[val]))),
        "fit_seconds": time.perf_counter() - started,
    }
    selected = max(records, key=lambda n: records[n]["validation"]["r2"])
    prediction = (
        rbf(scaler.transform(x[test])) if selected == "rbf" else fitted[selected].predict(x[test])
    )
    records[selected]["test"] = metrics(y[test], prediction)
    payload = {
        "schema_version": "1.0",
        "dataset": "UCI Airfoil Self-Noise",
        "dataset_doi": "10.24432/C5VW2C",
        "dataset_license": "CC BY 4.0",
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "seed": args.seed,
        "split_policy": "grouped by angle of attack, chord length, and free-stream velocity",
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "models": records,
        "selected_on_validation": selected,
        "contains_synthetic_data": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
