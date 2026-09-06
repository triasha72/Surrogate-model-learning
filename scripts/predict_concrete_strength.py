#!/usr/bin/env python3
"""Run one guarded concrete-strength prediction from the frozen artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("values", nargs=8, type=float, metavar="FEATURE")
    args = parser.parse_args()
    artifact = joblib.load(args.model)
    features = np.asarray([args.values], dtype=float)
    prediction = float(artifact["model"].predict(features)[0])
    tree_predictions = np.asarray(
        [tree.predict(features)[0] for tree in artifact["model"].estimators_]
    )
    half_width = max(float(np.std(tree_predictions, ddof=1)), 1e-6) * float(
        artifact["normalized_conformal_quantile"][0]
    )
    outside = bool(artifact["domain_guard"].outside_domain(features)[0])
    print(json.dumps({
        "prediction_mpa": prediction,
        "interval_mpa": [prediction - half_width, prediction + half_width],
        "nominal_coverage": artifact["nominal_coverage"],
        "outside_training_domain": outside,
    }, indent=2))
    return 2 if outside else 0


if __name__ == "__main__":
    raise SystemExit(main())
