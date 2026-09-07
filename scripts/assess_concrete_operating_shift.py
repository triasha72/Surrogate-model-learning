#!/usr/bin/env python3
"""Measure a frozen domain guard on a declared operating-condition tail."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from surrogate_reliability import summarize_operating_shift


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--tail", choices=("upper", "lower"), default="upper")
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.5 < args.quantile < 1.0:
        raise ValueError("quantile must be between 0.5 and 1.0")
    bundle = joblib.load(args.model)
    feature_names = list(bundle["feature_names"])
    if args.feature not in feature_names:
        raise ValueError(f"Unknown feature {args.feature!r}; expected one of {feature_names}")
    frame = pd.read_csv(args.data)
    features = frame[feature_names].to_numpy(dtype=float)
    values = frame[args.feature].to_numpy(dtype=float)
    percentile = args.quantile if args.tail == "upper" else 1 - args.quantile
    boundary = float(np.quantile(values, percentile))
    shifted_mask = values >= boundary if args.tail == "upper" else values <= boundary
    reference_mask = ~shifted_mask
    result = summarize_operating_shift(
        bundle["domain_guard"], features[reference_mask], features[shifted_mask]
    )
    payload = {
        "schema_version": "1.0",
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "shift_definition": {
            "feature": args.feature,
            "tail": args.tail,
            "quantile": args.quantile,
            "boundary": boundary,
        },
        "guard_summary": result,
        "interpretation": (
            "Retrospective public-data operating-slice diagnostic; it does not prove "
            "performance under a prospectively collected distribution shift."
        ),
        "contains_source_rows": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
