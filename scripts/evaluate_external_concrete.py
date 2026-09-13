"""Evaluate a trusted frozen artifact against a separately supplied batch dataset."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from surrogate_reliability.decision import DecisionPolicy
from surrogate_reliability.external import evaluate_external


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--expected-model-sha256", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--source-description", required=True)
    p.add_argument("--minimum-strength-mpa", type=float, required=True)
    p.add_argument("--maximum-half-width-mpa", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("use a new output path")
    model_bytes = a.model.read_bytes()
    digest = hashlib.sha256(model_bytes).hexdigest()
    if digest != a.expected_model_sha256:
        raise ValueError("frozen artifact hash mismatch")
    import io

    data = a.data.read_bytes()
    result = evaluate_external(
        joblib.load(io.BytesIO(model_bytes)),
        pd.read_csv(io.BytesIO(data)),
        DecisionPolicy(a.minimum_strength_mpa, a.maximum_half_width_mpa),
    )
    result.update(
        model_sha256=digest,
        data_sha256=hashlib.sha256(data).hexdigest(),
        source_description=a.source_description,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
