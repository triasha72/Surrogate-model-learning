#!/usr/bin/env python3
"""Build a readable model card from frozen concrete experiment receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(confirmation: dict[str, object], shift: dict[str, object]) -> str:
    point = confirmation["point_prediction"]
    reliability = confirmation["reliability"]
    normalized = reliability["normalized_conformal"]
    guard = reliability["nearest_neighbor_domain_guard"]
    shift_definition = shift["shift_definition"]
    shift_guard = shift["guard_summary"]
    dataset = (
        f"{confirmation['dataset']} ({confirmation['dataset_doi']}); "
        f"{confirmation['dataset_license']}"
    )
    rows = confirmation["rows"]
    return f"""# Concrete reliability model card

## What this model does

This is a research model for estimating concrete compressive strength from eight
laboratory mixture and curing inputs. It returns a point estimate, a nominal 90%
prediction interval, and a warning when an input is far from the training data.
It is not a structural-design, construction-release, or in-service safety tool.

## Frozen evidence

- Dataset: {dataset}.
- Split: {confirmation['split_policy']}.
- Rows: {rows['train']} train, {rows['calibration']} calibration, and {rows['test']} untouched test.
- Test performance: R² {point['r2']:.3f}, RMSE {point['rmse_mpa']:.3f} MPa,
  and MAE {point['mae_mpa']:.3f} MPa.
- Normalized conformal interval: {normalized['test_coverage']:.1%} test coverage
  at {reliability['nominal_coverage']:.0%} nominal coverage, with mean half-width
  {normalized['mean_half_width_mpa']:.3f} MPa.
- Domain guard: {guard['test_outside_domain_fraction']:.1%} of untouched test
  inputs were outside the training-distance threshold.

## Intended use

Use it only to explore a laboratory-style mixture space similar to the public
dataset, and treat the interval and domain warning as prompts for further
testing. Preserve the exact model artifact and source-data hash recorded in
`results/concrete_reliability_confirmation_v1.json` when reproducing a result.

## Known failure under a declared shift

The guard was tested retrospectively on the {shift_definition['tail']}
{shift_definition['quantile']:.0%} tail of `{shift_definition['feature']}`
(boundary {shift_definition['boundary']}).
Its outside-domain rate was {shift_guard['shifted_outside_domain_fraction']:.1%}
there, compared with {shift_guard['reference_outside_domain_fraction']:.1%} in
the reference rows. It therefore did **not** flag this age-tail slice reliably.
Do not use the guard as proof that the model will detect every operating shift.

## Limits and next check

The evidence comes from one public laboratory dataset and one frozen protocol.
Before any broader use, evaluate prospectively collected mixtures, compare the
predictions with fresh strength tests, and recalibrate intervals on the target
operating conditions.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    confirmation = json.loads(args.confirmation.read_text(encoding="utf-8"))
    shift = json.loads(args.shift.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(confirmation, shift), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
