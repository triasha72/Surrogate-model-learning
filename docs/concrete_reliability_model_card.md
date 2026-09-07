# Concrete reliability model card

## What this model does

This is a research model for estimating concrete compressive strength from eight
laboratory mixture and curing inputs. It returns a point estimate, a nominal 90%
prediction interval, and a warning when an input is far from the training data.
It is not a structural-design, construction-release, or in-service safety tool.

## Frozen evidence

- Dataset: UCI Concrete Compressive Strength (10.24432/C5PK67); CC BY 4.0.
- Split: grouped by seven-component mixture; ages cannot cross splits.
- Rows: 620 train, 194 calibration, and 216 untouched test.
- Test performance: R² 0.902, RMSE 5.025 MPa,
  and MAE 3.799 MPa.
- Normalized conformal interval: 95.8% test coverage
  at 90% nominal coverage, with mean half-width
  9.787 MPa.
- Domain guard: 15.3% of untouched test
  inputs were outside the training-distance threshold.

## Intended use

Use it only to explore a laboratory-style mixture space similar to the public
dataset, and treat the interval and domain warning as prompts for further
testing. Preserve the exact model artifact and source-data hash recorded in
`results/concrete_reliability_confirmation_v1.json` when reproducing a result.

## Known failure under a declared shift

The guard was tested retrospectively on the upper
90% tail of `Age`
(boundary 100.0).
Its outside-domain rate was 2.6%
there, compared with 6.9% in
the reference rows. It therefore did **not** flag this age-tail slice reliably.
Do not use the guard as proof that the model will detect every operating shift.

## Limits and next check

The evidence comes from one public laboratory dataset and one frozen protocol.
Before any broader use, evaluate prospectively collected mixtures, compare the
predictions with fresh strength tests, and recalibrate intervals on the target
operating conditions.
