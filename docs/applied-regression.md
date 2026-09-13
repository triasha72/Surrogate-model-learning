# Applied regression on building simulations

UCI Energy Efficiency contains 768 Ecotect simulations, not occupied-building measurements.
Source: https://archive.ics.uci.edu/dataset/242/energy+efficiency (CC BY 4.0).
This is an independent applied regression project, not a coursework claim.

```sh
pip install -e '.[regression]'
python scripts/regression_energy.py --data data/ENB2012_data.xlsx --output results/regression_energy
```

The split groups orientations of the same physical configuration. Training rows fit
encoding, remove exact aliases, and estimate coefficients. Validation RMSE selects
among OLS, log-OLS with training-only smearing, and ridge, with prespecified main
and interaction formulas. The selected model is evaluated on test rows once.
The fixed main-effects OLS also has a separately labeled interval diagnostic.

Seed 42 produced heating-load R² 0.88048 / RMSE 3.35903 and cooling-load
R² 0.82008 / RMSE 3.99279. Machine-readable results and plots are in
`results/applied_regression_energy/`. This extends a previously studied public
dataset retrospectively, so the test set is not a new independent confirmation.

HC3 intervals do not account for dependence between related simulations. Classical
prediction intervals require assumptions that may fail; coverage is measured,
not guaranteed. Coefficients describe associations, not causal building effects.
X2 surface area is omitted because it is algebraically redundant with wall/roof area.
QR filtering handles remaining exact aliases using training rows only.

## Frozen concrete evaluation on a new source

`python scripts/evaluate_external_concrete.py --help` describes the batch evaluator.
Supply a trusted artifact, its previously recorded SHA-256, a CSV with the artifact's
feature names and target plus `sample_id` and `batch_id`, a source description, and
prespecified strength/interval-width thresholds. Record units as mixture components
in kg/m³, age in days and strength in MPa; audit conversions before evaluation.
The runner verifies the artifact hash and performs no fitting or recalibration.
It writes row predictions, errors, coverage, decision metrics and batch-bootstrap
intervals to a new output file. Fewer than two batches yields no interval.

Source independence cannot be inferred from a filename or unique IDs. Audit against
training/calibration sources before labeling the result external confirmation. A
passing unit test is not independent laboratory evidence. This change includes no
new external measurements.
