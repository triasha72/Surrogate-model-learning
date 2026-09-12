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
