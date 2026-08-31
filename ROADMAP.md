# What I want to test next

[Read the project overview and measured results](README.md)

The repository now has two real-data studies rather than relying on analytic
functions. On the UCI Energy Efficiency holdout, Extra Trees reached heating-
load R² `0.9630` and cooling-load R² `0.9330`. The uncertainty result was less
comfortable: nominal 90% intervals covered only `74.1%` and `85.3%` of the two
targets.

That undercoverage is the most useful next problem in the project. It suggests
that ordinary split conformal calibration did not transfer cleanly across the
grouped physical-design shift.

## Next experiments

- Compare local, normalized, and group-aware conformal methods. All choices will
  be made on training and validation data, not the held-out test set.
- Report interval width alongside coverage so a method cannot look safe merely
  by returning very wide ranges.
- Create a deliberate operating-condition shift in a public dataset and measure
  whether the nearest-neighbor domain guard detects it.
- Produce a model card directly from the frozen experiment record, including
  data lineage, intended use, and known failure cases.
- Package one real-data model for a small reproducible inference example.

The analytic notebooks will stay because they explain the methods well. Their
role is teaching; the UCI experiments are the evidence for how the methods
behave on measured data.
