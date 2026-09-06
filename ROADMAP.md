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

- [x] Add a normalized conformal diagnostic using tree-ensemble disagreement.
  It improved both coverage values to about 88% but remains retrospective and
  below target.
- [x] Confirm the frozen normalized conformal method on a different untouched
  public dataset. On UCI Concrete, it covered 95.83% at 90% nominal coverage;
  the checked-in report also preserves the wider-than-nominal result.
- [x] Report interval width alongside coverage so a method cannot look safe
  merely by returning very wide ranges.
- Create a deliberate operating-condition shift in a public dataset and measure
  whether the nearest-neighbor domain guard detects it.
- Produce a model card directly from the frozen experiment record, including
  data lineage, intended use, and known failure cases.
- [x] Package one real-data model for a small reproducible inference example,
  including a nearest-neighbor domain warning.

The analytic notebooks will stay because they explain the methods well. Their
role is teaching; the UCI experiments are the evidence for how the methods
behave on measured data.
