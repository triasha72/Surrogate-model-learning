# Surrogate Model Learning project overview

## The problem

Surrogates can look reliable when related designs are split randomly. The real
question is whether point predictions, intervals, and out-of-domain warnings
remain useful when a new design family is held out.

## What I built

I evaluated Gaussian-process and conventional surrogate models on public
engineering data using grouped physical-design splits. The workflow includes
multi-seed analysis, split-conformal intervals, and a nearest-neighbor distance
guard instead of relying on a single random-split score.

## What the evidence says

On Airfoil Self-Noise, the selected Gaussian process reached held-out R² 0.8145
on the grouped split. Its ten-seed mean was 0.8662 with standard deviation
0.0680, so the apparent score changes materially with the held-out operating
conditions. The first nominal 90% intervals also missed their coverage target
after design shift. A later concrete-data confirmation is described in the
[model card](concrete_reliability_model_card.md), but it is still public-data
evidence rather than in-service validation.

## Reproduce a real-data run

```bash
python -m pip install -e ".[dev]"
PYTHONPATH=src python scripts/train_airfoil_real_data.py \
  --data data/external/airfoil_self_noise.csv
pytest -q
```

The public dataset must be obtained separately; generated model artifacts are
rebuilt locally rather than committed.

## Next validation

The next useful test is a held-out simulation or laboratory campaign with a
predeclared shift, an untouched calibration set, and a comparison against the
same frozen domain guard. That would test whether the reliability controls
transfer beyond the public datasets in this repository.
