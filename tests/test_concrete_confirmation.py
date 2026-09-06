import json
from pathlib import Path

import numpy as np

from scripts.confirm_concrete_reliability import mixture_grouped_split


def test_mixture_grouped_split_prevents_age_variant_leakage():
    rows = []
    for mixture in range(30):
        for age in (7, 28, 90):
            rows.append([mixture, mixture + 1, 0, 150, 2, 900, 700, age])
    features = np.asarray(rows, dtype=float)
    train, calibration, test = mixture_grouped_split(features)

    def mixtures(indices):
        return {tuple(features[index, :-1]) for index in indices}

    assert mixtures(train).isdisjoint(mixtures(calibration))
    assert mixtures(train).isdisjoint(mixtures(test))
    assert mixtures(calibration).isdisjoint(mixtures(test))


def test_published_confirmation_uses_real_data_and_reports_width():
    artifact = json.loads(
        (
            Path(__file__).parents[1]
            / "results/concrete_reliability_confirmation_v1.json"
        ).read_text()
    )
    assert artifact["contains_synthetic_data"] is False
    assert artifact["rows"]["test"] == 216
    normalized = artifact["reliability"]["normalized_conformal"]
    assert normalized["test_coverage"] >= 0.9
    assert normalized["mean_half_width_mpa"] > 0
