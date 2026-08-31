import json
from pathlib import Path

import numpy as np

from scripts.train_airfoil_real_data import grouped_split


def test_grouped_split_has_no_experimental_condition_leakage():
    rows = []
    for group in range(20):
        for frequency in (100, 200, 400):
            rows.append([frequency, group, group / 10 + 0.1, group + 20, 0.01])
    features = np.asarray(rows, dtype=float)
    train, validation, test = grouped_split(features, seed=42)

    def groups(indices):
        return {
            (features[index, 1], features[index, 2], features[index, 3])
            for index in indices
        }

    assert groups(train).isdisjoint(groups(validation))
    assert groups(train).isdisjoint(groups(test))
    assert groups(validation).isdisjoint(groups(test))


def test_published_real_result_preserves_grouped_holdout_contract():
    artifact = json.loads(
        (Path(__file__).parents[1] / "results/airfoil_real_data_v1.json").read_text()
    )
    assert artifact["dataset"] == "UCI Airfoil Self-Noise"
    assert artifact["contains_synthetic_data"] is False
    assert artifact["rows"] == {"train": 1068, "validation": 218, "test": 217}
    assert artifact["selected_on_validation"] == "gaussian_process"
    assert artifact["models"]["gaussian_process"]["test"]["r2"] > 0.8
