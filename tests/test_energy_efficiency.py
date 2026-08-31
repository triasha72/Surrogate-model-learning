import json
from pathlib import Path

import numpy as np

from scripts.train_energy_efficiency_real_data import grouped_split


def test_energy_split_keeps_orientations_of_each_design_together():
    rows = []
    for design in range(20):
        for orientation in (2, 3, 4, 5):
            rows.append([design, 1, 2, 3, 4, orientation, design % 3, design % 5])
    features = np.asarray(rows, dtype=float)
    train, validation, test = grouped_split(features, 42)

    def designs(indices):
        return {tuple(features[index, [0, 1, 2, 3, 4, 6, 7]]) for index in indices}

    assert designs(train).isdisjoint(designs(validation))
    assert designs(train).isdisjoint(designs(test))
    assert designs(validation).isdisjoint(designs(test))


def test_published_energy_result_is_real_held_out_evidence():
    root = Path(__file__).parents[1]
    artifact = json.loads((root / "results/energy_efficiency_v1.json").read_text())
    assert artifact["contains_synthetic_data"] is False
    assert artifact["dataset"] == "UCI Energy Efficiency"
    assert sum(artifact["rows"].values()) == 768
    selected = artifact["selected_on_validation"]
    assert set(artifact["models"][selected]["test"]) == {"heating_load", "cooling_load"}
