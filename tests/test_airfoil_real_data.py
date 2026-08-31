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
