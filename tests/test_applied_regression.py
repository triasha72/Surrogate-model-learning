import numpy as np
import pandas as pd

from scripts.regression_energy import design_matrices


def test_encoding_and_alias_removal_use_training_only():
    frame = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0, 100.0, 200.0], "alias": [0.0, 2.0, 4.0, 6.0, 200.0, 400.0]}
    )
    matrices, names, dropped = design_matrices(frame, [0, 1, 2, 3], [4], [5], "1 + x + alias")
    assert len(names) == 2
    assert len(dropped) == 1
    assert np.linalg.matrix_rank(matrices[0]) == 2
    assert matrices[1].shape == (1, 2)
    assert matrices[2].shape == (1, 2)
