import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from surrogate_reliability import NearestNeighborDomainGuard
from surrogate_reliability.decision import DecisionPolicy
from surrogate_reliability.external import evaluate_external


def test_frozen_batch_evaluation_and_missing_data():
    x = np.arange(20.0).reshape(-1, 1)
    model = ExtraTreesRegressor(n_estimators=5, random_state=7).fit(x, x.ravel())
    artifact = {
        "model": model,
        "feature_names": ["x"],
        "target_name": "y",
        "domain_guard": NearestNeighborDomainGuard().fit(x),
        "normalized_conformal_quantile": [2.0],
    }
    frame = pd.DataFrame(
        {
            "x": [3.0, 6.0, 9.0, 12.0],
            "y": [3.0, 6.0, 9.0, 12.0],
            "sample_id": ["a", "b", "c", "d"],
            "batch_id": ["a", "a", "b", "b"],
        }
    )
    before = model.predict(x)
    result = evaluate_external(artifact, frame, DecisionPolicy(5, 2), bootstrap_draws=100)
    assert result["batches"] == 2
    assert result["batch_bootstrap_95"] is not None
    np.testing.assert_array_equal(before, model.predict(x))
    frame.loc[0, "x"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        evaluate_external(artifact, frame, DecisionPolicy(5, 2))
