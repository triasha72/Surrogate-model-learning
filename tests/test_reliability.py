import numpy as np

from surrogate_reliability import (
    NearestNeighborDomainGuard,
    conformal_coverage,
    conformal_quantiles,
)


def test_conformal_intervals_cover_calibration_residuals():
    target = np.arange(20, dtype=float).reshape(10, 2)
    prediction = target + 1
    quantiles = conformal_quantiles(target, prediction, 0.9)
    np.testing.assert_array_equal(quantiles, [1, 1])
    np.testing.assert_array_equal(conformal_coverage(target, prediction, quantiles), [1, 1])


def test_domain_guard_flags_distant_inputs():
    train = np.column_stack((np.linspace(0, 1, 20), np.linspace(0, 1, 20)))
    guard = NearestNeighborDomainGuard().fit(train)
    assert not guard.outside_domain([[0.5, 0.5]])[0]
    assert guard.outside_domain([[10, 10]])[0]
