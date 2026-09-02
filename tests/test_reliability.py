import numpy as np

from surrogate_reliability import (
    NearestNeighborDomainGuard,
    conformal_coverage,
    conformal_quantiles,
    normalized_conformal_quantiles,
    normalized_interval_half_width,
)


def test_conformal_intervals_cover_calibration_residuals():
    target = np.arange(20, dtype=float).reshape(10, 2)
    prediction = target + 1
    quantiles = conformal_quantiles(target, prediction, 0.9)
    np.testing.assert_array_equal(quantiles, [1, 1])
    np.testing.assert_array_equal(conformal_coverage(target, prediction, quantiles), [1, 1])


def test_normalized_conformal_intervals_expand_with_local_scale():
    target = np.array([[1.0], [2.0], [4.0]])
    prediction = np.array([[0.0], [1.0], [2.0]])
    scale = np.array([[1.0], [1.0], [2.0]])
    quantile = normalized_conformal_quantiles(target, prediction, scale, coverage=0.5)
    widths = normalized_interval_half_width(np.array([[1.0], [3.0]]), quantile)
    assert widths[1, 0] == 3 * widths[0, 0]


def test_domain_guard_flags_distant_inputs():
    train = np.column_stack((np.linspace(0, 1, 20), np.linspace(0, 1, 20)))
    guard = NearestNeighborDomainGuard().fit(train)
    assert not guard.outside_domain([[0.5, 0.5]])[0]
    assert guard.outside_domain([[10, 10]])[0]
