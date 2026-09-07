import numpy as np

from surrogate_reliability import NearestNeighborDomainGuard, summarize_operating_shift


def test_operating_shift_summary_distinguishes_far_points() -> None:
    guard = NearestNeighborDomainGuard().fit(np.array([[0.0], [0.1], [0.2], [0.3]]))
    summary = summarize_operating_shift(
        guard,
        np.array([[0.15], [0.25]]),
        np.array([[8.0], [9.0]]),
    )
    assert summary["shifted_mean_distance"] > summary["reference_mean_distance"]
    assert summary["shifted_outside_domain_fraction"] == 1.0
