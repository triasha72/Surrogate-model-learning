import pytest

from surrogate_reliability.decision import DecisionPolicy, evaluate_decisions


def test_abstention_overrides_confident_prediction():
    p = DecisionPolicy(30, 5)
    assert p.decide(45, 2, True)["action"] == "request_measurement"
    assert p.decide(45, 6, False)["action"] == "request_measurement"
    assert p.decide(30, 2, False)["action"] == "request_measurement"
    assert p.decide(35, 5, False)["action"] == "screen_in"
    assert p.decide(20, 5, False)["action"] == "screen_out"


def test_risk_is_conditional_on_screened_in_rows():
    result = evaluate_decisions(
        [40, 40, 30], [1, 1, 3], [False] * 3, [35, 20, 20], DecisionPolicy(30, 5)
    )
    assert result["false_screen_in_fraction"] == 0.5
    assert result["measurement_fraction"] == pytest.approx(1 / 3)


def test_reject_nonfinite():
    with pytest.raises(ValueError):
        DecisionPolicy(30, 5).decide(float("nan"), 2, False)
