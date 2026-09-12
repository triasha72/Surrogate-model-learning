"""Explicit decision support for interval predictions; never certify a structure."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionPolicy:
    minimum_strength_mpa: float
    maximum_half_width_mpa: float

    def __post_init__(self):
        if not np.isfinite(self.minimum_strength_mpa) or self.minimum_strength_mpa <= 0:
            raise ValueError("minimum strength must be positive and finite")
        if not np.isfinite(self.maximum_half_width_mpa) or self.maximum_half_width_mpa <= 0:
            raise ValueError("maximum half-width must be positive and finite")

    def decide(self, prediction: float, half_width: float, outside_domain: bool) -> dict:
        if not np.isfinite([prediction, half_width]).all() or half_width < 0:
            raise ValueError("prediction and nonnegative half-width must be finite")
        lower, upper = prediction - half_width, prediction + half_width
        if outside_domain:
            action, reason = "request_measurement", "outside_training_domain"
        elif half_width > self.maximum_half_width_mpa:
            action, reason = "request_measurement", "interval_too_wide"
        elif lower >= self.minimum_strength_mpa:
            action, reason = "screen_in", "interval_above_requirement"
        elif upper < self.minimum_strength_mpa:
            action, reason = "screen_out", "interval_below_requirement"
        else:
            action, reason = "request_measurement", "interval_crosses_requirement"
        return {
            "action": action,
            "reason": reason,
            "interval_mpa": [lower, upper],
            "policy": asdict(self),
            "scope": "laboratory-data screening; not structural approval",
        }


def evaluate_decisions(predictions, half_widths, outside_domain, targets, policy):
    arrays = [
        np.asarray(x).reshape(-1) for x in (predictions, half_widths, outside_domain, targets)
    ]
    if not arrays[0].size or len({len(x) for x in arrays}) != 1:
        raise ValueError("nonempty aligned arrays required")
    if not np.isfinite(arrays[3]).all():
        raise ValueError("targets must be finite")
    decisions = [
        policy.decide(float(p), float(w), bool(o)) for p, w, o in zip(*arrays[:3], strict=True)
    ]
    accepted = np.array([x["action"] == "screen_in" for x in decisions])
    rejected = np.array([x["action"] == "screen_out" for x in decisions])
    truth = arrays[3] >= policy.minimum_strength_mpa
    count = int(accepted.sum())
    return {
        "rows": len(decisions),
        "policy": asdict(policy),
        "screen_in_rows": count,
        "screen_out_rows": int(rejected.sum()),
        "measurement_fraction": float(np.mean(~(accepted | rejected))),
        "false_screen_in_rows": int(np.sum(accepted & ~truth)),
        "false_screen_in_fraction": None if not count else float(np.mean(~truth[accepted])),
        "false_screen_out_rows": int(np.sum(rejected & truth)),
    }
