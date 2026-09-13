"""Evaluate a frozen concrete artifact without fitting or recalibrating it."""

import numpy as np

from surrogate_reliability.decision import evaluate_decisions


def evaluate_external(artifact, frame, policy, *, bootstrap_draws=1000, seed=2026):
    names = list(artifact["feature_names"])
    required = names + [artifact["target_name"], "sample_id", "batch_id"]
    if any(c not in frame for c in required) or frame.empty:
        raise ValueError(
            "nonempty data with artifact features, target, sample_id and batch_id required"
        )
    if frame[required].isna().any().any() or frame.sample_id.duplicated().any():
        raise ValueError("missing data or duplicate sample IDs")
    if any(frame[c].astype(str).str.strip().eq("").any() for c in ["sample_id", "batch_id"]):
        raise ValueError("nonempty identifiers required")
    x = frame[names].to_numpy(dtype=float)
    y = frame[artifact["target_name"]].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite features and outcomes required")
    if bootstrap_draws < 100:
        raise ValueError("at least 100 bootstrap draws required")
    prediction = artifact["model"].predict(x)
    ensemble = np.asarray([t.predict(x) for t in artifact["model"].estimators_])
    width = np.maximum(ensemble.std(axis=0, ddof=1), 1e-6) * float(
        np.asarray(artifact["normalized_conformal_quantile"]).ravel()[0]
    )
    outside = artifact["domain_guard"].outside_domain(x)
    decisions = evaluate_decisions(prediction, width, outside, y, policy)
    covered = (y >= prediction - width) & (y <= prediction + width)
    error = prediction - y

    def metrics(idx):
        return {
            "mae_mpa": float(np.abs(error[idx]).mean()),
            "rmse_mpa": float(np.sqrt(np.mean(error[idx] ** 2))),
            "coverage": float(covered[idx].mean()),
            "mean_width_mpa": float((2 * width[idx]).mean()),
            "outside_fraction": float(np.asarray(outside)[idx].mean()),
        }

    groups = frame.batch_id.astype(str).to_numpy()
    batches = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    ci = None
    if len(batches) >= 2:
        rng = np.random.default_rng(seed)
        indices = list(batches.values())
        draws = [
            metrics(
                np.concatenate([indices[i] for i in rng.integers(len(indices), size=len(indices))])
            )
            for _ in range(bootstrap_draws)
        ]
        ci = {
            key: np.quantile([v[key] for v in draws], [0.025, 0.975]).tolist() for key in draws[0]
        }
    return {
        "scope": "frozen-model evaluation; source independence must be audited",
        "metrics": metrics(np.arange(len(y))),
        "decisions": decisions,
        "batch_metrics": {g: metrics(i) for g, i in batches.items()},
        "batch_bootstrap_95": ci,
        "bootstrap_seed": seed,
        "bootstrap_draws": bootstrap_draws,
        "batches": len(batches),
        "rows": [
            {
                "sample_id": str(frame.iloc[i].sample_id),
                "prediction_mpa": float(prediction[i]),
                "half_width_mpa": float(width[i]),
                "outside_domain": bool(outside[i]),
            }
            for i in range(len(y))
        ],
    }
