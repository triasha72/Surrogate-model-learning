#!/usr/bin/env python3
"""Applied regression on UCI Ecotect simulations; independent project, not coursework."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import scipy.linalg
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    from .train_energy_efficiency_real_data import grouped_split
except ImportError:  # Direct script execution.
    from train_energy_efficiency_real_data import grouped_split

MAIN = "1 + X1 + X3 + X4 + X5 + C(X6) + X7 + C(X8)"
INTERACTIONS = MAIN + " + X5:X7 + X1:X7 + I(X7 ** 2)"


def design_matrices(frame, train, validation, test, formula):
    """Fit design encoding and remove exact aliases using training rows alone."""
    design = patsy.dmatrix(formula, frame.iloc[train], return_type="dataframe")
    rank = np.linalg.matrix_rank(design.to_numpy())
    _, _, pivot = scipy.linalg.qr(design.to_numpy(), mode="economic", pivoting=True)
    keep = sorted(pivot[:rank])
    dropped = [name for i, name in enumerate(design.columns) if i not in keep]
    matrices = [design.to_numpy()[:, keep]]
    for indices in (validation, test):
        matrices.append(
            np.asarray(patsy.build_design_matrices([design.design_info], frame.iloc[indices])[0])[
                :, keep
            ]
        )
    return matrices, [design.columns[i] for i in keep], dropped


def scores(y, pred):
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
    }


def run(data: Path, output: Path, seed: int = 42):
    if output.exists():
        raise ValueError("use a new output directory; previous experiments are immutable")
    frame = pd.read_excel(data).dropna(subset=[f"X{i}" for i in range(1, 9)] + ["Y1", "Y2"])
    if len(frame) != 768 or not np.isfinite(frame.iloc[:, :10].to_numpy()).all():
        raise ValueError("expected 768 finite UCI Energy Efficiency rows")
    train, validation, test = grouped_split(frame[[f"X{i}" for i in range(1, 9)]].to_numpy(), seed)
    output.mkdir(parents=True)
    results = {}
    for target in ("Y1", "Y2"):
        y = frame[target].to_numpy()
        candidates = {}
        for name, formula in [("main", MAIN), ("interactions", INTERACTIONS)]:
            matrices, terms, dropped = design_matrices(frame, train, validation, test, formula)
            fit = sm.OLS(y[train], matrices[0]).fit()
            candidates["ols_" + name] = (fit, matrices, terms, dropped, False, 1.0)
            log_fit = sm.OLS(np.log(y[train]), matrices[0]).fit()
            smear = float(np.exp(log_fit.resid).mean())
            candidates["log_ols_" + name] = (log_fit, matrices, terms, dropped, True, smear)
            for alpha in (0.1, 1.0, 10.0, 100.0):
                ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
                    matrices[0], y[train]
                )
                candidates[f"ridge_{name}_{alpha}"] = (ridge, matrices, terms, dropped, False, 1.0)
        validation_scores = {}
        for name, (model, matrices, _, _, log, smear) in candidates.items():
            pred = model.predict(matrices[1])
            if log:
                pred = np.exp(pred) * smear
            validation_scores[name] = scores(y[validation], pred)
        selected = min(validation_scores, key=lambda name: validation_scores[name]["rmse"])
        model, matrices, terms, dropped, log, smear = candidates[selected]
        prediction = model.predict(matrices[2])
        if log:
            prediction = np.exp(prediction) * smear
        # Prespecified OLS diagnostics always use training rows; not model-selection evidence.
        ols, xm, names, aliases, _, _ = candidates["ols_main"]
        influence = ols.get_influence()
        robust = ols.get_robustcov_results(cov_type="HC3")
        intervals = ols.get_prediction(xm[2]).summary_frame(alpha=0.05)
        vifs = {}
        for i, term in enumerate(names):
            if term == "Intercept":
                continue
            value = float(variance_inflation_factor(xm[0], i))
            vifs[term] = value if np.isfinite(value) else None
        robust_intervals = robust.conf_int()
        bp = het_breuschpagan(ols.resid, sm.add_constant(xm[0], has_constant="skip"))
        diagnostics = {
            "status": "Exploratory diagnostics; repeated simulation designs are dependent",
            "terms": names,
            "dropped_exact_aliases": aliases,
            "vif": vifs,
            "aic": float(ols.aic),
            "bic": float(ols.bic),
            "adjusted_r2": float(ols.rsquared_adj),
            "breusch_pagan_lm_pvalue": float(bp[1]),
            "max_cooks_distance": float(influence.cooks_distance[0].max()),
            "max_leverage": float(influence.hat_matrix_diag.max()),
            "coefficients_hc3": [
                {
                    "term": n,
                    "coefficient": float(robust.params[i]),
                    "ci95": robust_intervals[i].tolist(),
                }
                for i, n in enumerate(names)
            ],
            "classical_prediction_interval_coverage": float(
                np.mean((y[test] >= intervals.obs_ci_lower) & (y[test] <= intervals.obs_ci_upper))
            ),
            "classical_interval_assumptions": (
                "Linear mean, iid normal errors; assumptions not guaranteed"
            ),
        }
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        axes[0].scatter(ols.fittedvalues, ols.resid, s=8, alpha=0.5)
        axes[0].axhline(0, color="black", linewidth=0.7)
        axes[0].set(xlabel="Fitted load", ylabel="Residual", title="Training residuals")
        sm.qqplot(ols.resid, line="s", ax=axes[1])
        axes[1].set_title("Training residual Q–Q")
        axes[2].scatter(influence.hat_matrix_diag, influence.cooks_distance[0], s=8, alpha=0.5)
        axes[2].set(xlabel="Leverage", ylabel="Cook's distance", title="Training influence")
        fig.tight_layout()
        fig.savefig(output / f"{target}_diagnostics.png", dpi=160)
        plt.close(fig)
        results[target] = {
            "selected_on_validation": selected,
            "validation": validation_scores,
            "selected_test": scores(y[test], prediction),
            "ols_diagnostics": diagnostics,
        }
    payload = {
        "schema_version": "1.0",
        "dataset": "UCI Energy Efficiency",
        "source": "https://archive.ics.uci.edu/dataset/242/energy+efficiency",
        "data_origin": "Ecotect building simulations, not occupied-building measurements",
        "license": "CC BY 4.0",
        "source_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "project_status": "independent applied regression analysis; no coursework claim",
        "evaluation_status": "retrospective extension of a previously studied public dataset",
        "split_seed": seed,
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "split_policy": "physical configuration grouped across orientations",
        "results": results,
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v["selected_test"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run(a.data, a.output, a.seed)
