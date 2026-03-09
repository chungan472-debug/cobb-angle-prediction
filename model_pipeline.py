#!/usr/bin/env python3
"""Small-sample Cobb-force modeling pipeline.

This script is designed for datasets where each patient has multiple force/Cobb
measurements, but biological features are constant within a patient.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Monotone decreasing correction curve: Cobb(F) = C_inf + (C0 - C_inf) * exp(-k * F)
def cobb_curve(force: np.ndarray, c0: float, c_inf: float, k: float) -> np.ndarray:
    return c_inf + (c0 - c_inf) * np.exp(-k * force)


@dataclass
class PatientCurve:
    patient_id: str
    c0: float
    c_inf: float
    k: float
    f50: float
    max_correction_ratio: float


def _fit_single_patient_curve(g: pd.DataFrame, force_col: str, cobb_col: str, patient_id: str) -> PatientCurve:
    x = g[force_col].to_numpy(dtype=float)
    y = g[cobb_col].to_numpy(dtype=float)

    order = np.argsort(x)
    x, y = x[order], y[order]

    c0_init = float(y[np.argmin(x)])
    c_inf_init = float(np.min(y))
    k_init = 0.02

    # bounds: c_inf <= c0, k>0
    lower = [c0_init - 40.0, max(0.0, c_inf_init - 20.0), 1e-4]
    upper = [c0_init + 40.0, c0_init, 2.0]

    try:
        popt, _ = curve_fit(
            lambda f, c0, c_inf, k: cobb_curve(f, c0, c_inf, k),
            x,
            y,
            p0=[c0_init, c_inf_init, k_init],
            bounds=(lower, upper),
            maxfev=20000,
        )
    except RuntimeError:
        popt = [c0_init, c_inf_init, k_init]

    c0, c_inf, k = map(float, popt)
    correction = max(c0 - c_inf, 1e-6)
    f50 = float(np.log(2.0) / k)
    max_correction_ratio = float(correction / max(c0, 1e-6))
    return PatientCurve(
        patient_id=str(patient_id),
        c0=c0,
        c_inf=c_inf,
        k=k,
        f50=f50,
        max_correction_ratio=max_correction_ratio,
    )


def _first_varying_columns(df: pd.DataFrame, group_col: str, excluded: Iterable[str]) -> list[str]:
    candidates = [c for c in df.columns if c not in set(excluded)]
    usable = []
    for c in candidates:
        per_patient_unique = df.groupby(group_col)[c].nunique(dropna=False)
        if per_patient_unique.max() == 1 and df[c].nunique(dropna=False) > 1:
            usable.append(c)
    return usable


def fit_all_patient_curves(df: pd.DataFrame, patient_col: str, force_col: str, cobb_col: str) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby(patient_col):
        rows.append(_fit_single_patient_curve(g, force_col, cobb_col, str(pid)).__dict__)
    return pd.DataFrame(rows)


def train_meta_model(patient_level_df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", MultiOutputRegressor(BayesianRidge())),
        ]
    )
    X = patient_level_df[feature_cols].to_numpy(dtype=float)
    y = patient_level_df[["k", "c_inf"]].to_numpy(dtype=float)
    model.fit(X, y)
    return model


def predict_force(c0: float, target_cobb: float, k: float, c_inf: float) -> float:
    # Rearranged from target = c_inf + (c0 - c_inf) * exp(-kF)
    numer = target_cobb - c_inf
    denom = c0 - c_inf
    ratio = np.clip(numer / max(denom, 1e-6), 1e-6, 0.999999)
    return float(-np.log(ratio) / max(k, 1e-6))


def leave_one_patient_out_eval(
    patient_df: pd.DataFrame,
    feature_cols: list[str],
    target_cobb_for_eval: float | None,
) -> dict:
    preds = []
    truths = []
    force_preds = []
    force_truths = []

    for test_id in patient_df["patient_id"].tolist():
        tr = patient_df[patient_df["patient_id"] != test_id]
        te = patient_df[patient_df["patient_id"] == test_id]
        if len(tr) < 3:
            continue

        model = train_meta_model(tr, feature_cols)
        X_te = te[feature_cols].to_numpy(dtype=float)
        pred_k, pred_c_inf = model.predict(X_te)[0]
        true_k, true_c_inf = te[["k", "c_inf"]].to_numpy(dtype=float)[0]
        preds.append([pred_k, pred_c_inf])
        truths.append([true_k, true_c_inf])

        if target_cobb_for_eval is not None:
            c0 = float(te["c0"].iloc[0])
            force_preds.append(predict_force(c0, target_cobb_for_eval, pred_k, pred_c_inf))
            force_truths.append(predict_force(c0, target_cobb_for_eval, true_k, true_c_inf))

    if not preds:
        return {"error": "Not enough patients for LOPO evaluation."}

    preds_arr = np.array(preds)
    truths_arr = np.array(truths)

    result = {
        "k_mae": float(mae(truths_arr[:, 0], preds_arr[:, 0])),
        "c_inf_mae": float(mae(truths_arr[:, 1], preds_arr[:, 1])),
        "k_r2": float(safe_r2(truths_arr[:, 0], preds_arr[:, 0])),
        "c_inf_r2": float(safe_r2(truths_arr[:, 1], preds_arr[:, 1])),
    }
    if force_preds:
        result["force_mae_at_target"] = float(mae(np.array(force_truths), np.array(force_preds)))
    return result


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) <= 1:
        return float("nan")
    return r2_score(y_true, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Cobb-force predictor from small cohort data.")
    parser.add_argument("--input", required=True, help="CSV with repeated force/Cobb measurements.")
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--force-col", default="force")
    parser.add_argument("--cobb-col", default="cobb")
    parser.add_argument("--feature-cols", nargs="*", default=None, help="Optional biological feature columns.")
    parser.add_argument("--target-cobb", type=float, default=None, help="Optional target Cobb for evaluation.")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = [args.patient_col, args.force_col, args.cobb_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    patient_curve_df = fit_all_patient_curves(df, args.patient_col, args.force_col, args.cobb_col)

    excluded = required + ["target_cobb", "target_force"]
    if args.feature_cols:
        feature_cols = args.feature_cols
    else:
        feature_cols = _first_varying_columns(df, args.patient_col, excluded)

    patient_features = df.groupby(args.patient_col)[feature_cols].first().reset_index()
    patient_level = patient_curve_df.merge(patient_features, left_on="patient_id", right_on=args.patient_col, how="left")
    if args.patient_col != "patient_id":
        patient_level = patient_level.drop(columns=[args.patient_col])

    if len(feature_cols) == 0:
        raise ValueError("No usable biological feature columns found. Provide --feature-cols explicitly.")

    model = train_meta_model(patient_level, feature_cols)
    metrics = leave_one_patient_out_eval(patient_level, feature_cols, args.target_cobb)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    patient_level.to_csv(outdir / "patient_level_parameters.csv", index=False)
    (outdir / "features_used.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "lopo_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save a simple deploy artifact (coefficients are in each BayesianRidge estimator)
    estimators = model.named_steps["reg"].estimators_
    deploy = {
        "feature_cols": feature_cols,
        "targets": ["k", "c_inf"],
        "bayesian_ridge": [
            {
                "coef": est.coef_.tolist(),
                "intercept": float(est.intercept_),
                "alpha": float(est.alpha_),
                "lambda": float(est.lambda_),
            }
            for est in estimators
        ],
        "scaler": {
            "mean": model.named_steps["scaler"].mean_.tolist(),
            "scale": model.named_steps["scaler"].scale_.tolist(),
        },
    }
    (outdir / "deploy_model.json").write_text(json.dumps(deploy, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done. Saved:")
    for f in ["patient_level_parameters.csv", "features_used.json", "lopo_metrics.json", "deploy_model.json"]:
        print(f"- {outdir / f}")


if __name__ == "__main__":
    main()
