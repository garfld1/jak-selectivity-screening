"""
QSAR regression training with Morgan fingerprints and cross-validation.

Primary API (notebook):

    from src.qsar_train import run_qsar_cv
    run_qsar_cv(qsar_df, model="rf", split="scaffold", fp_type="binary")

CLI (loads a saved qsar_df CSV/TSV):

    python src/qsar_train.py --input qsar_df.csv --model rf --split scaffold
"""

from __future__ import annotations

import argparse
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

RDLogger.DisableLog("rdApp.*")

ModelName = Literal["linear", "rf", "svr", "xgb"]
SplitStrategy = Literal["random", "scaffold"]
FpType = Literal["binary", "count"]

RADIUS = 2
N_BITS = 2048
RANDOM_STATE = 42


def validate_qsar_df(df: pd.DataFrame, split: SplitStrategy) -> None:
    required = {"smiles", "delta_pIC50"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"qsar_df missing required columns: {sorted(missing)}")
    if split == "scaffold" and "scaffold" not in df.columns:
        raise ValueError(
            'split="scaffold" requires a "scaffold" column in qsar_df'
        )


def _build_fp_generator():
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=N_BITS,
        includeChirality=True,
        useBondTypes=True,
    )


def generate_morgan_fingerprints(
    smiles_list: list[str],
    fp_type: FpType,
) -> tuple[np.ndarray, np.ndarray]:
    fp_gen = _build_fp_generator()
    fps = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if fp_type == "binary":
            fp = fp_gen.GetFingerprintAsNumPy(mol)
        else:
            fp = fp_gen.GetCountFingerprintAsNumPy(mol)
        fps.append(fp)
        valid_indices.append(i)

    if not fps:
        raise ValueError("No valid SMILES found for fingerprint generation")

    X = np.array(fps, dtype=np.float32)
    return X, np.array(valid_indices, dtype=int)


def get_model(name: ModelName):
    if name == "linear":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
    if name == "rf":
        return RandomForestRegressor(random_state=RANDOM_STATE)
    if name == "svr":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR()),
            ]
        )
    if name == "xgb":
        return XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    raise ValueError(f"Unknown model: {name}")


def get_cv_splitter(strategy: SplitStrategy, n_splits: int, groups=None):
    if strategy == "random":
        return KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    if strategy == "scaffold":
        return GroupKFold(n_splits=n_splits)
    raise ValueError(f"Unknown split strategy: {strategy}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(rho),
    }


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_name: ModelName,
    splitter,
    groups: np.ndarray | None = None,
) -> tuple[list[dict[str, float]], pd.DataFrame]:
    fold_metrics: list[dict[str, float]] = []
    prediction_rows: list[dict[str, float | int]] = []

    if groups is not None:
        split_iter = splitter.split(X, y, groups=groups)
    else:
        split_iter = splitter.split(X, y)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        est = get_model(model_name)
        est.fit(X[train_idx], y[train_idx])
        y_pred = est.predict(X[test_idx])
        y_true = y[test_idx]

        metrics = compute_metrics(y_true, y_pred)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

        for true_val, pred_val in zip(y_true, y_pred):
            prediction_rows.append(
                {
                    "fold": fold_idx,
                    "y_true": float(true_val),
                    "y_pred": float(pred_val),
                }
            )

    predictions_df = pd.DataFrame(prediction_rows)
    return fold_metrics, predictions_df


def summarize_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    metric_names = ["r2", "rmse", "mae", "spearman"]
    summary: dict[str, float] = {}
    for name in metric_names:
        values = np.array([m[name] for m in fold_metrics], dtype=float)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std(ddof=1) if len(values) > 1 else 0.0)
    return summary


def save_predictions(predictions_df: pd.DataFrame, path: str) -> None:
    predictions_df.to_csv(path, index=False)


def save_metrics(
    fold_metrics: list[dict[str, float]],
    summary: dict[str, float],
    path: str,
) -> None:
    rows = []
    for m in fold_metrics:
        rows.append(
            {
                "fold": m["fold"],
                "r2": m["r2"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "spearman": m["spearman"],
            }
        )
    rows.append(
        {
            "fold": "mean",
            "r2": summary["r2_mean"],
            "rmse": summary["rmse_mean"],
            "mae": summary["mae_mean"],
            "spearman": summary["spearman_mean"],
        }
    )
    rows.append(
        {
            "fold": "std",
            "r2": summary["r2_std"],
            "rmse": summary["rmse_std"],
            "mae": summary["mae_std"],
            "spearman": summary["spearman_std"],
        }
    )
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_prefix: str,
) -> None:
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=12, edgecolors="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("Actual delta_pIC50")
    ax.set_ylabel("Predicted delta_pIC50")
    ax.set_title("Predicted vs Actual")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_pred_vs_actual.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, s=12, edgecolors="none")
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted delta_pIC50")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals vs Predicted")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_residuals.png", dpi=150)
    plt.close(fig)


def _output_prefix(
    output_dir: str,
    model: ModelName,
    split: SplitStrategy,
    fp_type: FpType,
) -> str:
    return os.path.join(output_dir, f"{model}_{split}_{fp_type}")


def print_summary(summary: dict[str, float]) -> None:
    print(f"R2:       {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
    print(f"RMSE:     {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")
    print(f"MAE:      {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
    print(f"Spearman: {summary['spearman_mean']:.4f} ± {summary['spearman_std']:.4f}")


def run_qsar_cv(
    qsar_df: pd.DataFrame,
    model: ModelName = "rf",
    split: SplitStrategy = "scaffold",
    fp_type: FpType = "binary",
    output_dir: str = "results/qsar",
    n_folds: int = 5,
) -> dict:
    validate_qsar_df(qsar_df, split)

    smiles = qsar_df["smiles"].astype(str).tolist()
    X, valid_indices = generate_morgan_fingerprints(smiles, fp_type)

    n_dropped = len(qsar_df) - len(valid_indices)
    if n_dropped:
        print(f"Dropped {n_dropped} rows with invalid SMILES")

    y = qsar_df["delta_pIC50"].to_numpy(dtype=float)[valid_indices]
    groups = None
    if split == "scaffold":
        groups = qsar_df["scaffold"].to_numpy()[valid_indices]

    splitter = get_cv_splitter(split, n_folds)
    fold_metrics, predictions_df = run_cross_validation(
        X, y, model, splitter, groups=groups
    )
    summary = summarize_metrics(fold_metrics)

    os.makedirs(output_dir, exist_ok=True)
    prefix = _output_prefix(output_dir, model, split, fp_type)

    save_predictions(predictions_df, f"{prefix}_predictions.csv")
    save_metrics(fold_metrics, summary, f"{prefix}_metrics.csv")
    plot_results(
        predictions_df["y_true"].to_numpy(),
        predictions_df["y_pred"].to_numpy(),
        prefix,
    )

    print_summary(summary)

    return {
        "fold_metrics": fold_metrics,
        "summary": summary,
        "predictions": predictions_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a QSAR regression model on qsar_df."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to saved qsar_df CSV or TSV (columns: smiles, delta_pIC50; scaffold if split=scaffold)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/qsar",
        help="Directory for metrics, predictions, and plots (default: results/qsar)",
    )
    parser.add_argument(
        "--model",
        choices=["linear", "rf", "svr", "xgb"],
        default="rf",
        help="Regression model (default: rf)",
    )
    parser.add_argument(
        "--split",
        choices=["random", "scaffold"],
        default="scaffold",
        help="CV split strategy (default: scaffold)",
    )
    parser.add_argument(
        "--fp-type",
        choices=["binary", "count"],
        default="binary",
        help="Morgan fingerprint type (default: binary)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    return parser.parse_args()


def _load_qsar_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    qsar_df = _load_qsar_df(args.input)
    run_qsar_cv(
        qsar_df,
        model=args.model,
        split=args.split,
        fp_type=args.fp_type,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
