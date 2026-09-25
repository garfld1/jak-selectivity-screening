"""
QSAR regression training with Morgan fingerprints and cross-validation.

Primary API (notebook):

    from src.qsar_train import run_qsar_cv
    run_qsar_cv(qsar_df, model="rf", split="scaffold", fp_type="binary")

CLI (loads a saved qsar_df CSV/TSV):

    python src/qsar_train.py --input qsar_df.csv --model rf --split scaffold

This version adds:
- checkpoint logging
- optional tqdm progress bars
- timing for major stages
- a run log saved alongside outputs
- dummy baseline model
"""

from __future__ import annotations

import argparse
import logging
import os
from time import perf_counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

RDLogger.DisableLog("rdApp.*")

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


ModelName = Literal["linear", "rf", "svr", "xgb", "dummy"]
SplitStrategy = Literal["random", "scaffold"]
FpType = Literal["binary", "count"]

RADIUS = 2
N_BITS = 2048
RANDOM_STATE = 42


def get_logger(name: str = "qsar_train") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


LOGGER = get_logger()


def add_file_logging(log_path: str) -> None:
    """Add a file handler once per run."""
    abs_log_path = os.path.abspath(log_path)
    if any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abs_log_path
        for h in LOGGER.handlers
    ):
        return

    os.makedirs(os.path.dirname(abs_log_path), exist_ok=True)
    file_handler = logging.FileHandler(abs_log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def validate_qsar_df(df: pd.DataFrame, split: SplitStrategy) -> None:
    required = {"smiles", "delta_pIC50"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"qsar_df missing required columns: {sorted(missing)}")
    if split == "scaffold" and "scaffold" not in df.columns:
        raise ValueError('split="scaffold" requires a "scaffold" column in qsar_df')


def _build_fp_generator():
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=N_BITS,
        includeChirality=True,
        useBondTypes=True,
    )


def _maybe_progress(iterable, total=None, desc=""):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def generate_morgan_fingerprints(
    smiles_list: list[str],
    fp_type: FpType,
) -> tuple[np.ndarray, np.ndarray]:
    fp_gen = _build_fp_generator()
    fps = []
    valid_indices = []

    iterator = _maybe_progress(enumerate(smiles_list), total=len(smiles_list), desc="Fingerprints")
    for i, smi in iterator:
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
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"Unknown model: {name}")


def get_cv_splitter(strategy: SplitStrategy, n_splits: int):
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
    if np.isnan(rho):
        rho = 0.0

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

    split_iter = _maybe_progress(
        split_iter,
        total=getattr(splitter, "n_splits", None),
        desc="CV folds",
    )

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        fold_start = perf_counter()
        LOGGER.info(
            "Fold %d/%s | train=%d | test=%d | training model...",
            fold_idx,
            getattr(splitter, "n_splits", "?"),
            len(train_idx),
            len(test_idx),
        )

        est = get_model(model_name)
        est.fit(X[train_idx], y[train_idx])

        LOGGER.info("Fold %d | predicting...", fold_idx)
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

        elapsed = perf_counter() - fold_start
        LOGGER.info(
            "Fold %d done in %.1fs | R2=%.4f | RMSE=%.4f | MAE=%.4f | Spearman=%.4f",
            fold_idx,
            elapsed,
            metrics["r2"],
            metrics["rmse"],
            metrics["mae"],
            metrics["spearman"],
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
    LOGGER.info("Summary over folds:")
    LOGGER.info("R2:       %.4f ± %.4f", summary["r2_mean"], summary["r2_std"])
    LOGGER.info("RMSE:     %.4f ± %.4f", summary["rmse_mean"], summary["rmse_std"])
    LOGGER.info("MAE:      %.4f ± %.4f", summary["mae_mean"], summary["mae_std"])
    LOGGER.info("Spearman: %.4f ± %.4f", summary["spearman_mean"], summary["spearman_std"])


def run_qsar_cv(
    qsar_df: pd.DataFrame,
    model: ModelName = "rf",
    split: SplitStrategy = "scaffold",
    fp_type: FpType = "binary",
    output_dir: str = "results/qsar",
    n_folds: int = 5,
) -> dict:
    t0 = perf_counter()

    LOGGER.info("=" * 70)
    LOGGER.info("Starting QSAR cross-validation")
    LOGGER.info(
        "Rows: %d | Model: %s | Split: %s | Fingerprint: %s | Folds: %d",
        len(qsar_df),
        model,
        split,
        fp_type,
        n_folds,
    )
    LOGGER.info("=" * 70)

    validate_qsar_df(qsar_df, split)
    LOGGER.info("Input validation passed")

    smiles = qsar_df["smiles"].astype(str).tolist()

    fp_start = perf_counter()
    LOGGER.info("Generating Morgan fingerprints...")
    X, valid_indices = generate_morgan_fingerprints(smiles, fp_type)
    fp_elapsed = perf_counter() - fp_start

    n_dropped = len(qsar_df) - len(valid_indices)
    LOGGER.info("Fingerprint matrix shape: %s", X.shape)
    LOGGER.info("Fingerprint generation took %.1fs", fp_elapsed)
    if n_dropped:
        LOGGER.info("Dropped %d rows with invalid SMILES", n_dropped)

    LOGGER.info("Preparing targets...")
    y = qsar_df["delta_pIC50"].to_numpy(dtype=float)[valid_indices]

    groups = None
    if split == "scaffold":
        LOGGER.info("Using scaffold groups for GroupKFold")
        groups = qsar_df["scaffold"].to_numpy()[valid_indices]

    LOGGER.info("Creating CV splitter...")
    splitter = get_cv_splitter(split, n_folds)

    cv_start = perf_counter()
    fold_metrics, predictions_df = run_cross_validation(
        X, y, model, splitter, groups=groups
    )
    cv_elapsed = perf_counter() - cv_start

    summary = summarize_metrics(fold_metrics)

    os.makedirs(output_dir, exist_ok=True)
    prefix = _output_prefix(output_dir, model, split, fp_type)
    log_path = f"{prefix}.log"
    add_file_logging(log_path)

    LOGGER.info("Saving outputs to %s", output_dir)
    LOGGER.info("Writing log to %s", log_path)

    save_predictions(predictions_df, f"{prefix}_predictions.csv")
    save_metrics(fold_metrics, summary, f"{prefix}_metrics.csv")
    LOGGER.info("Saved CSV outputs")

    LOGGER.info("Generating plots...")
    plot_results(
        predictions_df["y_true"].to_numpy(),
        predictions_df["y_pred"].to_numpy(),
        prefix,
    )
    LOGGER.info("Saved plots")

    print_summary(summary)

    total_elapsed = perf_counter() - t0
    LOGGER.info("Cross-validation time: %.1fs", cv_elapsed)
    LOGGER.info("Total run time: %.1fs", total_elapsed)
    LOGGER.info("Finished successfully")

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
        help="Directory for metrics, predictions, plots, and logs (default: results/qsar)",
    )
    parser.add_argument(
        "--model",
        choices=["linear", "rf", "svr", "xgb", "dummy"],
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