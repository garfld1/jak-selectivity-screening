"""Shared, non-mutating helpers for the extension scripts."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the original feature semantics; no original function is modified here.
from src.pharmacophore_discovery import feature_set_mask, j2_vs_background_stats

RESULTS = ROOT / "results" / "pharmacophore_analysis"
EXTENSIONS = ROOT / "extensions"


def paths():
    return {
        "matrix": RESULTS / "feature_matrix.csv",
        "candidates": RESULTS / "discovered_candidates.csv",
        "holdout": RESULTS / "holdout_results.csv",
        "validation": RESULTS / "pharmacophore validation" / "independent_validation_results.csv",
    }


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    metadata = {"ligand_id", "delta_pIC50", "selectivity_class", "SMILES", "smiles"}
    return [c for c in matrix.columns if c not in metadata]


def parse_features(value: object) -> tuple[str, ...]:
    return tuple(x for x in str(value).split("|") if x)


def target_vs_rest_stats(mask: pd.Series, classes: pd.Series, target_class: str) -> dict:
    """Class-vs-rest wrapper with the original JAK2 function as the JAK2 path."""
    if target_class == "JAK2_SELECTIVE":
        return j2_vs_background_stats(mask, classes, float(classes.eq(target_class).mean()))
    mask = pd.Series(mask, index=classes.index).astype(bool)
    positive = classes.eq(target_class)
    a, b = int((mask & positive).sum()), int((mask & ~positive).sum())
    c, d = int((~mask & positive).sum()), int((~mask & ~positive).sum())
    odds_ratio, fisher_p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    n, baseline = a + b, float(positive.mean())
    ppv = a / n if n else np.nan
    return {
        "n": n, "n_target": a, "n_not_target": b,
        "ppv": ppv, "baseline_fraction": baseline,
        "enrichment": ppv / baseline if baseline and np.isfinite(ppv) else np.nan,
        "odds_ratio": odds_ratio, "fisher_p": fisher_p,
    }


def frozen_rescore(candidates: pd.DataFrame, matrix: pd.DataFrame, subset: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Score frozen JAK2 patterns using this subset's local JAK2 baseline."""
    features = sorted({f for x in candidates.features for f in parse_features(x)})
    fm = subset.reindex(columns=features, fill_value=0).astype(bool)
    classes = subset["selectivity_class"]
    rows = []
    for _, candidate in candidates.iterrows():
        mask = feature_set_mask(fm, parse_features(candidate.features))
        stats = j2_vs_background_stats(mask, classes, float(classes.eq("JAK2_SELECTIVE").mean()))
        rows.append({"features": candidate.features, "pattern": candidate.pattern,
                     "n_features": int(candidate.n_features),
                     **{f"{prefix}_{key}": value for key, value in stats.items()}})
    return pd.DataFrame(rows)
