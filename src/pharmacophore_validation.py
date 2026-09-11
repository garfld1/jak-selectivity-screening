#!/usr/bin/env python3
"""
Independent Ki-set validation for frozen JAK2-selectivity pharmacophores.

IMPORTANT:
    This script DOES NOT discover or optimize pharmacophores.
    It applies a frozen candidate list produced by pharmacophore_pipeline.py
    to the independent Ki validation set.

Primary original endpoint:
    JAK2-selective vs EVERYTHING NOT JAK2-selective.

Secondary endpoint:
    Continuous delta_pKi = pKi(JAK2) - pKi(JAK1)
    and Mann-Whitney comparison of pattern-positive vs pattern-negative ligands.
"""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu


NON_INTERACTION_COLUMNS = {"ligand_id", "SMILES", "smiles", "delta_pIC50", "vina_score"}

EXPECTED_METADATA_COLUMNS = {"ligand_id", "SMILES", "smiles", "delta_pIC50", "vina_score"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate frozen JAK pharmacophores on independent Ki data.")
    p.add_argument("--jak1", default="results/plip_results_annotated_validation_all/JAK1_docking_results_validation.csv")
    p.add_argument("--jak2", default="results/plip_results_annotated_validation_all/JAK2_docking_results_validation.csv")
    p.add_argument("--ki", default="docking/docking_prep/validation_set_pdbqt.csv")
    p.add_argument("--candidates", default="results/pharmacophore_analysis/discovered_candidates.csv")
    p.add_argument("--outdir", default="results/pharmacophore_analysis/pharmacophore validation")
    p.add_argument("--threshold-fold", type=float, default=5.0)
    p.add_argument("--max-candidates", type=int, default=0,
                   help="0 = all frozen candidates; otherwise evaluate first N rows of the candidate file.")
    return p.parse_args()



def residue_columns(df: pd.DataFrame, label: str) -> list[str]:
    missing = EXPECTED_METADATA_COLUMNS - set(df.columns)
    # Only ligand_id is truly mandatory; SMILES/smiles/delta/vina vary across exports.
    if "ligand_id" not in df.columns:
        raise ValueError(f"{label} is missing ligand_id")
    cols = [c for c in df.columns if c not in NON_INTERACTION_COLUMNS]
    if not cols:
        raise ValueError(f"{label} contains no residue-interaction columns")
    return cols


def feature_residue(feature: str) -> str:
    parts = feature.split(":")
    if len(parts) < 2:
        return feature
    return f"{parts[0]}:{parts[1]}"


def split_cell_interactions(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def feature_name(jak: str, residue: str, interaction: str | None = None) -> str:
    return f"{jak}:{residue}" if interaction is None else f"{jak}:{residue}:{interaction}"


def build_feature_sets(df: pd.DataFrame, jak: str) -> list[set[str]]:
    residue_cols = residue_columns(df, f"{jak} PLIP")
    sets = []
    for _, row in df.iterrows():
        feats = set()
        for residue in residue_cols:
            interactions = split_cell_interactions(row[residue])
            if not interactions:
                continue
            feats.add(feature_name(jak, residue))
            for interaction in interactions:
                feats.add(feature_name(jak, residue, interaction))
        sets.append(feats)
    return sets


def load_validation(jak1_path: str, jak2_path: str, ki_path: str) -> pd.DataFrame:
    j1 = pd.read_csv(jak1_path)
    j2 = pd.read_csv(jak2_path)
    ki = pd.read_csv(ki_path)

    for label, df in [("JAK1 PLIP", j1), ("JAK2 PLIP", j2)]:
        if "ligand_id" not in df.columns:
            raise ValueError(f"{label} is missing ligand_id")
    required_ki = {"ligand_id", "JAK1_Ki_nM", "JAK2_Ki_nM"}
    if not required_ki.issubset(ki.columns):
        raise ValueError(f"Ki file missing: {sorted(required_ki - set(ki.columns))}")

    for label, df in [("JAK1 PLIP", j1), ("JAK2 PLIP", j2), ("Ki", ki)]:
        if df["ligand_id"].duplicated().any():
            raise ValueError(f"{label} has duplicated ligand_id values.")

    common = sorted(set(j1.ligand_id) & set(j2.ligand_id) & set(ki.ligand_id))
    if len(common) != len(j1) or len(common) != len(j2) or len(common) != len(ki):
        raise ValueError(
            f"Validation IDs are not identical: JAK1={len(j1)}, JAK2={len(j2)}, Ki={len(ki)}, common={len(common)}"
        )

    j1 = j1.set_index("ligand_id").loc[common].reset_index()
    j2 = j2.set_index("ligand_id").loc[common].reset_index()
    ki = ki.set_index("ligand_id").loc[common].reset_index()

    j1_feats = build_feature_sets(j1, "J1")
    j2_feats = build_feature_sets(j2, "J2")

    out = pd.DataFrame({
        "ligand_id": common,
        "smiles": ki["smiles"].to_numpy() if "smiles" in ki.columns else j1.get("SMILES", pd.Series([None] * len(ki))).to_numpy(),
        "JAK1_Ki_nM": pd.to_numeric(ki["JAK1_Ki_nM"], errors="coerce"),
        "JAK2_Ki_nM": pd.to_numeric(ki["JAK2_Ki_nM"], errors="coerce"),
        "feature_set": [a | b for a, b in zip(j1_feats, j2_feats)],
    })

    if (out["JAK1_Ki_nM"] <= 0).any() or (out["JAK2_Ki_nM"] <= 0).any():
        bad = out.loc[(out["JAK1_Ki_nM"] <= 0) | (out["JAK2_Ki_nM"] <= 0), "ligand_id"].head(10).tolist()
        raise ValueError(f"Ki values must be positive nM values. Bad ligand IDs: {bad}")

    # delta_pKi = log10(Ki_JAK1 / Ki_JAK2), exactly equivalent to pKi_JAK2 - pKi_JAK1.
    out["delta_pKi"] = np.log10(out["JAK1_Ki_nM"] / out["JAK2_Ki_nM"])
    return out


def assign_classes(delta: pd.Series, threshold: float) -> pd.Series:
    return pd.Series(
        np.where(delta >= threshold, "JAK2_SELECTIVE",
                 np.where(delta <= -threshold, "JAK1_SELECTIVE", "NONSELECTIVE")),
        index=delta.index,
        dtype="string",
    )


def pattern_mask(df: pd.DataFrame, pattern: list[str]) -> pd.Series:
    return pd.Series(
        [all(f in s for f in pattern) for s in df["feature_set"]],
        index=df.index,
        dtype=bool,
    )


def stats_for_pattern(mask: pd.Series, classes: pd.Series, delta: pd.Series) -> dict:
    mask = mask.astype(bool)
    pos = classes.eq("JAK2_SELECTIVE")
    neg = ~pos
    a = int((mask & pos).sum())
    b = int((mask & neg).sum())
    c = int((~mask & pos).sum())
    d = int((~mask & neg).sum())

    odds_ratio, fisher_p = fisher_exact([[a, b], [c, d]], alternative="two-sided")

    n = a + b
    baseline = pos.mean()
    ppv = a / n if n else np.nan
    recall = a / int(pos.sum()) if pos.sum() else np.nan
    enrichment = ppv / baseline if baseline and not np.isnan(ppv) else np.nan

    pos_values = delta[mask]
    neg_values = delta[~mask]
    if len(pos_values) and len(neg_values):
        mw_u, mw_p = mannwhitneyu(pos_values, neg_values, alternative="two-sided")
        mean_shift = float(pos_values.mean() - neg_values.mean())
        median_shift = float(pos_values.median() - neg_values.median())
    else:
        mw_u, mw_p, mean_shift, median_shift = np.nan, np.nan, np.nan, np.nan

    j2 = int((mask & classes.eq("JAK2_SELECTIVE")).sum())
    j1 = int((mask & classes.eq("JAK1_SELECTIVE")).sum())
    ns = int((mask & classes.eq("NONSELECTIVE")).sum())

    return {
        "n": int(mask.sum()),
        "n_jak2": j2,
        "n_jak1": j1,
        "n_nonselective": ns,
        "n_not_jak2": b,
        "ppv_jak2": ppv,
        "recall_jak2": recall,
        "baseline_jak2_fraction": float(baseline),
        "jak2_enrichment": enrichment,
        "odds_ratio": float(odds_ratio),
        "fisher_p": float(fisher_p),
        "mean_delta_pKi_pattern": float(pos_values.mean()) if len(pos_values) else np.nan,
        "mean_delta_pKi_nonpattern": float(neg_values.mean()) if len(neg_values) else np.nan,
        "mean_delta_pKi_shift": mean_shift,
        "median_delta_pKi_shift": median_shift,
        "mannwhitney_u": float(mw_u) if np.isfinite(mw_u) else np.nan,
        "mannwhitney_p": float(mw_p) if np.isfinite(mw_p) else np.nan,
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates)
    required_candidate_cols = {"pattern", "features", "n_features"}
    if not required_candidate_cols.issubset(candidates.columns):
        raise ValueError(f"Candidate file missing: {sorted(required_candidate_cols - set(candidates.columns))}")

    if args.max_candidates > 0:
        candidates = candidates.head(args.max_candidates).copy()

    val = load_validation(args.jak1, args.jak2, args.ki)
    threshold = math.log10(args.threshold_fold)
    val["selectivity_class"] = assign_classes(val["delta_pKi"], threshold)

    rows = []
    for _, cand in candidates.iterrows():
        pattern = [f.strip() for f in str(cand["features"]).split("|") if f.strip()]
        residues = [feature_residue(f) for f in pattern]
        if len(residues) != len(set(residues)):
            raise ValueError(f"Frozen candidate contains redundant same-residue features: {cand['pattern']}")
        mask = pattern_mask(val, pattern)
        stats = stats_for_pattern(mask, val["selectivity_class"], val["delta_pKi"])
        rows.append({
            "pattern": cand["pattern"],
            "features": cand["features"],
            "n_features": int(cand["n_features"]),
            **stats,
        })

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(
            ["jak2_enrichment", "n_jak2", "n"], ascending=[False, False, False]
        )
    results.to_csv(outdir / "independent_validation_results.csv", index=False)

    val_export = val.drop(columns=["feature_set"])
    val_export.to_csv(outdir / "validation_labeled.csv", index=False)

    metadata = {
        "n_validation_ligands": len(val),
        "threshold_fold": args.threshold_fold,
        "threshold_delta_pKi": threshold,
        "n_candidates_evaluated": len(candidates),
        "class_counts": val["selectivity_class"].value_counts().to_dict(),
        "primary_endpoint": "JAK2-selective vs everything not JAK2-selective",
        "delta_pKi_definition": "log10(JAK1_Ki_nM / JAK2_Ki_nM)",
    }
    with open(outdir / "validation_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Saved independent validation results to: {outdir}")
    print(f"Validation ligands: {len(val):,}")
    print(f"Candidates evaluated: {len(candidates):,}")
    print(f"Class counts: {val['selectivity_class'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
