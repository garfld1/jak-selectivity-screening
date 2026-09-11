#!/usr/bin/env python3
"""
JAK1/JAK2 hybrid pharmacophore discovery pipeline.

Original endpoint:
    JAK2-selective vs EVERYTHING NOT JAK2-selective
    (i.e. JAK1-selective + nonselective are both background)

Workflow:
    1. Load and align JAK1/JAK2 PLIP result tables by ligand_id.
    2. Build hybrid features:
         - binary residue-contact: J2:TYR931
         - exact interaction:       J2:TYR931:hydrogen_bond
    3. Assign 5x selectivity classes from delta_pIC50.
    4. Stratified 70/30 discovery/holdout split.
    5. Discover single features.
    6. Exhaustively search 2-feature AND cores.
    7. Beam-search 3..max_features feature AND cores.
    8. Re-score frozen candidates on the holdout set.
    9. Save feature matrix, discovery results, and holdout results.

Notes:
    - Row order is never trusted; ligand_id is the join key.
    - A ligand with "hydrogen_bond;pi_cation" creates BOTH exact features.
    - Binary and exact features coexist in the same feature pool.
    - Specific exact features imply their corresponding binary residue feature,
      but they remain separate Boolean columns.

Baseline convention:
    - Discovery-side stats (singles/pairs/beam) use the discovery-set JAK2
      baseline fraction, fixed once per discover() call, so enrichment values
      within the discovery search are directly comparable to one another.
    - Holdout-side stats use the holdout-set's OWN JAK2 baseline fraction
      (computed fresh from the holdout subset), so jak2_enrichment on holdout
      means "enrichment relative to actual holdout composition." The
      corresponding discovery-side numbers (discovery_n, discovery_jak2_enrichment,
      discovery_fisher_p, discovery_fdr_q) are merged in alongside for direct
      side-by-side comparison, rather than being blended into one baseline.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split


SELECTIVITY_THRESHOLD = math.log10(5.0)
DEFAULT_MIN_SUPPORT = 50
DEFAULT_MIN_JAK2 = 10
DEFAULT_MIN_ENRICHMENT = 1.5
DEFAULT_BEAM_WIDTH = 150
DEFAULT_MAX_FEATURES = 5
RANDOM_STATE = 42

NON_INTERACTION_COLUMNS = {"ligand_id", "SMILES", "smiles", "delta_pIC50", "vina_score"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discover hybrid JAK2-selectivity pharmacophores.")
    p.add_argument("--jak1", default="results/plip_results_annotated_training_all/JAK1_docking_results.csv")
    p.add_argument("--jak2", default="results/plip_results_annotated_training_all/JAK2_docking_results.csv")
    p.add_argument("--outdir", default="results/pharmacophore_analysis")
    p.add_argument("--threshold-fold", type=float, default=5.0,
                   help="Fold-selectivity cutoff; default 5x.")
    p.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    p.add_argument("--min-jak2", type=int, default=DEFAULT_MIN_JAK2)
    p.add_argument("--min-enrichment", type=float, default=DEFAULT_MIN_ENRICHMENT)
    p.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    p.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    p.add_argument("--holdout-size", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--skip-beam", action="store_true", help="Only run singles + exhaustive pairs.")
    p.add_argument("--fdr-alpha", type=float, default=0.05, help="BH-FDR target level used for discovery reporting.")
    return p.parse_args()


def canonical_feature_name(jak: str, residue: str, interaction: str | None = None) -> str:
    return f"{jak}:{residue}" if interaction is None else f"{jak}:{residue}:{interaction}"


def residue_columns(df: pd.DataFrame, label: str) -> list[str]:
    """Return validated residue columns; PLIP residue columns must look like e.g. TYR931."""
    if "ligand_id" not in df.columns:
        raise ValueError(f"{label} is missing ligand_id")
    cols = [c for c in df.columns if c not in NON_INTERACTION_COLUMNS]
    bad = [c for c in cols if not re.fullmatch(r"[A-Z][A-Z0-9]*[0-9]+", str(c))]
    if bad:
        raise ValueError(
            f"{label} has unexpected non-residue columns that would otherwise be parsed as features: {bad[:10]}"
        )
    if not cols:
        raise ValueError(f"{label} contains no residue-interaction columns")
    return cols


def split_cell_interactions(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def build_feature_sets(df: pd.DataFrame, jak_label: str) -> tuple[list[set[str]], list[str]]:
    """Return one feature-set per ligand and the sorted union of feature names."""
    residue_cols = residue_columns(df, jak_label)
    row_features: list[set[str]] = []
    all_features: set[str] = set()

    for _, row in df.iterrows():
        feats: set[str] = set()
        for residue in residue_cols:
            interactions = split_cell_interactions(row[residue])
            if not interactions:
                continue
            feats.add(canonical_feature_name(jak_label, residue))
            for interaction in interactions:
                feats.add(canonical_feature_name(jak_label, residue, interaction))
        row_features.append(feats)
        all_features.update(feats)

    return row_features, sorted(all_features)


def load_and_align(jak1_path: str, jak2_path: str) -> pd.DataFrame:
    j1 = pd.read_csv(jak1_path)
    j2 = pd.read_csv(jak2_path)

    required = {"ligand_id", "delta_pIC50"}
    for label, df in [("JAK1", j1), ("JAK2", j2)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{label} file missing required columns: {sorted(missing)}")

    if j1["ligand_id"].duplicated().any() or j2["ligand_id"].duplicated().any():
        raise ValueError("ligand_id must be unique in each PLIP result file.")

    common = sorted(set(j1.ligand_id) & set(j2.ligand_id))
    if len(common) != len(j1) or len(common) != len(j2):
        raise ValueError(
            f"JAK1/JAK2 ligand IDs are not one-to-one: JAK1={len(j1)}, JAK2={len(j2)}, common={len(common)}"
        )

    j1 = j1.set_index("ligand_id").loc[common].reset_index()
    j2 = j2.set_index("ligand_id").loc[common].reset_index()

    # Preserve one canonical selectivity value. Check that both tables agree.
    merged = j1[["ligand_id", "delta_pIC50"]].merge(
        j2[["ligand_id", "delta_pIC50"]].rename(columns={"delta_pIC50": "delta_pIC50_j2"}),
        on="ligand_id", how="inner"
    )
    mismatch = ~np.isclose(merged["delta_pIC50"], merged["delta_pIC50_j2"], equal_nan=True)
    if mismatch.any():
        bad = merged.loc[mismatch, "ligand_id"].head(10).tolist()
        raise ValueError(f"delta_pIC50 mismatch between JAK1/JAK2 files for ligand IDs: {bad}")

    j1_features, f1 = build_feature_sets(j1, "J1")
    j2_features, f2 = build_feature_sets(j2, "J2")

    feature_sets = [a | b for a, b in zip(j1_features, j2_features)]
    features = sorted(set(f1) | set(f2))

    out = pd.DataFrame({
        "ligand_id": common,
        "delta_pIC50": merged["delta_pIC50"].to_numpy(),
        "feature_set": feature_sets,
    })
    return out, features


def assign_classes(delta: pd.Series, threshold: float) -> pd.Series:
    return pd.Series(
        np.where(delta >= threshold, "JAK2_SELECTIVE",
                 np.where(delta <= -threshold, "JAK1_SELECTIVE", "NONSELECTIVE")),
        index=delta.index,
        dtype="string",
    )


def materialize_features(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    feature_sets = df["feature_set"].tolist()
    # Build columns in one operation to avoid pandas DataFrame fragmentation.
    return pd.DataFrame(
        {feat: [feat in s for s in feature_sets] for feat in features},
        index=df.index,
    )


def j2_vs_background_stats(mask: pd.Series, classes: pd.Series, baseline_j2: float | None = None) -> dict:
    """Original method: JAK2-selective vs everything else."""
    mask = pd.Series(mask, index=classes.index).astype(bool)
    pos = classes.eq("JAK2_SELECTIVE")
    a = int((mask & pos).sum())
    b = int((mask & ~pos).sum())
    c = int((~mask & pos).sum())
    d = int((~mask & ~pos).sum())
    n = a + b
    total = a + b + c + d

    table = [[a, b], [c, d]]
    try:
        odds_ratio, fisher_p = fisher_exact(table, alternative="two-sided")
    except Exception:
        odds_ratio, fisher_p = np.nan, np.nan

    ppv = a / n if n else np.nan
    recall = a / (a + c) if (a + c) else np.nan
    # Baseline is always caller-supplied and fixed for the duration of the caller's
    # loop (see discover() and evaluate_patterns()) so it never gets silently
    # recomputed from whatever local subset `mask` happens to describe.
    baseline = float(baseline_j2) if baseline_j2 is not None else ((a + c) / total if total else np.nan)
    enrichment = ppv / baseline if baseline and not np.isnan(ppv) else np.nan

    return {
        "n": n,
        "n_jak2": a,
        "n_not_jak2": b,
        "n_jak1": int((mask & classes.eq("JAK1_SELECTIVE")).sum()),
        "n_nonselective": int((mask & classes.eq("NONSELECTIVE")).sum()),
        "ppv_jak2": ppv,
        "recall_jak2": recall,
        "baseline_jak2_fraction": baseline,
        "jak2_enrichment": enrichment,
        "odds_ratio": odds_ratio,
        "fisher_p": fisher_p,
    }


def feature_set_mask(feature_matrix: pd.DataFrame, pattern: Sequence[str]) -> pd.Series:
    if not pattern:
        return pd.Series(True, index=feature_matrix.index)
    return feature_matrix[list(pattern)].all(axis=1)


def valid_for_reporting(stats: dict, min_support: int, min_jak2: int, min_enrichment: float) -> bool:
    return (
        stats["n"] >= min_support
        and stats["n_jak2"] >= min_jak2
        and np.isfinite(stats["jak2_enrichment"])
        and stats["jak2_enrichment"] >= min_enrichment
    )


def pattern_to_str(pattern: Sequence[str]) -> str:
    return " AND ".join(pattern)


def score_for_beam(stats: dict) -> float:
    """Ranking score emphasizing JAK2 odds ratio while mildly rewarding support."""
    or_value = stats["odds_ratio"]
    n = stats["n"]
    if not np.isfinite(or_value) or or_value <= 0 or n <= 0:
        return -np.inf
    return math.log(or_value) * math.sqrt(n)


def summarize_patterns(
    patterns: Iterable[tuple[str, ...]],
    fm: pd.DataFrame,
    classes: pd.Series,
    source: str,
    baseline_j2: float,
    **extra,
) -> pd.DataFrame:
    rows = []
    for pattern in patterns:
        stats = j2_vs_background_stats(feature_set_mask(fm, pattern), classes, baseline_j2=baseline_j2)
        rows.append({
            "pattern": pattern_to_str(pattern),
            "n_features": len(pattern),
            "features": "|".join(pattern),
            "source": source,
            **extra,
            **stats,
            "beam_score": score_for_beam(stats),
        })
    return pd.DataFrame(rows)


def add_bh_fdr(results: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Add Benjamini-Hochberg q-values to a discovery result table."""
    if results.empty:
        results["fdr_q"] = pd.Series(dtype=float)
        results["passes_fdr"] = pd.Series(dtype=bool)
        return results
    pvals = pd.to_numeric(results["fisher_p"], errors="coerce").to_numpy()
    valid = np.isfinite(pvals)
    qvals = np.full(len(results), np.nan, dtype=float)
    rejected = np.zeros(len(results), dtype=bool)
    if valid.any():
        reject, q, _, _ = multipletests(pvals[valid], alpha=alpha, method="fdr_bh")
        qvals[valid] = q
        rejected[valid] = reject
    results = results.copy()
    results["fdr_q"] = qvals
    results["passes_fdr"] = rejected
    return results


def discover(
    df: pd.DataFrame,
    features: list[str],
    discovery_idx: pd.Index,
    min_support: int,
    min_jak2: int,
    min_enrichment: float,
    beam_width: int,
    max_features: int,
    skip_beam: bool,
    fdr_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    disc = df.loc[discovery_idx].copy()
    classes = disc["selectivity_class"]
    fm = materialize_features(disc, features)
    baseline_j2 = float(classes.eq("JAK2_SELECTIVE").mean())

    # Singles are all tested hypotheses in this family.
    singles = summarize_patterns(
        ((f,) for f in features), fm, classes, "single", baseline_j2=baseline_j2
    )
    singles["passes_reporting_filter"] = (
        (singles["n"] >= min_support) &
        (singles["n_jak2"] >= min_jak2) &
        (singles["jak2_enrichment"] >= min_enrichment)
    )

    # Feature seeds for the practical beam search. This is deliberately heuristic.
    seed_features = singles.loc[
        singles["passes_reporting_filter"], "features"
    ].tolist()

    # Exhaustive pairs, excluding redundant same-residue combinations.
    pair_patterns = []
    for a, b in itertools.combinations(features, 2):
        if feature_residue(a) == feature_residue(b):
            continue
        pair_patterns.append(tuple(sorted((a, b))))
    pairs = summarize_patterns(
        pair_patterns, fm, classes, "pair", baseline_j2=baseline_j2
    )
    pairs["passes_reporting_filter"] = (
        (pairs["n"] >= min_support) &
        (pairs["n_jak2"] >= min_jak2) &
        (pairs["jak2_enrichment"] >= min_enrichment)
    )

    # All pair hypotheses seed the beam only after filtering, matching the original-style search.
    accepted_pairs = pairs.loc[pairs["passes_reporting_filter"]].copy()

    tested_multi = [pairs.copy()]
    beam_kept = []
    if not skip_beam and max_features >= 3:
        current = [tuple(x.split("|")) for x in accepted_pairs["features"].head(beam_width)]
        used_residues = {p: {feature_residue(x) for x in p} for p in current}

        for k in range(3, max_features + 1):
            candidate_patterns = set()
            for pat in current:
                for f in features:
                    if f in pat:
                        continue
                    if feature_residue(f) in used_residues[pat]:
                        continue
                    candidate_patterns.add(tuple(sorted(pat + (f,))))

            if not candidate_patterns:
                break

            tested = summarize_patterns(
                candidate_patterns, fm, classes, f"beam_{k}", baseline_j2=baseline_j2
            )
            tested["passes_reporting_filter"] = (
                (tested["n"] >= min_support) &
                (tested["n_jak2"] >= min_jak2) &
                (tested["jak2_enrichment"] >= min_enrichment)
            )
            tested_multi.append(tested.copy())

            ranked = tested.loc[
                (tested["n"] >= min_support) &
                (tested["n_jak2"] >= min_jak2)
            ].sort_values("beam_score", ascending=False).head(beam_width)
            if ranked.empty:
                break
            beam_kept.append(ranked.copy())
            current = [tuple(x.split("|")) for x in ranked["features"]]
            used_residues = {p: {feature_residue(x) for x in p} for p in current}

    # IMPORTANT: BH-FDR is calculated once across every unique hypothesis actually tested
    # in this discovery search (singles + exhaustive pairs + every pre-truncation beam level).
    all_tested = pd.concat([singles.copy(), *tested_multi], ignore_index=True)
    all_tested = all_tested.drop_duplicates(subset=["features"], keep="first")
    all_tested = add_bh_fdr(all_tested, alpha=fdr_alpha)

    # Split the corrected audit table back into the components users may want to inspect.
    single_corrected = all_tested.loc[all_tested["n_features"] == 1].copy()
    multi_corrected = all_tested.loc[all_tested["n_features"] >= 2].copy()

    # Candidate pool is intentionally based on discovery support/enrichment PLUS FDR.
    candidates = all_tested.loc[
        all_tested["passes_reporting_filter"].fillna(False) &
        all_tested["passes_fdr"].fillna(False)
    ].copy()
    candidates = candidates.sort_values(
        ["n_features", "jak2_enrichment", "n_jak2", "n"],
        ascending=[True, False, False, False]
    ).drop_duplicates(subset=["features"])

    # Restore convenient ordering for audit outputs.
    single_corrected = single_corrected.sort_values(
        ["jak2_enrichment", "n_jak2", "n"], ascending=[False, False, False]
    )
    multi_corrected = multi_corrected.sort_values(
        ["n_features", "jak2_enrichment", "n_jak2", "n"], ascending=[True, False, False, False]
    )

    return single_corrected, candidates


def feature_residue(feature: str) -> str:
    parts = feature.split(":")
    if len(parts) < 2:
        return feature
    return f"{parts[0]}:{parts[1]}"


def evaluate_patterns(
    pattern_df: pd.DataFrame,
    df: pd.DataFrame,
    subset_name: str,
    seed: int,
    fm_cache: pd.DataFrame | None = None,
    baseline_j2: float | None = None,
) -> pd.DataFrame:
    """Re-score frozen candidate patterns on an evaluation subset (e.g. holdout).

    Baseline convention (see module docstring): by default this uses the
    evaluation subset's OWN JAK2 baseline fraction, computed fresh from `df`,
    so the reported jak2_enrichment is "enrichment relative to this subset's
    actual composition." Pass an explicit `baseline_j2` if you instead want
    enrichment expressed relative to some other (e.g. discovery-set) baseline;
    the two should be close under stratified splitting but are not guaranteed
    to be identical, and mixing them silently changes what the number means.
    """
    if pattern_df.empty:
        return pd.DataFrame()
    all_features = sorted({f for s in pattern_df["features"] for f in s.split("|")})
    fm = fm_cache if fm_cache is not None else materialize_features(df, all_features)
    classes = df["selectivity_class"]
    if baseline_j2 is None:
        baseline_j2 = float(classes.eq("JAK2_SELECTIVE").mean())
    rows = []
    for _, row in pattern_df.iterrows():
        pattern = tuple(row["features"].split("|"))
        stats = j2_vs_background_stats(feature_set_mask(fm, pattern), classes, baseline_j2=baseline_j2)
        rows.append({
            "pattern": row["pattern"],
            "features": row["features"],
            "n_features": int(row["n_features"]),
            "subset": subset_name,
            "discovery_seed": seed,
            **stats,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    threshold = math.log10(args.threshold_fold)
    df, features = load_and_align(args.jak1, args.jak2)
    df["selectivity_class"] = assign_classes(df["delta_pIC50"], threshold)

    if len(df) < 10:
        raise ValueError("Dataset is unexpectedly small.")

    discovery_idx, holdout_idx = train_test_split(
        df.index,
        test_size=args.holdout_size,
        random_state=args.seed,
        stratify=df["selectivity_class"],
    )

    feature_matrix = materialize_features(df, features)
    export = pd.concat([
        df.drop(columns=["feature_set"]),
        feature_matrix.astype(np.int8)
    ], axis=1)
    export.to_csv(outdir / "feature_matrix.csv", index=False)

    singles, candidates = discover(
        df=df,
        features=features,
        discovery_idx=discovery_idx,
        min_support=args.min_support,
        min_jak2=args.min_jak2,
        min_enrichment=args.min_enrichment,
        beam_width=args.beam_width,
        max_features=args.max_features,
        skip_beam=args.skip_beam,
        fdr_alpha=args.fdr_alpha,
    )

    # Discovery results for auditability.
    singles.to_csv(outdir / "single_feature_discovery.csv", index=False)
    candidates.to_csv(outdir / "discovered_candidates.csv", index=False)

    # Holdout evaluates the frozen discovery candidates, without re-optimizing them.
    holdout = df.loc[holdout_idx].copy()

    # Rebuild a compact matrix only for features used by candidates.
    candidate_features = sorted({f for s in candidates["features"] for f in s.split("|")}) if not candidates.empty else []
    holdout_fm = materialize_features(holdout, candidate_features) if candidate_features else pd.DataFrame(index=holdout.index)

    # Holdout stats use the holdout subset's OWN JAK2 baseline fraction (the default
    # behavior of evaluate_patterns when baseline_j2 is omitted). This means
    # jak2_enrichment here is "enrichment relative to actual holdout composition,"
    # not the discovery-set baseline. The discovery-side numbers are merged in
    # below under discovery_* columns for direct side-by-side comparison.
    holdout_results = evaluate_patterns(candidates, holdout, "holdout", args.seed, holdout_fm)
    if not holdout_results.empty:
        discovery_lookup = candidates[["features", "n", "jak2_enrichment", "fisher_p", "fdr_q"]].rename(
            columns={
                "n": "discovery_n",
                "jak2_enrichment": "discovery_jak2_enrichment",
                "fisher_p": "discovery_fisher_p",
                "fdr_q": "discovery_fdr_q",
            }
        )
        holdout_results = holdout_results.merge(discovery_lookup, on="features", how="left", validate="one_to_one")
        holdout_results = holdout_results.sort_values(
            ["jak2_enrichment", "n_jak2", "n"], ascending=[False, False, False]
        )
    holdout_results.to_csv(outdir / "holdout_results.csv", index=False)

    metadata = {
        "jak1_path": os.path.abspath(args.jak1),
        "jak2_path": os.path.abspath(args.jak2),
        "n_ligands": len(df),
        "n_features": len(features),
        "threshold_fold": args.threshold_fold,
        "threshold_delta_pIC50": threshold,
        "min_support": args.min_support,
        "min_jak2": args.min_jak2,
        "min_enrichment": args.min_enrichment,
        "beam_width": args.beam_width,
        "max_features": args.max_features,
        "holdout_size": args.holdout_size,
        "seed": args.seed,
        "discovery_n": len(discovery_idx),
        "holdout_n": len(holdout_idx),
        "discovery_class_counts": df.loc[discovery_idx, "selectivity_class"].value_counts().to_dict(),
        "holdout_class_counts": df.loc[holdout_idx, "selectivity_class"].value_counts().to_dict(),
        "candidate_count": len(candidates),
        "discovery_baseline_jak2_fraction": float(df.loc[discovery_idx, "selectivity_class"].eq("JAK2_SELECTIVE").mean()),
        "holdout_baseline_jak2_fraction": float(df.loc[holdout_idx, "selectivity_class"].eq("JAK2_SELECTIVE").mean()),
        "fdr_method": "Benjamini-Hochberg",
        "fdr_alpha": args.fdr_alpha,
    }
    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Saved results to: {outdir}")
    print(f"Ligands: {len(df):,}; features: {len(features):,}")
    print(f"Discovery: {len(discovery_idx):,}; holdout: {len(holdout_idx):,}")
    print(f"Candidates passing discovery filter: {len(candidates):,}")


if __name__ == "__main__":
    main()