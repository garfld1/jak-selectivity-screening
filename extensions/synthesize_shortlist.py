#!/usr/bin/env python3
"""Strict joins and a final, baseline-aware shortlist of JAK2 regions."""
from __future__ import annotations
import argparse
from collections import Counter
import sys
from pathlib import Path
import pandas as pd
if __package__ is None: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.common import EXTENSIONS, parse_features, paths

SPECIFICITY_THRESHOLD = 1.5  # Addition; not part of the original method.

def _rename(df, prefix, keep=("features",)):
    return df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c not in keep})

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default=paths()["candidates"]); p.add_argument("--holdout", default=paths()["holdout"])
    p.add_argument("--validation", default=paths()["validation"]); p.add_argument("--scaffold", default=EXTENSIONS / "scaffold_holdout_results.csv")
    p.add_argument("--specificity", default=EXTENSIONS / "specificity_results.csv"); p.add_argument("--ablation", default=EXTENSIONS / "ablation_flags.csv")
    p.add_argument("--jak1-candidates", default=EXTENSIONS / "jak1_candidates.csv",
                   help="Optional fresh JAK1-search list used only as a generic-binder red-flag cross-check.")
    p.add_argument("--out", default=EXTENSIONS / "final_shortlist.csv"); p.add_argument("--report", default=EXTENSIONS / "shortlist_report.md")
    args = p.parse_args()
    discovered = pd.read_csv(args.candidates); holdout = pd.read_csv(args.holdout); validation = pd.read_csv(args.validation)
    scaffold, specificity, ablation = map(pd.read_csv, (args.scaffold, args.specificity, args.ablation))
    table = discovered.merge(_rename(holdout, "random_holdout"), on="features", how="left", validate="one_to_one")
    table = table.merge(_rename(validation, "independent_validation"), on="features", how="left", validate="one_to_one")
    table = table.merge(scaffold, on="features", how="left", validate="one_to_one").merge(specificity, on="features", how="left", validate="one_to_one")
    table = table.merge(ablation[["features", "stable_under_ablation"]], on="features", how="left")
    jak1_path = Path(args.jak1_candidates)
    if jak1_path.exists():
        jak1_features = {f for value in pd.read_csv(jak1_path).features for f in parse_features(value)}
        table["jak1_feature_overlap"] = table.features.map(lambda value: "|".join(sorted(set(parse_features(value)) & jak1_features)))
        table["generic_binder_red_flag"] = table.jak1_feature_overlap.ne("")
    else:
        table["jak1_feature_overlap"] = ""
        table["generic_binder_red_flag"] = False
    keep = (table.passes_fdr.fillna(False) & (table.random_holdout_jak2_enrichment > 1) &
            (table.scaffold_holdout_jak2_enrichment > 1) & (table.independent_validation_fisher_p < .05) &
            (table.independent_validation_mannwhitney_p < .05) & (table.independent_validation_mean_delta_pKi_shift > 0) &
            (table.specificity_score >= SPECIFICITY_THRESHOLD) & table.stable_under_ablation.fillna(False))
    final = table.loc[keep].copy().sort_values("independent_validation_jak2_enrichment", ascending=False)
    final.to_csv(args.out, index=False)
    feature_counts = Counter(f for value in final.features for f in parse_features(value))
    recurrence = "\n".join(f"- {f}: {n}" for f, n in feature_counts.most_common()) or "- No candidates survived all filters."
    typed = sum(1 for f in feature_counts if f.count(":") >= 2); bare = len(feature_counts) - typed
    args.report.write_text(
        "# Final shortlist\n\n"
        f"{len(final)} patterns passed all pre-specified filters. Recurrence is ranked separately from any single pattern.\n\n"
        "## Feature/residue recurrence\n\n" + recurrence + "\n\n"
        f"Bare residue features: {bare}; typed interaction features: {typed}.\n\n"
        "## Interpretation\n\nPPV and enrichment are reported against each stage's own baseline in the CSV; magnitudes are not directly comparable across discovery, random holdout, scaffold holdout, and Ki validation. Ki validation also differs from IC50 discovery. A feature that also appears among JAK1-search hits is a generic-binder red flag and must be cross-checked against `specificity_score`, not treated as independent support.\n",
        encoding="utf-8")

if __name__ == "__main__": main()
