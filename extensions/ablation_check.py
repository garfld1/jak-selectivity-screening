#!/usr/bin/env python3
"""Ablate one feature at a time from small, cross-check survivor patterns."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
if __package__ is None: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.common import EXTENSIONS, feature_set_mask, j2_vs_background_stats, parse_features, paths

# Addition: every (k-1) subset must retain >=50% of full-pattern JAK2 support and
# must not exceed the full enrichment by more than 10% (near-monotonic increase).
SUPPORT_RETAINED = 0.50
ENRICHMENT_TOLERANCE = 1.10

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default=paths()["matrix"]); p.add_argument("--survivors",
                   help="Optional small CSV with a features column. If omitted, construct it from prior extension results.")
    p.add_argument("--candidates", default=paths()["candidates"]); p.add_argument("--holdout", default=paths()["holdout"])
    p.add_argument("--scaffold", default=EXTENSIONS / "scaffold_holdout_results.csv"); p.add_argument("--specificity", default=EXTENSIONS / "specificity_results.csv")
    p.add_argument("--out", default=EXTENSIONS / "ablation_flags.csv")
    args = p.parse_args(); matrix = pd.read_csv(args.matrix)
    if args.survivors:
        survivors = pd.read_csv(args.survivors)
    else:
        discovered, holdout = pd.read_csv(args.candidates), pd.read_csv(args.holdout)
        scaffold, specificity = pd.read_csv(args.scaffold), pd.read_csv(args.specificity)
        survivors = discovered.merge(holdout[["features", "jak2_enrichment", "fisher_p"]], on="features", suffixes=("", "_holdout"))
        survivors = survivors.merge(scaffold[["features", "scaffold_holdout_jak2_enrichment", "scaffold_holdout_fisher_p"]], on="features")
        survivors = survivors.merge(specificity[["features", "specificity_score"]], on="features")
        survivors = survivors.loc[(survivors.passes_fdr) & (survivors.jak2_enrichment_holdout >= 1.5) &
                                  (survivors.fisher_p_holdout < .05) & (survivors.scaffold_holdout_jak2_enrichment >= 1.5) &
                                  (survivors.scaffold_holdout_fisher_p < .05) & (survivors.specificity_score >= 1.5)].copy()
        survivors.to_csv(EXTENSIONS / "pre_ablation_survivors.csv", index=False)
    classes, baseline = matrix.selectivity_class.astype("string"), matrix.selectivity_class.eq("JAK2_SELECTIVE").mean()
    rows = []
    for _, survivor in survivors.iterrows():
        features = parse_features(survivor.features)
        if len(features) < 2: continue
        full = j2_vs_background_stats(feature_set_mask(matrix, features), classes, baseline)
        subsets = [tuple(f for j, f in enumerate(features) if j != i) for i in range(len(features))]
        stats = [j2_vs_background_stats(feature_set_mask(matrix, subset), classes, baseline) for subset in subsets]
        stable = all(s["n_jak2"] >= full["n_jak2"] * SUPPORT_RETAINED and
                     s["jak2_enrichment"] <= full["jak2_enrichment"] * ENRICHMENT_TOLERANCE for s in stats)
        rows.append({"features": survivor.features, "n_features": len(features), "full_n_jak2": full["n_jak2"],
                     "full_jak2_enrichment": full["jak2_enrichment"], "subset_n_jak2": "|".join(map(str, [s["n_jak2"] for s in stats])),
                     "subset_jak2_enrichment": "|".join(f"{s['jak2_enrichment']:.6g}" for s in stats),
                     "stable_under_ablation": stable})
    pd.DataFrame(rows).to_csv(args.out, index=False)

if __name__ == "__main__": main()
