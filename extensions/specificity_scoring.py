#!/usr/bin/env python3
"""Score frozen JAK2 candidates against JAK1-selective and nonselective classes."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
if __package__ is None: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.common import EXTENSIONS, feature_set_mask, parse_features, paths, target_vs_rest_stats

SPECIFICITY_THRESHOLD = 1.5  # Addition; not part of the original discovery method.

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default=paths()["matrix"])
    p.add_argument("--candidates", default=paths()["candidates"])
    p.add_argument("--out", default=EXTENSIONS / "specificity_results.csv")
    args = p.parse_args()
    matrix, candidates = pd.read_csv(args.matrix), pd.read_csv(args.candidates)
    classes = matrix.selectivity_class.astype("string")
    rows = []
    for _, candidate in candidates.iterrows():
        mask = feature_set_mask(matrix, parse_features(candidate.features))
        j2 = target_vs_rest_stats(mask, classes, "JAK2_SELECTIVE")
        j1 = target_vs_rest_stats(mask, classes, "JAK1_SELECTIVE")
        ns = target_vs_rest_stats(mask, classes, "NONSELECTIVE")
        score = j2["jak2_enrichment"] / max(j1["enrichment"], ns["enrichment"], 0.5)
        rows.append({"features": candidate.features, "specificity_jak2_enrichment": j2["jak2_enrichment"],
                     "jak1_enrichment": j1["enrichment"], "nonselective_enrichment": ns["enrichment"],
                     "specificity_score": score, "passes_specificity_threshold": score >= SPECIFICITY_THRESHOLD})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} specificity scores to {args.out}")

if __name__ == "__main__": main()
