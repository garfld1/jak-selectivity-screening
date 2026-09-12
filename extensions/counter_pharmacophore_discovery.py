#!/usr/bin/env python3
"""Independent JAK1 enrichment and JAK2-depletion searches on the feature matrix."""
from __future__ import annotations
import argparse
import itertools
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests
if __package__ is None: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.common import EXTENSIONS, feature_columns, feature_set_mask, parse_features, paths, target_vs_rest_stats
from src.pharmacophore_discovery import feature_residue, materialize_features, score_for_beam
from src.pharmacophore_validation import assign_classes, load_validation

MIN_SUPPORT, MIN_TARGET, MIN_ENRICHMENT = 50, 10, 1.5
DEPLETION_CUTOFF = 1 / MIN_ENRICHMENT  # Addition: <= 0.6667, not in the original method.

def _discover(matrix, target, depleted=False, beam_width=150, max_features=5):
    """Fresh singles→pairs→beam search using original mask/ranking helpers."""
    features, classes = feature_columns(matrix), matrix.selectivity_class.astype("string")
    # Work from the already materialized Boolean matrix; frozen rescoring below
    # deliberately uses the imported feature_set_mask for identical semantics.
    values = matrix[features].to_numpy(dtype=bool)
    positions = {feature: i for i, feature in enumerate(features)}
    def stat(pattern):
        mask = values[:, [positions[f] for f in pattern]].all(axis=1)
        return target_vs_rest_stats(pd.Series(mask, index=matrix.index), classes, target)
    def acceptable(s):
        return s["n"] >= MIN_SUPPORT and s["n_target"] >= MIN_TARGET and np.isfinite(s["enrichment"]) and ((s["enrichment"] <= DEPLETION_CUTOFF) if depleted else (s["enrichment"] >= MIN_ENRICHMENT))
    tested = []
    def assess(patterns, source):
        rows=[]
        for pat in patterns:
            s=stat(pat); rows.append({"pattern":" AND ".join(pat), "features":"|".join(pat), "n_features":len(pat), "source":source, **s, "beam_score":score_for_beam({"odds_ratio": (1/s["odds_ratio"] if depleted and s["odds_ratio"] else s["odds_ratio"]), "n":s["n"]})})
        return pd.DataFrame(rows)
    singles=assess(((f,) for f in features), "single"); tested.append(singles)
    pairs=assess((tuple(sorted((a,b))) for a,b in itertools.combinations(features,2) if feature_residue(a)!=feature_residue(b)), "pair"); tested.append(pairs)
    current=[parse_features(x) for x in pairs.loc[pairs.apply(lambda r: acceptable(r), axis=1)].nlargest(beam_width,"beam_score").features]
    for k in range(3,max_features+1):
        patterns=set()
        for pat in current:
            residues={feature_residue(f) for f in pat}
            patterns.update(tuple(sorted((*pat,f))) for f in features if f not in pat and feature_residue(f) not in residues)
        level=assess(patterns, f"beam_{k}")
        if level.empty: break
        tested.append(level); current=[parse_features(x) for x in level[level.n >= MIN_SUPPORT].nlargest(beam_width,"beam_score").features]
        if not current: break
    all_tests=pd.concat(tested, ignore_index=True).drop_duplicates("features")
    all_tests["fdr_q"]=multipletests(all_tests.fisher_p.fillna(1), method="fdr_bh")[1]
    all_tests["passes_fdr"]=all_tests.fdr_q <= .05
    return all_tests.loc[all_tests.apply(lambda r: acceptable(r), axis=1) & all_tests.passes_fdr].sort_values(["n_features","enrichment"], ascending=[True,depleted]).copy()

def _rescore(candidates, subset, target, prefix):
    classes=subset.selectivity_class.astype("string"); rows=[]
    for _, row in candidates.iterrows():
        s=target_vs_rest_stats(feature_set_mask(subset, parse_features(row.features)), classes, target)
        rows.append({"features":row.features,"pattern":row.pattern,"n_features":row.n_features, **{f"{prefix}_{k}":v for k,v in s.items()}})
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--matrix",default=paths()["matrix"]); p.add_argument("--outdir",default=EXTENSIONS)
    p.add_argument("--validation-matrix", help="Optional independent feature-matrix export (must include labels/features).")
    p.add_argument("--validation-jak1", default="results/plip_results_annotated_validation_all/JAK1_docking_results_validation.csv")
    p.add_argument("--validation-jak2", default="results/plip_results_annotated_validation_all/JAK2_docking_results_validation.csv")
    p.add_argument("--validation-ki", default="docking/docking_prep/validation_set_pdbqt.csv")
    args=p.parse_args(); outdir=pd.io.common.stringify_path(args.outdir); outdir_path=__import__('pathlib').Path(outdir); outdir_path.mkdir(exist_ok=True)
    matrix=pd.read_csv(args.matrix); discovery_idx, holdout_idx=train_test_split(matrix.index,test_size=.30,random_state=42,stratify=matrix.selectivity_class)
    j1=_discover(matrix.loc[discovery_idx],"JAK1_SELECTIVE"); anti=_discover(matrix.loc[discovery_idx],"JAK2_SELECTIVE",depleted=True)
    j1.to_csv(outdir_path/"jak1_candidates.csv",index=False); anti.to_csv(outdir_path/"jak2_antipharmacophore_candidates.csv",index=False)
    _rescore(j1,matrix.loc[holdout_idx],"JAK1_SELECTIVE","jak1_holdout").to_csv(outdir_path/"jak1_holdout_results.csv",index=False)
    _rescore(anti,matrix.loc[holdout_idx],"JAK2_SELECTIVE","jak2_antipharmacophore_holdout").to_csv(outdir_path/"jak2_antipharmacophore_holdout_results.csv",index=False)
    if args.validation_matrix:
        val=pd.read_csv(args.validation_matrix)
    else:
        raw_val=load_validation(args.validation_jak1,args.validation_jak2,args.validation_ki)
        val=raw_val.drop(columns="feature_set").join(materialize_features(raw_val,feature_columns(matrix)).astype("int8"))
        val["selectivity_class"]=assign_classes(val.delta_pKi,math.log10(5.0))
        val.to_csv(outdir_path/"counter_validation_feature_matrix.csv",index=False)
    _rescore(j1,val,"JAK1_SELECTIVE","jak1_independent_validation").to_csv(outdir_path/"jak1_independent_validation_results.csv",index=False)
    _rescore(anti,val,"JAK2_SELECTIVE","jak2_antipharmacophore_independent_validation").to_csv(outdir_path/"jak2_antipharmacophore_independent_validation_results.csv",index=False)

if __name__=="__main__": main()
