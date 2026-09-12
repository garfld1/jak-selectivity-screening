#!/usr/bin/env python3
"""Scaffold-diverse frozen-candidate evaluation (Tanimoto cutoff 0.40)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
if __package__ is None: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.common import EXTENSIONS, frozen_rescore, paths

TANIMOTO_CUTOFF = 0.40  # Addition; not part of the original method.
HOLDOUT_FRACTION = 0.18  # Addition; chosen within the requested 15--20% range.

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default=paths()["matrix"]); p.add_argument("--candidates", default=paths()["candidates"])
    p.add_argument("--ligand-map", default=Path(__file__).resolve().parents[1] / "docking/docking_prep/ligands_pdbqt.csv",
                   help="Training ligand_id-to-SMILES mapping supplied for this extension.")
    p.add_argument("--out", default=EXTENSIONS / "scaffold_holdout_results.csv"); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); matrix, candidates = pd.read_csv(args.matrix), pd.read_csv(args.candidates)
    smiles_col = next((c for c in ("SMILES", "smiles") if c in matrix), None)
    if smiles_col is None:
        ligand_map = pd.read_csv(args.ligand_map, usecols=["ligand_id", "smiles"])
        if ligand_map.ligand_id.duplicated().any(): raise ValueError("Ligand-to-SMILES mapping has duplicate ligand IDs.")
        matrix = matrix.merge(ligand_map, on="ligand_id", how="left", validate="one_to_one")
        smiles_col = "smiles"
        if matrix[smiles_col].isna().any():
            raise ValueError("Ligand-to-SMILES mapping does not cover every feature-matrix ligand.")
    mols = [Chem.MolFromSmiles(x) for x in matrix[smiles_col]]
    if any(m is None for m in mols): raise ValueError("Invalid SMILES in feature matrix.")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(m) for m in mols]
    distances = [1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i in range(1, len(fps)) for j in range(i)]
    clusters = Butina.ClusterData(distances, len(fps), 1 - TANIMOTO_CUTOFF, isDistData=True)
    rng, selected, n = np.random.default_rng(args.seed), [], 0
    for cluster in rng.permutation(len(clusters)):
        selected.extend(clusters[cluster]); n += len(clusters[cluster])
        if n >= round(len(matrix) * HOLDOUT_FRACTION): break
    results = frozen_rescore(candidates, matrix, matrix.iloc[sorted(selected)], "scaffold_holdout")
    results.to_csv(args.out, index=False); print(f"Saved {len(results)} scaffold-holdout scores to {args.out}")

if __name__ == "__main__": main()
