"""Rank compounds by worst-case JAK2-over-JAK1 MM/GBSA selectivity across engines.

ddG = MM/GBSA(JAK2) - MM/GBSA(JAK1) per engine (negative = JAK2-favored).
Worst case = max(ddG_vina, ddG_rdock), i.e. the less JAK2-favorable engine.

Filters (both on by default):
  --require-negative-jak2  JAK2 MM/GBSA < 0 in both engines
  --score-cutoff 200       all four MM/GBSA scores within +/- cutoff

Outputs (in --outdir): top_selective_compounds.csv, scaffold_groups.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina


def rank_compounds(scores: pd.DataFrame, require_negative_jak2: bool, cutoff: float) -> pd.DataFrame:
    p = scores.pivot_table(index="ligand_id", columns=["engine", "target"],
                           values="mmgbsa_score", aggfunc="first").dropna()
    out = pd.DataFrame({
        "vina_JAK1": p[("vina", "JAK1")], "vina_JAK2": p[("vina", "JAK2")],
        "rdock_JAK1": p[("rdock", "JAK1")], "rdock_JAK2": p[("rdock", "JAK2")],
    })
    out["ddG_vina"] = out.vina_JAK2 - out.vina_JAK1
    out["ddG_rdock"] = out.rdock_JAK2 - out.rdock_JAK1
    out["worst_ddG"] = out[["ddG_vina", "ddG_rdock"]].max(axis=1)

    keep = pd.Series(True, index=out.index)
    if require_negative_jak2:
        keep &= (out.vina_JAK2 < 0) & (out.rdock_JAK2 < 0)
    if cutoff:
        keep &= (out[["vina_JAK1", "vina_JAK2", "rdock_JAK1", "rdock_JAK2"]].abs() < cutoff).all(axis=1)
    return out[keep].sort_values("worst_ddG")


def scaffold_groups(smiles: pd.Series, cluster_distance: float) -> pd.DataFrame:
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(m) for m in mols]
    n = len(fps)
    dist = [1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i in range(1, n) for j in range(i)]
    clusters = Butina.ClusterData(dist, n, cluster_distance, isDistData=True)
    cluster_of = {idx: c for c, members in enumerate(clusters, 1) for idx in members}
    return pd.DataFrame({
        "ligand_id": smiles.index,
        "murcko_scaffold": [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for m in mols],
        "butina_cluster": [cluster_of[i] for i in range(n)],
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="results/jh1_colabfold_coconut/docking_scores.csv")
    ap.add_argument("--outdir", default="results/jh1_colabfold_coconut")
    ap.add_argument("--top", type=int, default=9)
    ap.add_argument("--score-cutoff", type=float, default=200.0, help="0 disables")
    ap.add_argument("--no-require-negative-jak2", action="store_true")
    ap.add_argument("--cluster-distance", type=float, default=0.7, help="Butina distance (1 - Tanimoto)")
    args = ap.parse_args()

    scores = pd.read_csv(args.scores)
    ranked = rank_compounds(scores, not args.no_require_negative_jak2, args.score_cutoff)
    print(f"{len(ranked)} compounds pass filters")

    meta = scores.drop_duplicates("ligand_id").set_index("ligand_id")[["smiles", "QSAR_rank"]]
    top = ranked.head(args.top).join(meta)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    top.round(2).to_csv(out / "top_selective_compounds.csv")

    groups = scaffold_groups(top["smiles"], args.cluster_distance)
    groups.to_csv(out / "scaffold_groups.csv", index=False)
    print(top.round(1).drop(columns="smiles").to_string())
    print(f"{groups.murcko_scaffold.nunique()} distinct Murcko scaffolds, "
          f"{groups.butina_cluster.nunique()} Butina clusters at distance {args.cluster_distance}")


if __name__ == "__main__":
    main()
