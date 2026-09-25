"""
ligand_prep.py
==============
Step 2 of 3 in the JAK docking pipeline.

Reads a CSV of ligands (must contain a SMILES column), generates a stable
ligand_id (SHA256 hash of the SMILES) if one isn't already present,
converts each SMILES to a 3D conformer (RDKit), then to a PDBQT string
using Meeko (replaces the old RDKit -> OpenBabel conversion path — Meeko
is pip-installable, native on Apple Silicon, and maintained by the same
lab that maintains AutoDock Vina). Writes out a new CSV with the added
PDBQT column for docking_analysis.py to consume.

Requirements: pandas, rdkit, meeko  (pip install pandas rdkit meeko)

Run:
  python ligand_prep.py --input ligand_subset.csv --output ligand_subset_pdbqt.csv

Input CSV requirements:
  - a SMILES column (default name: "smiles")
  - optionally a ligand_id column (default name: "ligand_id"); if missing,
    one is generated automatically from a hash of the SMILES string.
"""

# ============================================================
# IMPORTS (all imported up front, before any use)
# ============================================================

import argparse
import hashlib

import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from meeko import MoleculePreparation, PDBQTWriterLegacy


# ============================================================
# LIGAND ID GENERATION
# ============================================================

def add_smiles_hash_column(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    out_col: str = "ligand_id",
    algorithm: str = "sha256",
) -> pd.DataFrame:
    """Add a stable ligand_id column derived from hashing each SMILES string,
    if out_col isn't already present."""
    if out_col in df.columns:
        return df

    def hash_smiles(smiles: str) -> str:
        return hashlib.new(algorithm, str(smiles).encode("utf-8")).hexdigest()

    df = df.copy()
    df[out_col] = df[smiles_col].apply(hash_smiles)
    return df


# ============================================================
# SMILES -> 3D CONFORMER (RDKit) -> PDBQT (Meeko)
# ============================================================

def generate_3d_conformer(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    if AllChem.EmbedMolecule(mol, randomSeed=0xF00D) != 0:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            raise RuntimeError("Failed to generate 3D conformer")

    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol)

    return mol


def rdkit_mol_to_pdbqt_text(mol: Chem.Mol) -> str:
    """Convert an RDKit 3D molecule directly to a PDBQT string using Meeko's
    Python API — no external converter or temp files needed."""
    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    if not mol_setups:
        raise RuntimeError("Meeko produced no molecule setups for this ligand.")

    # A molecule can, in principle, yield multiple setups (e.g. multiple
    # protonation/tautomer states); take the first, which is Meeko's
    # default choice.
    setup = mol_setups[0]
    pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(setup)
    if not is_ok:
        raise RuntimeError(f"Meeko failed to write PDBQT: {err_msg}")

    return pdbqt_string


def prepare_ligand(smiles: str) -> str:
    """Full ligand pipeline: SMILES -> 3D conformer -> PDBQT text."""
    mol3d = generate_3d_conformer(smiles)
    return rdkit_mol_to_pdbqt_text(mol3d)


def prepare_all_ligands(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """Convert every SMILES in df to PDBQT text; failures are logged and
    kept as None rows rather than crashing the whole batch."""
    df = df.copy()
    pdbqt_list = []
    for i, smi in enumerate(df[smiles_col]):
        try:
            pdbqt_list.append(prepare_ligand(smi))
        except Exception as e:
            print(f"[!] Failed to prepare ligand {i} ({smi}): {e}")
            pdbqt_list.append(None)
    df["PDBQT"] = pdbqt_list
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare ligand PDBQTs from a SMILES CSV.")
    parser.add_argument("--input", required=True, help="Input CSV/TSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path (adds PDBQT column).")
    parser.add_argument("--smiles-col", default="smiles", help="Name of the SMILES column.")
    parser.add_argument("--sep", default=",", help="Delimiter for the input file (e.g. '\\t' for TSV).")
    parser.add_argument("--n", type=int, default=None, help="Optional: only process the first N rows.")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=args.sep)
    if args.n is not None:
        df = df.head(args.n)

    df = add_smiles_hash_column(df, smiles_col=args.smiles_col, out_col="ligand_id")

    print(f"Preparing {len(df)} ligands from '{args.input}' ...")
    df = prepare_all_ligands(df, smiles_col=args.smiles_col)

    n_failed = df["PDBQT"].isna().sum()
    print(f"Done. {len(df) - n_failed}/{len(df)} ligands prepared successfully.")

    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()