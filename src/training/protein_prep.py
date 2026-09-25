"""
protein_prep.py
================
Step 1 of 3 in the JAK docking pipeline.

For JAK1 and JAK2 crystal structures:
  1) Auto-detect the binding site by finding the largest bound organic
     ligand in the CIF (excluding waters, ions, crystallization additives,
     and modified amino acids) and computing its centroid.
  2) Clean the structure (drop waters/additives, keep structural ions and
     phosphotyrosine) and convert to a receptor PDBQT via Meeko
     (mk_prepare_receptor.py), which replaces the old MGLTools step.
  3) Write a receptors.json describing each prepared receptor + its
     docking box, consumed by docking_analysis.py.

Uses Meeko instead of MGLTools for receptor preparation:
  - pure Python, pip-installable, native on Apple Silicon
  - actively maintained by the same lab (Forli Lab, Scripps) that
    maintains AutoDock Vina itself

Requirements: biopython, numpy, meeko  (pip install biopython numpy meeko)
Also requires the AutoDock Vina binary for later steps, but not for this one.

Run:  python protein_prep.py
Output:
  prepared_receptors/pdbqt/JAK1.pdbqt, JAK2.pdbqt
  prepared_receptors/pdb/JAK1.pdb,     JAK2.pdb      (cleaned, for PLIP)
  receptors.json
"""

# ============================================================
# IMPORTS (all imported up front, before any use)
# ============================================================

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from Bio.PDB import MMCIFParser, PDBIO, Select


# ============================================================
# CONFIG
# ============================================================

ISOFORMS: Dict[str, str] = {
    "JAK1": "Crystal Structures/JAK1_4EI4.cif",
    "JAK2": "Crystal Structures/JAK2_4JI9.cif",
}

PROTEIN_PDBQT_DIR = Path("prepared_receptors/pdbqt")
PROTEIN_PDB_DIR = Path("prepared_receptors/pdb")
RECEPTORS_JSON = Path("receptors.json")

DEFAULT_BOX_SIZE = (20, 20, 20)
DEFAULT_EXHAUSTIVENESS = 8

# Residue handling during CIF -> PDB cleanup (applies regardless of which
# ligand is auto-detected as the binding-site reference)
KEEP_HET_RESIDUES = {"PTR"}                      # phosphotyrosine, keep
KEEP_IONS = {"ZN", "MG", "MN", "FE", "CA", "K"}   # structural ions, keep
REMOVE_EXPLICIT = {"HOH", "PEG", "PG4", "DTT"}    # crystallization junk, drop

# ------------------------------------------------------------
# Exclusion blacklist for AUTO-DETECTING the bound organic ligand.
# Anything in this set is not considered a candidate "ligand" when
# searching for the largest bound organic molecule.
# ------------------------------------------------------------
WATERS = {"HOH", "DOD"}
COMMON_IONS = {"ZN", "MG", "MN", "FE", "CA", "K", "NA", "CL", "CO", "NI", "CU", "CD", "BR", "IOD", "SO4", "PO4"}
CRYO_ADDITIVES = {
    "PEG", "PG4", "PGE", "GOL", "EDO", "DMS", "TRS", "MES", "ACT", "FMT",
    "EPE", "BME", "MPD", "IPA", "MOH", "CIT", "TAM", "1PE", "P6G", "PEO",
    "DIO", "BOG", "OGA", "SIN",
}
MODIFIED_AMINO_ACIDS = {"PTR", "SEP", "TPO", "MSE", "CSO", "KCX", "MLY", "M3L"}

LIGAND_EXCLUDE_RESNAMES = WATERS | COMMON_IONS | CRYO_ADDITIVES | MODIFIED_AMINO_ACIDS

# Name of the Meeko CLI receptor-prep entry point (installed by `pip install
# meeko`, lands on PATH automatically inside your active environment).
MK_PREPARE_RECEPTOR = "mk_prepare_receptor.py"


# ============================================================
# BINDING SITE AUTO-DETECTION
# ============================================================

def find_largest_organic_ligand(cif_file: str) -> Dict:
    """
    Scan a CIF file for hetero residues, exclude waters/ions/crystallization
    additives/modified amino acids, and return the largest remaining
    hetero residue (by atom count) along with its centroid.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("receptor", cif_file)

    candidates = []  # (n_atoms, resname, chain_id, centroid)

    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag = residue.id[0]
                if hetflag == " ":
                    continue  # standard protein residue, not a ligand
                resname = residue.get_resname().strip().upper()
                if resname in LIGAND_EXCLUDE_RESNAMES:
                    continue

                coords = [atom.coord for atom in residue]
                if not coords:
                    continue

                n_atoms = len(coords)
                # cast to plain Python floats so the centroid is JSON-serializable
                centroid = tuple(float(v) for v in np.array(coords).mean(axis=0))
                candidates.append((n_atoms, resname, chain.id, centroid))
        break  # only consider the first model

    if not candidates:
        raise ValueError(
            f"No candidate organic ligand found in {cif_file} "
            f"(after excluding waters/ions/additives/modified residues)."
        )

    candidates.sort(key=lambda c: c[0], reverse=True)
    n_atoms, resname, chain_id, centroid = candidates[0]

    return {
        "resname": resname,
        "chain": chain_id,
        "n_atoms": n_atoms,
        "centroid": centroid,
    }


# ============================================================
# HETERO RESIDUE AUDIT (diagnostic only, no files changed)
# ============================================================

def audit_hetero_residues(cif_file: str, isoform_name: str) -> None:
    """Print every distinct hetero residue found in the structure and
    whether CleanProteinSelect will keep or drop it, so nothing gets
    silently discarded without visibility."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(isoform_name, cif_file)

    print(f"    --- Hetero residue audit for {isoform_name} ---")
    seen = set()
    any_hetero = False

    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag = residue.id[0]
                if hetflag == " ":
                    continue  # standard protein residue, not part of this audit

                any_hetero = True
                resname = residue.get_resname().strip().upper()
                resseq = residue.id[1]
                key = (chain.id, resname, resseq)
                if key in seen:
                    continue
                seen.add(key)

                if resname in REMOVE_EXPLICIT:
                    decision = "REMOVED (explicit junk list)"
                elif resname in KEEP_HET_RESIDUES or resname in KEEP_IONS:
                    decision = "KEPT (whitelist)"
                else:
                    decision = "REMOVED (not on keep-list)"

                print(f"        {resname:6s} chain {chain.id} resseq {resseq:>5}  ->  {decision}")
        break  # first model only

    if not any_hetero:
        print("        (no hetero residues found)")


# ============================================================
# CIF -> CLEANED PDB
# ============================================================

class CleanProteinSelect(Select):
    """Biopython residue filter: keep standard protein residues, drop
    crystallization junk, keep a whitelist of hetero residues/ions."""

    def accept_residue(self, residue):
        resname = residue.get_resname().strip().upper()
        hetflag = residue.id[0]
        if hetflag == " ":
            return 1
        if resname in REMOVE_EXPLICIT:
            return 0
        return 1 if (resname in KEEP_HET_RESIDUES or resname in KEEP_IONS) else 0


def cif_to_pdb_cleaned(cif_file: str, pdb_file: str) -> None:
    """CIF -> cleaned PDB (protein + kept hetero residues/ions). This same
    cleaned PDB is used both as the Meeko receptor-prep input and later as
    the PLIP-side receptor file after docking."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("receptor", cif_file)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_file, CleanProteinSelect())


# ============================================================
# PDB -> PDBQT (Meeko, replaces MGLTools' prepare_receptor4.py)
# ============================================================

def prepare_receptor_pdbqt_with_meeko(cleaned_pdb_path: Path, output_pdbqt_path: Path) -> None:
    """
    Convert a cleaned receptor PDB to PDBQT using Meeko's CLI tool
    (mk_prepare_receptor.py), the modern, actively-maintained replacement
    for MGLTools' prepare_receptor4.py.

    Meeko's tool writes "<basename>.pdbqt" based on -o/--output_basename,
    so we prepare into a temp basename and move the result to the exact
    path we want.
    """
    output_pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
    basename = str(output_pdbqt_path.with_suffix(""))  # strip .pdbqt

    cmd = [
        MK_PREPARE_RECEPTOR,
        "--read_pdb", str(cleaned_pdb_path),
        "-o", basename,
        "-p",  # write PDBQT
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mk_prepare_receptor.py failed:\n{result.stderr}")

    written = Path(f"{basename}.pdbqt")
    if not written.exists():
        raise RuntimeError(
            f"mk_prepare_receptor.py did not produce the expected output "
            f"({written}). Full output:\n{result.stdout}\n{result.stderr}"
        )
    if written != output_pdbqt_path:
        written.rename(output_pdbqt_path)


def validate_receptor_pdbqt(pdbqt_path: Path) -> Tuple[bool, str]:
    """Lightweight sanity check on the Meeko-prepared receptor PDBQT:
    file exists, is non-empty, and has at least one ATOM/HETATM line."""
    if not pdbqt_path.exists():
        return False, "File does not exist."

    n_atoms = 0
    with open(pdbqt_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                n_atoms += 1

    if n_atoms == 0:
        return False, "File contains no ATOM/HETATM records."
    return True, f"Atoms: {n_atoms}"


def prepare_isoform(isoform_name: str, cif_path: str) -> Path:
    """Full protein prep for one isoform: CIF -> cleaned PDB -> Meeko PDBQT
    -> validate. Also writes the cleaned PDB used later by PLIP."""
    output_pdbqt = PROTEIN_PDBQT_DIR / f"{isoform_name}.pdbqt"
    PROTEIN_PDBQT_DIR.mkdir(parents=True, exist_ok=True)
    PROTEIN_PDB_DIR.mkdir(parents=True, exist_ok=True)

    pdb_out = PROTEIN_PDB_DIR / f"{isoform_name}.pdb"
    cif_to_pdb_cleaned(cif_path, str(pdb_out))

    print(f"[>] Converting {isoform_name} cleaned PDB -> PDBQT (Meeko) ...")
    prepare_receptor_pdbqt_with_meeko(pdb_out, output_pdbqt)

    success, info = validate_receptor_pdbqt(output_pdbqt)
    print(f"[{'OK' if success else 'FAIL'}] {isoform_name}: {info}")

    return output_pdbqt


# ============================================================
# MAIN
# ============================================================

def main():
    receptor_map = {}

    for isoform, cif_path in ISOFORMS.items():
        print(f"\n=== {isoform} ({cif_path}) ===")

        if not os.path.exists(cif_path):
            print(f"[!] CIF file not found, skipping: {cif_path}")
            continue

        # 1) Auto-detect binding site from the largest bound organic ligand
        site_info = find_largest_organic_ligand(cif_path)
        print(
            f"    Detected binding-site ligand: {site_info['resname']} "
            f"(chain {site_info['chain']}, {site_info['n_atoms']} atoms)"
        )
        print(f"    Docking box center: {site_info['centroid']}")

        # 2) Audit every hetero residue's keep/drop fate before cleanup
        audit_hetero_residues(cif_path, isoform)

        # 3) Prepare receptor (cleaned PDB + PDBQT via Meeko)
        pdbqt_path = prepare_isoform(isoform, cif_path)

        receptor_map[isoform] = {
            "pdbqt": str(pdbqt_path),
            "pdb": str(PROTEIN_PDB_DIR / f"{isoform}.pdb"),
            "center": site_info["centroid"],
            "size": DEFAULT_BOX_SIZE,
            "exhaustiveness": DEFAULT_EXHAUSTIVENESS,
            "detected_ligand_resname": site_info["resname"],
            "detected_ligand_chain": site_info["chain"],
            "detected_ligand_natoms": site_info["n_atoms"],
        }

    with open(RECEPTORS_JSON, "w") as f:
        json.dump(receptor_map, f, indent=2)

    print(f"\nWrote receptor config for {list(receptor_map.keys())} to {RECEPTORS_JSON}")


if __name__ == "__main__":
    main()