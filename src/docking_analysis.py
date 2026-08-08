"""
docking_analysis_fixed.py
=========================
Step 3 of 3 in the JAK docking pipeline.

Key fixes vs the original version:
1) Strip receptor hetero-residues before PLIP so PTR/co-crystal ligands do not
   get mistaken for the docked ligand.
2) Do not round-trip the docked pose through RDKit for the final complex PDB.
   Instead, convert the top-pose PDBQT directly to a proper HETATM ligand PDB.
3) Preserve a real chain ID and residue name for the docked ligand so PLIP can
   recognize it as a ligand.
4) Make file paths relative to the script location by default.

Reads:
  - receptors.json         (prepared receptor paths + docking box center/size)
  - ligand CSV             (ligand_id + PDBQT columns)

For every ligand x isoform pair:
  1) Dock with AutoDock Vina
  2) Parse the top-pose binding score from the Vina log
  3) Convert the top pose PDBQT -> PDB with a direct writer
  4) Merge with a protein-only receptor PDB into a complex file
  5) Run PLIP on the complex and record every interacting amino acid

Output: one wide CSV per isoform with ligand_id, vina_score, then one binary
column per residue seen in that isoform.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tqdm

from plip.structure.preparation import PDBComplex

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RECEPTORS_JSON = PROJECT_ROOT / "docking" / "docking_prep" / "receptors.json"
COMPLEX_OUT_DIR = PROJECT_ROOT / "saved_complexes"
DEFAULT_EXHAUSTIVENESS = 32
VINA_BIN = os.path.expanduser("~/bin/vina")

# Ligand residue naming used in the merged complex PDB.
LIGAND_RESNAME = "LIG"
LIGAND_CHAIN_ID = "Z"
LIGAND_RESSEQ = 1

PLIP_ATTRS = {
    "hbonds_ligand": "hydrogen_bond",
    "hydrophobic_contacts": "hydrophobic",
    "pistacking": "pi_stacking",
    "saltbridges_ligand": "salt_bridge",
    "halogenbonds": "halogen_bond",
    "waterbridges": "water_bridge",
    "metal_complex": "metal_complex",
}

# ============================================================
# VINA DOCKING
# ============================================================

def dock_with_vina(
    ligand_pdbqt: str,
    receptor_pdbqt: str,
    out_pdbqt: str,
    log_path: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    exhaustiveness: int = 32,
    cpu: int = 1,
    timeout: int = 300,
) -> None:
    if not os.path.exists(VINA_BIN):
        raise RuntimeError(f"Vina binary not found at {VINA_BIN}")

    cx, cy, cz = center
    sx, sy, sz = size
    cmd = [
        VINA_BIN,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--out", out_pdbqt,
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
        "--exhaustiveness", str(exhaustiveness),
        "--cpu", str(cpu),
    ]

    with open(log_path, "w") as log_file:
        subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
            text=True,
        )

def parse_vina_score_from_log(log_path: str) -> Optional[float]:
    """Return the top-pose affinity from a Vina log file."""
    if not os.path.exists(log_path):
        return None

    with open(log_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "1":
                try:
                    return float(parts[1])
                except ValueError:
                    pass

    best = None
    with open(log_path, "r") as f:
        for line in f:
            for tok in line.split():
                try:
                    v = float(tok)
                    if best is None or v < best:
                        best = v
                except ValueError:
                    pass
    return best

# ============================================================
# PDB/PDBQT HELPERS
# ============================================================

def _infer_element(atom_name: str, line: str) -> str:
    """
    Best-effort element inference for a PDBQT atom line.
    Prefer the explicit atom-type token if available; otherwise infer from
    the atom name.
    """
    tail = line[66:].strip().split()
    if tail:
        token = tail[-1].strip()
        # common Vina atom types
        token = re.sub(r"[^A-Za-z]", "", token)
        if token:
            token = token[:2].title()
            return token

    name = re.sub(r"[^A-Za-z]", "", atom_name).strip()
    if not name:
        return "C"
    if len(name) >= 2 and name[1].islower():
        return name[:2].title()
    return name[0].upper()

def pdbqt_pose_to_ligand_pdb_string(
    pdbqt_path: str,
    resname: str = LIGAND_RESNAME,
    chain_id: str = LIGAND_CHAIN_ID,
    resseq: int = LIGAND_RESSEQ,
) -> str:
    """
    Convert the top docked pose PDBQT into a clean HETATM-only PDB ligand block.
    This preserves coordinates but intentionally normalizes residue/chain naming
    so PLIP sees the docked molecule as the ligand.
    """
    if not os.path.exists(pdbqt_path) or os.path.getsize(pdbqt_path) < 50:
        raise RuntimeError(f"Vina output missing/empty: {pdbqt_path}")

    atom_lines: List[str] = []
    serial_out = 1

    with open(pdbqt_path, "r") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM", "HETATM")):
                continue

            # PDBQT is PDB-like in the first 66 columns.
            atom_name = line[12:16]
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            occ = 1.00
            temp = 0.00
            if line[54:60].strip():
                try:
                    occ = float(line[54:60])
                except ValueError:
                    pass
            if line[60:66].strip():
                try:
                    temp = float(line[60:66])
                except ValueError:
                    pass

            element = _infer_element(atom_name, line)

            # Standard PDB fixed-width formatting.
            atom_lines.append(
                f"HETATM{serial_out:5d} {atom_name:<4s}{' ':1s}{resname:>3s} {chain_id:1s}{resseq:4d}{' ':1s}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp:6.2f}          {element:>2s}"
            )
            serial_out += 1

    if not atom_lines:
        raise RuntimeError(f"No ligand atoms found in docked pose: {pdbqt_path}")

    atom_lines.append("TER")
    atom_lines.append("END")
    return "\n".join(atom_lines) + "\n"

def protein_only_pdb(pdb_text: str) -> str:
    """
    Keep only protein ATOM records from the receptor PDB.
    This removes receptor HETATM records such as PTR, waters, cofactors, etc.,
    which otherwise can be mistaken by PLIP for the ligand.
    """
    keep: List[str] = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            keep.append(line)
    keep.append("TER")
    keep.append("END")
    return "\n".join(keep) + "\n"

# ============================================================
# PLIP ANALYSIS
# ============================================================

def run_plip_all_residues(complex_pdb: str) -> List[str]:
    """
    Run PLIP on a receptor-ligand complex and return the sorted list of every
    interacting amino-acid residue label.
    """
    pc = PDBComplex()
    pc.load_pdb(complex_pdb)
    pc.analyze()

    interacting = set()
    for _, interactions in pc.interaction_sets.items():
        for attr in PLIP_ATTRS:
            arr = getattr(interactions, attr, None)
            if not arr:
                continue
            for entry in arr:
                resnr = getattr(entry, "resnr", None) or getattr(entry, "res_seq", None) or getattr(entry, "resid", None)
                restype = getattr(entry, "restype", None) or getattr(entry, "resname", None) or getattr(entry, "residue", None)
                if restype and resnr:
                    try:
                        interacting.add(f"{str(restype).upper()}{int(resnr)}")
                    except (TypeError, ValueError):
                        pass
    return sorted(interacting)

# ============================================================
# MAIN DOCKING + ANALYSIS LOOP
# ============================================================

def dock_and_analyze_all(
    ligand_df: pd.DataFrame,
    receptor_map: Dict[str, Dict],
    ligand_id_col: str = "ligand_id",
    pdbqt_col: str = "PDBQT",
) -> Dict[str, pd.DataFrame]:
    COMPLEX_OUT_DIR.mkdir(parents=True, exist_ok=True)
    workspace_root = Path(tempfile.mkdtemp(prefix="vina_workspace_"))

    raw_results: Dict[str, List[Dict]] = {iso: [] for iso in receptor_map}
    total_jobs = len(ligand_df) * len(receptor_map)
    print(f"Running {total_jobs} docking jobs ({len(ligand_df)} ligands x {len(receptor_map)} isoforms)...")

    try:
        with tqdm.tqdm(total=total_jobs, desc="DOCKING") as pbar:
            for _, row in ligand_df.iterrows():
                lig_id = str(row[ligand_id_col])
                lig_pdbqt_str = row[pdbqt_col]

                if not isinstance(lig_pdbqt_str, str) or not lig_pdbqt_str.strip():
                    print(f"[!] Skipping {lig_id}: no PDBQT available.")
                    for iso_name in receptor_map:
                        raw_results[iso_name].append({"ligand_id": lig_id, "vina_score": None, "residues": set()})
                        pbar.update(1)
                    continue

                for iso_name, rec in receptor_map.items():
                    job_dir = workspace_root / f"{lig_id[:8]}_{iso_name}"
                    job_dir.mkdir(parents=True, exist_ok=True)

                    lig_in = job_dir / "lig.pdbqt"
                    out_pdbqt = job_dir / "docked.pdbqt"
                    log_path = job_dir / "vina.log"
                    saved_complex_path = COMPLEX_OUT_DIR / f"{lig_id[:8]}_{iso_name}_complex.pdb"

                    try:
                        with open(lig_in, "w") as f:
                            f.write(lig_pdbqt_str)

                        dock_with_vina(
                            str(lig_in),
                            rec["pdbqt"],
                            str(out_pdbqt),
                            str(log_path),
                            center=tuple(rec["center"]),
                            size=tuple(rec["size"]),
                            exhaustiveness=rec.get("exhaustiveness", DEFAULT_EXHAUSTIVENESS),
                        )

                        score = parse_vina_score_from_log(str(log_path))
                        ligand_pose_pdb = pdbqt_pose_to_ligand_pdb_string(str(out_pdbqt))

                        with open(rec["pdb"], "r") as f_rec:
                            receptor_pdb_text = f_rec.read()
                        protein_pdb_text = protein_only_pdb(receptor_pdb_text)

                        with open(saved_complex_path, "w") as f_comp:
                            f_comp.write(protein_pdb_text)
                            f_comp.write(ligand_pose_pdb)

                        residues = run_plip_all_residues(str(saved_complex_path))

                        raw_results[iso_name].append(
                            {"ligand_id": lig_id, "vina_score": score, "residues": set(residues)}
                        )

                    except Exception as e:
                        print(f"\nERROR docking {lig_id} x {iso_name}: {e}")
                        raw_results[iso_name].append({"ligand_id": lig_id, "vina_score": None, "residues": set()})
                    finally:
                        shutil.rmtree(job_dir, ignore_errors=True)
                        pbar.update(1)
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)

    wide_results: Dict[str, pd.DataFrame] = {}
    for iso_name, records in raw_results.items():
        all_residues = sorted({r for rec in records for r in rec["residues"]})

        rows = []
        for rec in records:
            row = {"ligand_id": rec["ligand_id"], "vina_score": rec["vina_score"]}
            for res in all_residues:
                row[res] = 1 if res in rec["residues"] else 0
            rows.append(row)

        df = pd.DataFrame(rows)
        ordered_cols = ["ligand_id", "vina_score"] + all_residues
        df = df.reindex(columns=ordered_cols)
        wide_results[iso_name] = df

    return wide_results

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Dock ligands against prepared JAK receptors and run PLIP.")
    parser.add_argument("--receptors", default=str(RECEPTORS_JSON), help="Path to receptors.json from protein_prep.py")
    parser.add_argument("--ligands", required=True, help="Path to ligand PDBQT CSV from ligand_prep.py")
    parser.add_argument("--outdir", default=str(PROJECT_ROOT), help="Directory to write per-isoform results CSVs")
    args = parser.parse_args()

    with open(args.receptors, "r") as f:
        receptor_map = json.load(f)

    ligand_df = pd.read_csv(args.ligands)
    wide_results = dock_and_analyze_all(ligand_df, receptor_map)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for iso_name, df in wide_results.items():
        out_path = outdir / f"{iso_name}_docking_results.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({df.shape[0]} ligands x {df.shape[1] - 2} residue columns)")

if __name__ == "__main__":
    main()