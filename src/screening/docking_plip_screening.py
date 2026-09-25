#!/usr/bin/env python3
"""
jak_screening_pipeline.py
=========================
Combined ligand-preparation + JAK1/JAK2 docking + PLIP screening pipeline.

Input CSV requirements
----------------------
Required:
    smiles

Recommended / expected for the user's screening CSV:
    zinc_id
    predicted_delta_pIC50
    QSAR_rank

The script adds:
    ligand_id
    PDBQT
    PDBQT_prep_error

Then it docks every prepared ligand against every receptor in receptors.json
(e.g. JAK1 and JAK2), keeps the top Vina pose/score, builds a clean
receptor+ligand complex PDB, runs PLIP, and writes one wide results CSV per
isoform.

Outputs (inside --outdir)
-------------------------
    <input_stem>_prepared.csv
    docking_checkpoint.jsonl
    <isoform>_docking_results.csv
    saved_complexes_PLIP/
        <ligand>_<isoform>_<unique>_complex.pdb

Example
-------
python jak_screening_pipeline.py \
    --input /path/to/compounds.csv \
    --outdir /path/to/screening_results \
    --receptors /path/to/receptors.json \
    --workers 7 \
    --cpu-per-job 1 \
    --exhaustiveness 32

Resume an interrupted run
-------------------------
python jak_screening_pipeline.py \
    --input /path/to/compounds.csv \
    --outdir /path/to/screening_results \
    --receptors /path/to/receptors.json \
    --workers 7 \
    --cpu-per-job 1 \
    --resume

Requirements
------------
pandas, rdkit, meeko, plip, tqdm, and an AutoDock Vina executable.

Notes
-----
- The input's predicted_delta_pIC50 is carried through as a screening/QSAR
  prediction. This script does NOT recompute it from experimental IC50 values.
- If ligand_id is already present, it is preserved.
- Otherwise, zinc_id is copied into ligand_id when available.
- If neither exists, ligand_id is generated as SHA256(smiles).
- PDBQT preparation uses RDKit -> 3D conformer -> Meeko.
- Each docking job is one (ligand, isoform) pair.
- Process-level parallelism is used for docking + PLIP.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from plip.structure.preparation import PDBComplex


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path("/Users/shreechatterjee/jak-selectivity-screening")
DEFAULT_RECEPTORS_JSON = PROJECT_ROOT / "docking" / "docking_prep" / "receptors.json"
DEFAULT_VINA_BIN = Path(os.path.expanduser("~/bin/vina"))

DEFAULT_EXHAUSTIVENESS = 32
DEFAULT_WORKERS = 1
DEFAULT_CPU_PER_JOB = 1
DEFAULT_TIMEOUT = 900

LIGAND_ID_COL = "ligand_id"
ZINC_ID_COL = "zinc_id"
SMILES_COL = "smiles"
PREDICTED_DELTA_COL = "predicted_delta_pIC50"
QSAR_RANK_COL = "QSAR_rank"
PDBQT_COL = "PDBQT"
PDBQT_ERROR_COL = "PDBQT_prep_error"

COMPLEX_OUT_NAME = "saved_complexes_PLIP"
CHECKPOINT_NAME = "docking_checkpoint.jsonl"

LIGAND_RESNAME = "LIG"
LIGAND_CHAIN_ID = "Z"
LIGAND_RESSEQ = 1
INTERACTION_TYPE_SEP = ";"

EXCLUDE_RESNAMES = {
    "PTR",  # phosphotyrosine
    "SEP",  # phosphoserine
    "TPO",  # phosphothreonine
    "HOH",  # water
    "MSE",  # selenomethionine
}

# AutoDock/Vina atom type -> PDB element.
# Important: NA is a nitrogen AutoDock atom type, NOT sodium; SA is sulfur,
# HD/HS are hydrogen, etc.
AD_TYPE_TO_ELEMENT = {
    "A": "C",
    "C": "C",
    "N": "N",
    "NA": "N",
    "NS": "N",
    "O": "O",
    "OA": "O",
    "OS": "O",
    "S": "S",
    "SA": "S",
    "H": "H",
    "HD": "H",
    "HS": "H",
    "F": "F",
    "CL": "Cl",
    "BR": "Br",
    "I": "I",
    "MG": "Mg",
    "CA": "Ca",
    "MN": "Mn",
    "FE": "Fe",
    "ZN": "Zn",
    "P": "P",
    "SI": "Si",
    "B": "B",
}

PLIP_ATTRS = {
    "hbonds_pdon": "hydrogen_bond",
    "hbonds_ldon": "hydrogen_bond",
    "hydrophobic_contacts": "hydrophobic",
    "pistacking": "pi_stacking",
    "pication_laro": "pi_cation",
    "pication_paro": "pi_cation",
    "saltbridge_lneg": "salt_bridge",
    "saltbridge_pneg": "salt_bridge",
    "halogen_bonds": "halogen_bond",
    "water_bridges": "water_bridge",
    "metal_complexes": "metal_complex",
}


# ============================================================
# LIGAND ID + SMILES / 3D / PDBQT PREPARATION
# ============================================================

def _sha256_smiles(smiles: str) -> str:
    return hashlib.sha256(str(smiles).encode("utf-8")).hexdigest()


def add_ligand_id_column(
    df: pd.DataFrame,
    ligand_id_col: str = LIGAND_ID_COL,
    zinc_id_col: str = ZINC_ID_COL,
    smiles_col: str = SMILES_COL,
) -> pd.DataFrame:
    """Ensure a ligand_id column exists.

    Priority:
      1. existing ligand_id
      2. zinc_id, copied into ligand_id
      3. SHA256(smiles)
    """
    df = df.copy()
    if ligand_id_col in df.columns:
        return df

    if zinc_id_col in df.columns:
        df[ligand_id_col] = df[zinc_id_col].astype(str)
        return df

    if smiles_col not in df.columns:
        raise ValueError(f"Input CSV must contain '{smiles_col}' to generate ligand IDs.")

    df[ligand_id_col] = df[smiles_col].apply(_sha256_smiles)
    return df


def generate_3d_conformer(smiles: str) -> Chem.Mol:
    """SMILES -> RDKit molecule -> explicit H -> 3D conformer -> minimized geometry."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    # Deterministic first attempt.
    if AllChem.EmbedMolecule(mol, randomSeed=0xF00D) != 0:
        # Fallback to ETKDG.
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            raise RuntimeError("Failed to generate 3D conformer")

    # Minimize geometry. If MMFF cannot parameterize the molecule, try UFF.
    try:
        status = AllChem.MMFFOptimizeMolecule(mol)
        if status not in (0, 1):
            # RDKit can return a non-success code for unusual cases.
            AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception as e:
            raise RuntimeError(f"Geometry optimization failed: {e}") from e

    return mol


def rdkit_mol_to_pdbqt_text(mol: Chem.Mol) -> str:
    """Convert a prepared RDKit 3D molecule to PDBQT using Meeko.

    Supports both the tuple-returning and string-returning forms of
    PDBQTWriterLegacy.write_string seen across Meeko versions.
    """
    preparator = MoleculePreparation()

    # Current Meeko documents both call-style and prepare-style APIs in the
    # broader ecosystem; use prepare() when available, then fall back to call.
    if hasattr(preparator, "prepare"):
        mol_setups = preparator.prepare(mol)
    else:
        mol_setups = preparator(mol)

    if not mol_setups:
        raise RuntimeError("Meeko produced no molecule setup for this ligand.")

    setup = mol_setups[0]
    result = PDBQTWriterLegacy.write_string(setup)

    if isinstance(result, tuple):
        # Older/alternate API: (pdbqt_string, is_ok, err_msg)
        if len(result) == 3:
            pdbqt_string, is_ok, err_msg = result
            if not is_ok:
                raise RuntimeError(f"Meeko failed to write PDBQT: {err_msg}")
            return pdbqt_string
        if len(result) == 1:
            return str(result[0])

    # Current API documented by Meeko returns the PDBQT string directly.
    if not isinstance(result, str):
        raise RuntimeError(f"Unexpected Meeko PDBQT writer return type: {type(result)!r}")
    return result


def prepare_ligand(smiles: str) -> str:
    mol3d = generate_3d_conformer(smiles)
    return rdkit_mol_to_pdbqt_text(mol3d)


def prepare_all_ligands(
    df: pd.DataFrame,
    smiles_col: str = SMILES_COL,
) -> pd.DataFrame:
    """Prepare every ligand; failures are recorded instead of aborting the batch."""
    df = df.copy()
    pdbqt_values: List[Optional[str]] = []
    errors: List[str] = []

    for i, smi in enumerate(df[smiles_col]):
        if pd.isna(smi):
            pdbqt_values.append(None)
            errors.append("Missing SMILES")
            continue

        smi = str(smi).strip()
        try:
            pdbqt_values.append(prepare_ligand(smi))
            errors.append("")
        except Exception as e:
            print(f"[!] Failed to prepare ligand {i} ({smi}): {e}")
            pdbqt_values.append(None)
            errors.append(str(e))

    df[PDBQT_COL] = pdbqt_values
    df[PDBQT_ERROR_COL] = errors
    return df


# ============================================================
# VINA DOCKING
# ============================================================

def dock_with_vina(
    vina_bin: str,
    ligand_pdbqt: str,
    receptor_pdbqt: str,
    out_pdbqt: str,
    log_path: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    exhaustiveness: int = DEFAULT_EXHAUSTIVENESS,
    cpu: int = DEFAULT_CPU_PER_JOB,
    timeout: int = DEFAULT_TIMEOUT,
) -> None:
    if not os.path.exists(vina_bin):
        raise RuntimeError(f"Vina binary not found at {vina_bin}")

    cx, cy, cz = center
    sx, sy, sz = size

    # NOTE: AutoDock Vina 1.2.x does NOT support the --log option (it was
    # removed after 1.1.2). Passing it makes Vina reject the command line and
    # exit with status 1. Vina's stdout/stderr are captured into log_path below.
    cmd = [
        vina_bin,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--out", out_pdbqt,
        "--center_x", str(cx),
        "--center_y", str(cy),
        "--center_z", str(cz),
        "--size_x", str(sx),
        "--size_y", str(sy),
        "--size_z", str(sz),
        "--exhaustiveness", str(exhaustiveness),
        "--cpu", str(cpu),
        "--num_modes", "1",
    ]

    # Capture stdout+stderr into the log file; do NOT use check=True so we can
    # read Vina's real error message and include it in the exception.
    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
        )

    if proc.returncode != 0:
        tail = Path(log_path).read_text(errors="replace")[-1500:]
        raise RuntimeError(f"Vina exited with status {proc.returncode}:\n{tail}")


def parse_vina_score_from_log(log_path: str) -> Optional[float]:
    """Parse the top mode affinity from a Vina log.

    Primary parser: look for the first docking-table row beginning with mode 1.
    Fallback: look for a standard table row containing an integer mode followed
    by a numeric affinity. Avoids taking arbitrary coordinates/metadata numbers.
    """
    if not os.path.exists(log_path):
        return None

    lines = Path(log_path).read_text(errors="replace").splitlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "1":
            try:
                return float(parts[1])
            except ValueError:
                pass

    # More forgiving fallback for minor formatting differences.
    for line in lines:
        m = re.match(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    return None


# ============================================================
# PDB/PDBQT HELPERS
# ============================================================

def _infer_element(atom_name: str, line: str) -> str:
    """Best-effort PDB element inference from PDBQT atom type, then atom name."""
    tail = line[66:].strip().split()
    if tail:
        token = re.sub(r"[^A-Za-z]", "", tail[-1]).upper()
        if token in AD_TYPE_TO_ELEMENT:
            return AD_TYPE_TO_ELEMENT[token]

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
    """Convert the first/top PDBQT pose to a clean HETATM-only PDB ligand block."""
    if not os.path.exists(pdbqt_path) or os.path.getsize(pdbqt_path) < 50:
        raise RuntimeError(f"Vina output missing/empty: {pdbqt_path}")

    atom_lines: List[str] = []
    serial_out = 1

    with open(pdbqt_path, "r") as f:
        for line in f:
            if line.startswith(("ENDMDL", "MODEL")):
                if line.startswith("ENDMDL"):
                    break
                # Ignore MODEL line itself.
                continue

            if not line.startswith(("ATOM", "HETATM")):
                continue

            try:
                atom_name = line[12:16]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (IndexError, ValueError) as e:
                raise RuntimeError(f"Could not parse PDBQT atom line: {line.rstrip()}") from e

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
    """Keep receptor ATOM/HETATM records except excluded residue names."""
    keep: List[str] = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            resname = line[17:20].strip()
            if resname in EXCLUDE_RESNAMES:
                continue
            keep.append(line)
    return "\n".join(keep) + "\n"


def write_complex_pdb(protein_pdb_text: str, ligand_pdb_text: str, out_path: str) -> None:
    """Write protein + ligand with a single final END record."""
    with open(out_path, "w") as f:
        f.write(protein_pdb_text)
        if not protein_pdb_text.rstrip().endswith("TER"):
            f.write("TER\n")
        f.write(ligand_pdb_text)


# ============================================================
# PLIP ANALYSIS
# ============================================================

def run_plip_all_residues(complex_pdb: str) -> Dict[str, Set[str]]:
    """Return residue label -> exact PLIP interaction type(s)."""
    pc = PDBComplex()
    pc.load_pdb(complex_pdb)
    pc.analyze()

    interacting: Dict[str, Set[str]] = {}

    for _, interactions in pc.interaction_sets.items():
        for attr, interaction_label in PLIP_ATTRS.items():
            arr = getattr(interactions, attr, None)
            if not arr:
                continue

            for entry in arr:
                resnr = (
                    getattr(entry, "resnr", None)
                    or getattr(entry, "res_seq", None)
                    or getattr(entry, "resid", None)
                )
                restype = (
                    getattr(entry, "restype", None)
                    or getattr(entry, "resname", None)
                    or getattr(entry, "residue", None)
                )

                if restype is None or resnr is None:
                    continue

                try:
                    res_label = f"{str(restype).upper()}{int(resnr)}"
                except (TypeError, ValueError):
                    continue

                interacting.setdefault(res_label, set()).add(interaction_label)

    return interacting


# ============================================================
# JOB STRUCTURE
# ============================================================

@dataclass
class DockJob:
    ligand_id: str
    zinc_id: Optional[str]
    smiles: Optional[str]
    predicted_delta_pic50: Optional[float]
    qsar_rank: Optional[float]
    ligand_pdbqt: str
    iso_name: str
    receptor: Dict
    exhaustiveness: int
    cpu_per_job: int
    timeout: int
    vina_bin: str
    input_order: int
    complex_out_dir: str


def _run_one_job(job: DockJob) -> Dict:
    """Run one independent (ligand, isoform) docking + PLIP job."""
    job_dir = Path(tempfile.mkdtemp(prefix=f"vina_{job.iso_name}_"))
    lig_in = job_dir / "lig.pdbqt"
    out_pdbqt = job_dir / "docked.pdbqt"
    log_path = job_dir / "vina.log"

    unique_suffix = uuid.uuid4().hex[:6]
    safe_ligand = re.sub(r"[^A-Za-z0-9_.-]", "_", job.ligand_id)
    saved_complex_path = Path(job.complex_out_dir) / (
        f"{safe_ligand[:32]}_{job.iso_name}_{unique_suffix}_complex.pdb"
    )

    try:
        with open(lig_in, "w") as f:
            f.write(job.ligand_pdbqt)

        dock_with_vina(
            vina_bin=job.vina_bin,
            ligand_pdbqt=str(lig_in),
            receptor_pdbqt=job.receptor["pdbqt"],
            out_pdbqt=str(out_pdbqt),
            log_path=str(log_path),
            center=tuple(job.receptor["center"]),
            size=tuple(job.receptor["size"]),
            exhaustiveness=job.exhaustiveness,
            cpu=job.cpu_per_job,
            timeout=job.timeout,
        )

        score = parse_vina_score_from_log(str(log_path))
        if score is None:
            raise RuntimeError("Could not parse a Vina affinity from the log.")

        ligand_pose_pdb = pdbqt_pose_to_ligand_pdb_string(str(out_pdbqt))

        with open(job.receptor["pdb"], "r") as f_rec:
            receptor_pdb_text = f_rec.read()

        protein_pdb_text = protein_only_pdb(receptor_pdb_text)
        write_complex_pdb(protein_pdb_text, ligand_pose_pdb, str(saved_complex_path))

        residue_interactions = run_plip_all_residues(str(saved_complex_path))

        return {
            "input_order": job.input_order,
            "ligand_id": job.ligand_id,
            "zinc_id": job.zinc_id,
            "SMILES": job.smiles,
            "predicted_delta_pIC50": job.predicted_delta_pic50,
            "QSAR_rank": job.qsar_rank,
            "iso_name": job.iso_name,
            "vina_score": score,
            "residue_interactions": residue_interactions,
            "complex_path": str(saved_complex_path),
            "error": None,
        }

    except Exception as e:
        return {
            "input_order": job.input_order,
            "ligand_id": job.ligand_id,
            "zinc_id": job.zinc_id,
            "SMILES": job.smiles,
            "predicted_delta_pIC50": job.predicted_delta_pic50,
            "QSAR_rank": job.qsar_rank,
            "iso_name": job.iso_name,
            "vina_score": None,
            "residue_interactions": {},
            "complex_path": None,
            "error": str(e),
        }

    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


# ============================================================
# CHECKPOINT / RESUME
# ============================================================

def _checkpoint_key(ligand_id: str, iso_name: str) -> Tuple[str, str]:
    return str(ligand_id), str(iso_name)


def _serialize_checkpoint_result(result: Dict) -> Dict:
    out = dict(result)
    interactions = result.get("residue_interactions", {}) or {}
    out["residue_interactions"] = {
        str(res): sorted(str(x) for x in types)
        for res, types in interactions.items()
    }
    return out


def _deserialize_checkpoint_result(record: Dict) -> Dict:
    out = dict(record)
    interactions = record.get("residue_interactions", {}) or {}
    out["residue_interactions"] = {
        str(res): set(types) for res, types in interactions.items()
    }
    return out


def load_checkpoint(checkpoint_path: Path) -> Tuple[List[Dict], Set[Tuple[str, str]]]:
    """Load successful jobs from JSONL; ignore malformed/incomplete final lines."""
    records: List[Dict] = []
    completed: Set[Tuple[str, str]] = set()

    if not checkpoint_path.exists():
        return records, completed

    with open(checkpoint_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = _deserialize_checkpoint_result(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

            if record.get("error") is not None:
                continue

            if "ligand_id" not in record or "iso_name" not in record:
                continue

            key = _checkpoint_key(record["ligand_id"], record["iso_name"])
            if key in completed:
                continue

            records.append(record)
            completed.add(key)

    return records, completed


def append_checkpoint(checkpoint_path: Path, result: Dict) -> None:
    """Append one successful result from the parent process."""
    record = _serialize_checkpoint_result(result)
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ============================================================
# MAIN SCREENING
# ============================================================

def _safe_float(value) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        val = float(value)
        if math.isnan(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


def build_jobs(
    ligand_df: pd.DataFrame,
    receptor_map: Dict[str, Dict],
    workers_config: Dict,
    completed_keys: Set[Tuple[str, str]],
) -> List[DockJob]:
    jobs: List[DockJob] = []

    has_zinc = ZINC_ID_COL in ligand_df.columns
    has_smiles = SMILES_COL in ligand_df.columns
    has_predicted_delta = PREDICTED_DELTA_COL in ligand_df.columns
    has_rank = QSAR_RANK_COL in ligand_df.columns

    for input_order, (_, row) in enumerate(ligand_df.iterrows()):
        ligand_id = str(row[LIGAND_ID_COL])
        ligand_pdbqt = row[PDBQT_COL]
        smiles = None if not has_smiles or pd.isna(row[SMILES_COL]) else str(row[SMILES_COL])
        zinc_id = None if not has_zinc or pd.isna(row[ZINC_ID_COL]) else str(row[ZINC_ID_COL])
        predicted_delta = (
            _safe_float(row[PREDICTED_DELTA_COL]) if has_predicted_delta else None
        )
        qsar_rank = _safe_float(row[QSAR_RANK_COL]) if has_rank else None

        if not isinstance(ligand_pdbqt, str) or not ligand_pdbqt.strip():
            print(f"[!] Skipping docking for {ligand_id}: no valid PDBQT was prepared.")
            continue

        for iso_name, receptor in receptor_map.items():
            if _checkpoint_key(ligand_id, iso_name) in completed_keys:
                continue

            jobs.append(
                DockJob(
                    ligand_id=ligand_id,
                    zinc_id=zinc_id,
                    smiles=smiles,
                    predicted_delta_pic50=predicted_delta,
                    qsar_rank=qsar_rank,
                    ligand_pdbqt=ligand_pdbqt,
                    iso_name=iso_name,
                    receptor=receptor,
                    exhaustiveness=workers_config["exhaustiveness"],
                    cpu_per_job=workers_config["cpu_per_job"],
                    timeout=workers_config["timeout"],
                    vina_bin=workers_config["vina_bin"],
                    input_order=input_order,
                    complex_out_dir=workers_config["complex_out_dir"],
                )
            )

    return jobs


def make_wide_results(
    raw_results: Dict[str, List[Dict]],
) -> Dict[str, pd.DataFrame]:
    """Convert per-job results to one wide CSV-shaped DataFrame per isoform."""
    wide_results: Dict[str, pd.DataFrame] = {}

    for iso_name, records in raw_results.items():
        records = sorted(records, key=lambda r: r.get("input_order", 10**18))

        # Union of residues observed for this isoform across all successful PLIP jobs.
        all_residues = sorted({
            res
            for rec in records
            for res in (rec.get("residue_interactions", {}) or {}).keys()
        })

        rows: List[Dict] = []
        for rec in records:
            row = {
                "ligand_id": rec.get("ligand_id"),
                "zinc_id": rec.get("zinc_id"),
                "SMILES": rec.get("SMILES"),
                "predicted_delta_pIC50": rec.get("predicted_delta_pIC50"),
                "QSAR_rank": rec.get("QSAR_rank"),
                "vina_score": rec.get("vina_score"),
            }

            interactions = rec.get("residue_interactions", {}) or {}
            for residue in all_residues:
                types = interactions.get(residue)
                row[residue] = (
                    INTERACTION_TYPE_SEP.join(sorted(types)) if types else ""
                )

            rows.append(row)

        df = pd.DataFrame(rows)
        ordered_cols = [
            "ligand_id",
            "zinc_id",
            "SMILES",
            "predicted_delta_pIC50",
            "QSAR_rank",
            "vina_score",
        ] + all_residues
        wide_results[iso_name] = df.reindex(columns=ordered_cols)

    return wide_results


def run_screening(
    ligand_df: pd.DataFrame,
    receptor_map: Dict[str, Dict],
    outdir: Path,
    vina_bin: str,
    workers: int,
    cpu_per_job: int,
    exhaustiveness: int,
    timeout: int,
    checkpoint_path: Path,
    resume: bool,
) -> Dict[str, pd.DataFrame]:
    complex_out_dir = outdir / COMPLEX_OUT_NAME
    complex_out_dir.mkdir(parents=True, exist_ok=True)

    raw_results: Dict[str, List[Dict]] = {iso: [] for iso in receptor_map}
    completed_keys: Set[Tuple[str, str]] = set()

    if resume:
        checkpoint_records, completed_keys = load_checkpoint(checkpoint_path)
        for record in checkpoint_records:
            iso_name = record.get("iso_name")
            if iso_name in raw_results:
                raw_results[iso_name].append(record)
        print(f"Resuming: loaded {len(completed_keys)} completed jobs from {checkpoint_path}")

    worker_config = {
        "exhaustiveness": exhaustiveness,
        "cpu_per_job": cpu_per_job,
        "timeout": timeout,
        "vina_bin": vina_bin,
        "complex_out_dir": str(complex_out_dir),
    }

    jobs = build_jobs(ligand_df, receptor_map, worker_config, completed_keys)

    total_possible = len(ligand_df) * len(receptor_map)
    print(
        f"Prepared {len(ligand_df)} input ligands -> "
        f"{total_possible} possible docking jobs across {len(receptor_map)} isoforms."
    )
    print(
        f"Running {len(jobs)} remaining jobs with {workers} worker process(es), "
        f"{cpu_per_job} Vina CPU(s)/job, exhaustiveness={exhaustiveness}."
    )

    if not jobs:
        return make_wide_results(raw_results)

    if workers <= 1:
        with tqdm.tqdm(total=len(jobs), desc="DOCKING") as pbar:
            for job in jobs:
                result = _run_one_job(job)
                if result["error"]:
                    print(
                        f"\nERROR {result['ligand_id']} x {result['iso_name']}: "
                        f"{result['error']}"
                    )
                else:
                    append_checkpoint(checkpoint_path, result)
                    completed_keys.add(
                        _checkpoint_key(result["ligand_id"], result["iso_name"])
                    )

                raw_results[result["iso_name"]].append(result)
                pbar.update(1)
    else:
        executor = cf.ProcessPoolExecutor(max_workers=workers)
        futures = [executor.submit(_run_one_job, job) for job in jobs]

        try:
            with tqdm.tqdm(total=len(jobs), desc="DOCKING") as pbar:
                for future in cf.as_completed(futures):
                    result = future.result()
                    if result["error"]:
                        print(
                            f"\nERROR {result['ligand_id']} x {result['iso_name']}: "
                            f"{result['error']}"
                        )
                    else:
                        append_checkpoint(checkpoint_path, result)
                        completed_keys.add(
                            _checkpoint_key(result["ligand_id"], result["iso_name"])
                        )

                    raw_results[result["iso_name"]].append(result)
                    pbar.update(1)

        except KeyboardInterrupt:
            print(
                "\nStopping requested. Completed jobs have already been checkpointed. "
                "Rerun with --resume to continue."
            )
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    return make_wide_results(raw_results)


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SMILES, dock ligands against JAK receptors, and run PLIP."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input compounds CSV containing at least a smiles column.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory where prepared CSV, result CSVs, checkpoint, and PLIP complexes are written.",
    )
    parser.add_argument(
        "--receptors",
        default=str(DEFAULT_RECEPTORS_JSON),
        help=f"Path to receptors.json. Default: {DEFAULT_RECEPTORS_JSON}",
    )
    parser.add_argument(
        "--vina",
        default=str(DEFAULT_VINA_BIN),
        help=f"Path to AutoDock Vina executable. Default: {DEFAULT_VINA_BIN}",
    )
    parser.add_argument(
        "--smiles-col",
        default=SMILES_COL,
        help=f"SMILES column name. Default: {SMILES_COL}",
    )
    parser.add_argument(
        "--id-col",
        default=ZINC_ID_COL,
        help=f"Preferred existing ligand identifier column if ligand_id is absent. Default: {ZINC_ID_COL}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of independent (ligand, isoform) processes. Default: 1",
    )
    parser.add_argument(
        "--cpu-per-job",
        type=int,
        default=DEFAULT_CPU_PER_JOB,
        help="Vina --cpu value inside each worker. Keep small when workers > 1. Default: 1",
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=DEFAULT_EXHAUSTIVENESS,
        help=f"Vina exhaustiveness when not overridden in receptors.json. Default: {DEFAULT_EXHAUSTIVENESS}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds per Vina job. Default: {DEFAULT_TIMEOUT}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N input rows (useful for testing).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume successful jobs from docking_checkpoint.jsonl in --outdir.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.cpu_per_job < 1:
        raise ValueError("--cpu-per-job must be >= 1")
    if args.exhaustiveness < 1:
        raise ValueError("--exhaustiveness must be >= 1")
    if args.timeout < 1:
        raise ValueError("--timeout must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = outdir / CHECKPOINT_NAME

    # Fresh run: don't silently mix an older checkpoint with a new input file
    # or changed docking settings.
    if not args.resume and checkpoint_path.exists():
        checkpoint_path.unlink()

    # Read input CSV.
    ligand_df = pd.read_csv(args.input)

    # Let --smiles-col override the default expected name by normalizing it to
    # the internal SMILES_COL key.
    if args.smiles_col not in ligand_df.columns:
        raise ValueError(
            f"Input CSV does not contain SMILES column '{args.smiles_col}'. "
            f"Available columns: {list(ligand_df.columns)}"
        )
    if args.smiles_col != SMILES_COL:
        ligand_df = ligand_df.rename(columns={args.smiles_col: SMILES_COL})

    # --id-col is the preferred fallback identifier if ligand_id doesn't exist.
    if LIGAND_ID_COL not in ligand_df.columns:
        if args.id_col in ligand_df.columns:
            ligand_df = ligand_df.rename(columns={args.id_col: ZINC_ID_COL})
        ligand_df = add_ligand_id_column(
            ligand_df,
            ligand_id_col=LIGAND_ID_COL,
            zinc_id_col=ZINC_ID_COL,
            smiles_col=SMILES_COL,
        )

    if args.limit is not None:
        ligand_df = ligand_df.head(args.limit).copy()

    # Prepare PDBQT in memory, then write the prepared CSV to the requested
    # output directory. This is the exact CSV used for the docking stage.
    print(f"Preparing {len(ligand_df)} ligands from '{args.input}' ...")
    ligand_df = prepare_all_ligands(ligand_df, smiles_col=SMILES_COL)

    n_failed = ligand_df[PDBQT_COL].isna().sum()
    prepared_path = outdir / f"{Path(args.input).stem}_prepared.csv"
    ligand_df.to_csv(prepared_path, index=False)
    print(
        f"Prepared {len(ligand_df) - int(n_failed)}/{len(ligand_df)} ligands successfully."
    )
    print(f"Wrote prepared ligands: {prepared_path}")

    # Load receptor definitions.
    receptors_path = Path(args.receptors).expanduser().resolve()
    with open(receptors_path, "r") as f:
        receptor_map = json.load(f)

    if not isinstance(receptor_map, dict) or not receptor_map:
        raise ValueError("receptors.json must contain a non-empty JSON object of isoform definitions.")

    # Basic receptor validation before launching a large screen.
    # Also resolve receptor file paths to absolute paths so worker processes
    # don't depend on the current working directory. Relative paths are tried
    # against the CWD first, then against the receptors.json directory, then
    # against PROJECT_ROOT.
    def _resolve_path(p: str) -> str:
        p_obj = Path(p).expanduser()
        if p_obj.is_absolute():
            return str(p_obj)
        for base in (Path.cwd(), receptors_path.parent, PROJECT_ROOT):
            candidate = (base / p_obj).resolve()
            if candidate.exists():
                return str(candidate)
        return str(p_obj.resolve())

    for iso_name, rec in receptor_map.items():
        for key in ("pdbqt", "pdb", "center", "size"):
            if key not in rec:
                raise ValueError(f"Receptor '{iso_name}' is missing required key '{key}'.")
        rec["pdbqt"] = _resolve_path(rec["pdbqt"])
        rec["pdb"] = _resolve_path(rec["pdb"])
        if not os.path.exists(rec["pdbqt"]):
            raise FileNotFoundError(f"{iso_name} receptor PDBQT not found: {rec['pdbqt']}")
        if not os.path.exists(rec["pdb"]):
            raise FileNotFoundError(f"{iso_name} receptor PDB not found: {rec['pdb']}")

    print(f"Isoforms found: {', '.join(receptor_map.keys())}")

    wide_results = run_screening(
        ligand_df=ligand_df,
        receptor_map=receptor_map,
        outdir=outdir,
        vina_bin=str(Path(args.vina).expanduser()),
        workers=args.workers,
        cpu_per_job=args.cpu_per_job,
        exhaustiveness=args.exhaustiveness,
        timeout=args.timeout,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )

    for iso_name, df in wide_results.items():
        safe_iso = re.sub(r"[^A-Za-z0-9_.-]", "_", str(iso_name))
        out_path = outdir / f"{safe_iso}_docking_results.csv"
        df.to_csv(out_path, index=False)

        residue_count = max(df.shape[1] - 6, 0)
        print(
            f"Wrote {out_path} "
            f"({df.shape[0]} ligands x {residue_count} residue columns)"
        )

    print("Screening complete.")


if __name__ == "__main__":
    main()