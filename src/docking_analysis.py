"""
docking_analysis.py

(this is the updated version to record PLIP interaction types)
=========================
Step 3 of 3 in the JAK docking pipeline.

Key fixes vs the original version:
1) Strip receptor hetero-residues by RESIDUE NAME (not by ATOM/HETATM record
   type) before PLIP so PTR/co-crystal ligands do not get mistaken for the
   docked ligand. Modified residues like phosphotyrosine are frequently
   emitted as ATOM records by receptor-prep tools, so filtering on record
   type alone lets them through.
2) Do not round-trip the docked pose through RDKit for the final complex PDB.
   Instead, convert the top-pose PDBQT directly to a proper HETATM ligand PDB.
3) Preserve a real chain ID and residue name for the docked ligand so PLIP can
   recognize it as a ligand.
4) Only write a single TER/END pair at the very end of the merged complex
   file. Previously the protein block ended in its own END record and the
   ligand block was appended after it, so parsers that stop reading at END
   (including OpenBabel, which PLIP uses for bond perception) never saw the
   ligand at all -- this was the actual root cause of PLIP reporting PTR
   contacts and the ligand rendering as broken/disconnected.
5) Map AutoDock/Vina PDBQT atom types (HD, HS, NA, OA, SA, A, ...) to correct
   PDB element symbols via an explicit table instead of blindly title-casing
   the type string, which previously produced invalid/wrong elements (e.g.
   NA -> "Na" sodium, SA -> "Sa", HS -> "Hs" hassium) and broke bond
   perception in viewers and in PLIP itself.
6) Make file paths relative to the script location by default.
7) Parallelize across (ligand, isoform) jobs with a process pool (--workers).
   Each job is fully independent (its own temp dir, its own output file), so
   this scales close to linearly with core count. Uses multiprocessing, not
   threading, since PLIP's global config state and OpenBabel bindings are
   safer isolated per-process than shared across threads in one interpreter.
8) Complexes are now saved under saved_complexes_PLIP/ (previously
   saved_complexes/), to keep PLIP-analyzed complexes clearly separated from
   any other complex outputs.
9) Per-isoform result CSVs now carry SMILES and delta_pIC50 alongside
   ligand_id, in addition to the vina_score column. SMILES is pulled
   straight from the input ligand CSV's "smiles" column. delta_pIC50 is
   *derived*, not read directly: it's computed from the "JAK1_IC50_nM" and
   "JAK2_IC50_nM" columns as pIC50(JAK2) - pIC50(JAK1), so more positive
   values indicate greater JAK2 selectivity.
10) Residue columns no longer hold a binary 0/1 "did this residue interact
    at all" flag. Instead each cell holds the exact PLIP interaction type(s)
    observed for that ligand/residue pair (e.g. "hydrogen_bond" or
    "hydrophobic;pi_stacking" when more than one type is seen), and is left
    blank when that residue did not interact with that ligand's pose.

Reads:
  - receptors.json         (prepared receptor paths + docking box center/size)
  - ligand CSV              (ligand_id + PDBQT columns, plus "smiles",
                              "JAK1_IC50_nM", and "JAK2_IC50_nM" columns used
                              for reporting / delta_pIC50 only)

For every ligand x isoform pair:
  1) Dock with AutoDock Vina
  2) Parse the top-pose binding score from the Vina log
  3) Convert the top pose PDBQT -> PDB with a direct writer
  4) Merge with a protein-only receptor PDB into a complex file
  5) Run PLIP on the complex and record the exact interaction type(s) for
     every interacting amino acid

Output: one wide CSV per isoform with ligand_id, SMILES, delta_pIC50,
vina_score, then one column per residue seen in that isoform, holding the
exact interaction type(s) observed (semicolon-separated if multiple, blank
if none).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math

import pandas as pd
import tqdm

from plip.structure.preparation import PDBComplex

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RECEPTORS_JSON = PROJECT_ROOT / "docking" / "docking_prep" / "receptors.json"
# (1) Complexes analyzed by PLIP now live in their own, clearly-named folder.
COMPLEX_OUT_DIR = PROJECT_ROOT / "saved_complexes_PLIP"
DEFAULT_EXHAUSTIVENESS = 32
VINA_BIN = os.path.expanduser("~/bin/vina")

# Ligand residue naming used in the merged complex PDB.
LIGAND_RESNAME = "LIG"
LIGAND_CHAIN_ID = "Z"
LIGAND_RESSEQ = 1

# Column names expected on the input ligand CSV. Adjust here if your CSV
# uses different headers.
LIGAND_ID_COL = "ligand_id"
LIGAND_PDBQT_COL = "PDBQT"
LIGAND_SMILES_COL = "smiles"
LIGAND_JAK1_IC50_NM_COL = "JAK1_IC50_nM"
LIGAND_JAK2_IC50_NM_COL = "JAK2_IC50_nM"

# delta_pIC50 is not a column on the input CSV -- it's derived from the
# JAK1/JAK2 IC50 (nM) columns as:
#     pIC50 = 9 - log10(IC50_nM)          [i.e. -log10(IC50 in molar)]
#     delta_pIC50 = pIC50(JAK2) - pIC50(JAK1)
# A compound that is more JAK2-selective is more potent on JAK2 (smaller
# JAK2_IC50_nM -> larger pIC50(JAK2)) and/or less potent on JAK1 (larger
# JAK1_IC50_nM -> smaller pIC50(JAK1)), so this difference increases as
# JAK2 selectivity increases -- matching "more positive = more JAK2
# selective."

# Separator used when more than one interaction type is observed between a
# given ligand pose and a given residue (e.g. both a hydrogen bond and a
# hydrophobic contact to the same residue).
INTERACTION_TYPE_SEP = ";"

# Residue names to strip from the receptor before merging with the docked
# ligand. These are modified/non-canonical residues and crystallographic
# extras that PLIP can otherwise mistake for the docked ligand. Extend this
# set if your receptors carry other modified residues or cofactors.
EXCLUDE_RESNAMES = {
    "PTR",  # phosphotyrosine
    "SEP",  # phosphoserine
    "TPO",  # phosphothreonine
    "HOH",  # water
    "MSE",  # selenomethionine (usually fine to keep, but excluded by default)
}

# AutoDock/Vina PDBQT atom types -> correct PDB element symbol.
# Blindly title-casing the raw type string is wrong for several common types
# (e.g. "NA" is an N acceptor, not sodium; "SA" is an S acceptor, not
# tantalum's neighbor; "HD"/"HS" are polar/non-polar hydrogens, not the
# elements Hd/Hs).
AD_TYPE_TO_ELEMENT = {
    "A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N",
    "O": "O", "OA": "O", "OS": "O", "S": "S", "SA": "S",
    "H": "H", "HD": "H", "HS": "H",
    "F": "F", "CL": "Cl", "BR": "Br", "I": "I",
    "MG": "Mg", "CA": "Ca", "MN": "Mn", "FE": "Fe", "ZN": "Zn",
    "P": "P", "SI": "Si", "B": "B",
}

# Maps (attribute name on a PLIP PLInteraction object) -> (interaction label).
# NOTE: PLIP splits several interaction types into two attributes (donor vs.
# acceptor side) rather than one combined list. Verified against the
# installed plip package (plip/structure/detection.py); getattr() on a wrong
# name returns None silently, so a mismatch here does not raise an error --
# it just quietly drops that whole interaction type from every result.
def _ic50_nm_to_pic50(ic50_nm) -> Optional[float]:
    """
    Convert an IC50 in nanomolar to pIC50 = -log10(IC50 in molar)
    = 9 - log10(IC50_nM). Returns None for missing/non-positive/unparsable
    values rather than raising, since a single bad IC50 shouldn't take down
    delta_pIC50 for every ligand.
    """
    try:
        val = float(ic50_nm)
    except (TypeError, ValueError):
        return None
    if val <= 0 or math.isnan(val):
        return None
    return 9.0 - math.log10(val)

def compute_delta_pic50(jak1_ic50_nm, jak2_ic50_nm) -> Optional[float]:
    """
    delta_pIC50 = pIC50(JAK2) - pIC50(JAK1).

    More positive => more potent on JAK2 relative to JAK1 => more
    JAK2-selective. Returns None if either IC50 is missing/invalid so a
    partial row doesn't silently produce a misleading number.
    """
    pic50_jak1 = _ic50_nm_to_pic50(jak1_ic50_nm)
    pic50_jak2 = _ic50_nm_to_pic50(jak2_ic50_nm)
    if pic50_jak1 is None or pic50_jak2 is None:
        return None
    return pic50_jak2 - pic50_jak1

PLIP_ATTRS = {
    "hbonds_pdon": "hydrogen_bond",         # protein is the H-bond donor
    "hbonds_ldon": "hydrogen_bond",         # ligand is the H-bond donor
    "hydrophobic_contacts": "hydrophobic",
    "pistacking": "pi_stacking",
    "pication_laro": "pi_cation",           # ligand aromatic ring, protein cation
    "pication_paro": "pi_cation",           # protein aromatic ring, ligand cation
    "saltbridge_lneg": "salt_bridge",       # ligand carries the negative charge
    "saltbridge_pneg": "salt_bridge",       # protein carries the negative charge
    "halogen_bonds": "halogen_bond",
    "water_bridges": "water_bridge",
    "metal_complexes": "metal_complex",
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
    exhaustiveness: int = 16,
    cpu: int = 1,
    timeout: int = 900,
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
    Prefer the explicit AutoDock atom-type token, mapped through
    AD_TYPE_TO_ELEMENT, since that type encodes hybridization/H-bonding role
    rather than the element directly (e.g. "NA" = N acceptor, not sodium).
    Fall back to inferring from the atom name only if the type token isn't
    recognized.
    """
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
    """
    Convert the top docked pose PDBQT into a clean HETATM-only PDB ligand block.
    This preserves coordinates but intentionally normalizes residue/chain naming
    so PLIP sees the docked molecule as the ligand.

    Note: this returns a block WITHOUT a leading END record from any prior
    section, and includes its own trailing TER/END. Callers must make sure no
    earlier END record precedes this block in the same file, or downstream
    parsers (including PLIP's OpenBabel-based bond perception) will stop
    reading before ever seeing these atoms.
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
    Keep only protein ATOM/HETATM records from the receptor PDB, filtered by
    residue name rather than by record type. Modified residues such as
    phosphotyrosine (PTR) are frequently emitted as ATOM records (not
    HETATM) by receptor-prep tools so they stay part of the polypeptide
    chain, so filtering on the record type alone does not remove them. This
    also drops waters and other excluded hetero groups.

    Deliberately does NOT append a TER/END record -- the caller is
    responsible for writing exactly one TER/END pair after the ligand block
    is appended, so no parser stops reading before reaching the ligand.
    """
    keep: List[str] = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            resname = line[17:20].strip()
            if resname in EXCLUDE_RESNAMES:
                continue
            keep.append(line)
    return "\n".join(keep) + "\n"

def write_complex_pdb(protein_pdb_text: str, ligand_pdb_text: str, out_path: str) -> None:
    """
    Write receptor + ligand into a single, well-formed complex PDB with
    exactly one TER/END pair at the very end. protein_pdb_text must NOT
    already contain an END record (see protein_only_pdb), otherwise parsers
    that stop at the first END -- including OpenBabel, which PLIP relies on
    for bond perception -- will silently never see the ligand atoms.
    """
    with open(out_path, "w") as f:
        f.write(protein_pdb_text)
        if not protein_pdb_text.rstrip().endswith("TER"):
            f.write("TER\n")
        # ligand_pdb_text already ends with its own TER/END
        f.write(ligand_pdb_text)

# ============================================================
# PLIP ANALYSIS
# ============================================================

def run_plip_all_residues(complex_pdb: str) -> Dict[str, set]:
    """
    Run PLIP on a receptor-ligand complex and return a mapping of
    residue label (e.g. "TYR231") -> set of exact PLIP interaction type
    labels observed for that residue (e.g. {"hydrogen_bond", "hydrophobic"}).

    This replaces the old binary "did this residue interact at all" flag:
    callers now get the precise interaction type(s) per residue instead of
    a single 0/1 flag, so a residue that forms both a hydrogen bond and a
    hydrophobic contact is distinguishable from one that only does one or
    the other.
    """
    pc = PDBComplex()
    pc.load_pdb(complex_pdb)
    pc.analyze()

    interacting: Dict[str, set] = {}
    for _, interactions in pc.interaction_sets.items():
        for attr, interaction_label in PLIP_ATTRS.items():
            arr = getattr(interactions, attr, None)
            if not arr:
                continue
            for entry in arr:
                resnr = getattr(entry, "resnr", None) or getattr(entry, "res_seq", None) or getattr(entry, "resid", None)
                restype = getattr(entry, "restype", None) or getattr(entry, "resname", None) or getattr(entry, "residue", None)
                if restype and resnr:
                    try:
                        res_label = f"{str(restype).upper()}{int(resnr)}"
                    except (TypeError, ValueError):
                        continue
                    interacting.setdefault(res_label, set()).add(interaction_label)
    return interacting

# ============================================================
# PER-JOB WORKER (runs inside a worker process)
# ============================================================

@dataclass
class DockJob:
    """One (ligand, isoform) unit of work. Must be picklable to cross the
    process-pool boundary, so it carries plain data only -- no open file
    handles, no PDBComplex objects, etc."""
    ligand_id: str
    ligand_pdbqt: str
    iso_name: str
    receptor: Dict
    exhaustiveness: int
    cpu_per_job: int
    smiles: Optional[str] = None
    delta_pic50: Optional[float] = None

def _run_one_job(job: DockJob) -> Dict:
    """
    Dock one ligand against one isoform and run PLIP on the result. This is a
    module-level function (not a method/closure) so it can be pickled and
    sent to a separate worker process by ProcessPoolExecutor.

    Each call gets its OWN private temp directory (tempfile.mkdtemp) rather
    than sharing one workspace_root across jobs, so concurrent workers never
    write into each other's files -- important now that jobs run in parallel
    instead of one at a time. The saved complex filename also gets a short
    random suffix for the same reason: two different ligand_ids that happen
    to share their first 8 characters would otherwise be able to race on the
    same output path when run concurrently.
    """
    job_dir = Path(tempfile.mkdtemp(prefix=f"vina_{job.iso_name}_"))
    lig_in = job_dir / "lig.pdbqt"
    out_pdbqt = job_dir / "docked.pdbqt"
    log_path = job_dir / "vina.log"
    unique_suffix = uuid.uuid4().hex[:6]
    saved_complex_path = COMPLEX_OUT_DIR / f"{job.ligand_id[:8]}_{job.iso_name}_{unique_suffix}_complex.pdb"

    try:
        with open(lig_in, "w") as f:
            f.write(job.ligand_pdbqt)

        dock_with_vina(
            str(lig_in),
            job.receptor["pdbqt"],
            str(out_pdbqt),
            str(log_path),
            center=tuple(job.receptor["center"]),
            size=tuple(job.receptor["size"]),
            exhaustiveness=job.exhaustiveness,
            cpu=job.cpu_per_job,
        )

        score = parse_vina_score_from_log(str(log_path))
        ligand_pose_pdb = pdbqt_pose_to_ligand_pdb_string(str(out_pdbqt))

        with open(job.receptor["pdb"], "r") as f_rec:
            receptor_pdb_text = f_rec.read()
        protein_pdb_text = protein_only_pdb(receptor_pdb_text)

        write_complex_pdb(protein_pdb_text, ligand_pose_pdb, str(saved_complex_path))

        residue_interactions = run_plip_all_residues(str(saved_complex_path))

        return {
            "ligand_id": job.ligand_id,
            "smiles": job.smiles,
            "delta_pic50": job.delta_pic50,
            "iso_name": job.iso_name,
            "vina_score": score,
            "residue_interactions": residue_interactions,
            "error": None,
        }

    except Exception as e:
        return {
            "ligand_id": job.ligand_id,
            "smiles": job.smiles,
            "delta_pic50": job.delta_pic50,
            "iso_name": job.iso_name,
            "vina_score": None,
            "residue_interactions": {},
            "error": str(e),
        }
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

# ============================================================
# CHECKPOINT / RESUME HELPERS
# ============================================================

def _checkpoint_key(ligand_id: str, iso_name: str) -> Tuple[str, str]:
    return (str(ligand_id), str(iso_name))


def _serialize_checkpoint_result(result: Dict) -> Dict:
    """Make a completed result JSON-serializable for persistent checkpointing."""
    out = dict(result)
    interactions = result.get("residue_interactions", {}) or {}
    out["residue_interactions"] = {
        str(res): sorted(str(x) for x in types)
        for res, types in interactions.items()
    }
    return out


def _deserialize_checkpoint_result(record: Dict) -> Dict:
    """Restore residue-interaction lists from JSON back to sets."""
    out = dict(record)
    interactions = record.get("residue_interactions", {}) or {}
    out["residue_interactions"] = {
        str(res): set(types)
        for res, types in interactions.items()
    }
    return out


def load_checkpoint(checkpoint_path: Path) -> Tuple[List[Dict], set]:
    """Load successful completed jobs from a JSONL checkpoint file."""
    records: List[Dict] = []
    completed = set()
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
                # Ignore a malformed/incomplete final line rather than losing
                # the rest of the checkpoint.
                continue

            # Only successful jobs are resumable. Failed jobs should be retried
            # on the next run rather than permanently skipped.
            if record.get("error") is not None:
                continue

            key = _checkpoint_key(record["ligand_id"], record["iso_name"])
            if key in completed:
                continue
            records.append(record)
            completed.add(key)

    return records, completed


def append_checkpoint(checkpoint_path: Path, result: Dict) -> None:
    """Append one successful completed job atomically enough for normal use."""
    record = _serialize_checkpoint_result(result)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    # Write the new JSON record to a temporary file, then append its contents.
    # The checkpoint is deliberately append-only so a stopped run retains all
    # previously completed jobs.
    with open(tmp_path, "w") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    with open(checkpoint_path, "a") as f:
        f.write(tmp_path.read_text())
        f.flush()
        os.fsync(f.fileno())
    tmp_path.unlink(missing_ok=True)


# ============================================================
# MAIN DOCKING + ANALYSIS LOOP
# ============================================================

def dock_and_analyze_all(
    ligand_df: pd.DataFrame,
    receptor_map: Dict[str, Dict],
    ligand_id_col: str = LIGAND_ID_COL,
    pdbqt_col: str = LIGAND_PDBQT_COL,
    smiles_col: str = LIGAND_SMILES_COL,
    jak1_ic50_col: str = LIGAND_JAK1_IC50_NM_COL,
    jak2_ic50_col: str = LIGAND_JAK2_IC50_NM_COL,
    workers: int = 1,
    cpu_per_job: int = 1,
    checkpoint_path: Optional[Path] = None,
    resume: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    workers: number of (ligand, isoform) jobs to run concurrently, each in
        its own process. workers=1 reproduces the original fully-sequential
        behavior. Set this close to os.cpu_count() for maximum throughput,
        leaving a core or two free for the OS/orchestration.
    cpu_per_job: threads Vina itself uses per docking call (its --cpu flag).
        Keep this small (1-2) when workers > 1 -- workers * cpu_per_job
        competing for the same physical cores causes oversubscription and
        can make things SLOWER than fewer, cheaper jobs run more of at once.
    """
    COMPLEX_OUT_DIR.mkdir(parents=True, exist_ok=True)

    has_smiles = smiles_col in ligand_df.columns
    has_jak1_ic50 = jak1_ic50_col in ligand_df.columns
    has_jak2_ic50 = jak2_ic50_col in ligand_df.columns
    if not has_smiles:
        print(f"[!] Ligand CSV has no '{smiles_col}' column -- SMILES will be blank in the output.")
    if not (has_jak1_ic50 and has_jak2_ic50):
        missing = [c for c, present in [(jak1_ic50_col, has_jak1_ic50), (jak2_ic50_col, has_jak2_ic50)] if not present]
        print(f"[!] Ligand CSV missing {missing} -- delta_pIC50 will be blank in the output.")

    jobs: List[DockJob] = []
    # Per-isoform accumulated results. When --resume is enabled, successful
    # jobs from the checkpoint are loaded here and omitted from the new job list.
    raw_results: Dict[str, List[Dict]] = {iso: [] for iso in receptor_map}
    completed_keys = set()

    if resume and checkpoint_path is not None:
        checkpoint_records, completed_keys = load_checkpoint(checkpoint_path)
        for record in checkpoint_records:
            iso_name = record.get("iso_name")
            if iso_name in raw_results:
                raw_results[iso_name].append(record)
        print(f"Resuming: loaded {len(completed_keys)} completed docking jobs from {checkpoint_path}")

    for _, row in ligand_df.iterrows():
        lig_id = str(row[ligand_id_col])
        lig_pdbqt_str = row[pdbqt_col]
        lig_smiles = row[smiles_col] if has_smiles else None
        lig_delta_pic50 = (
            compute_delta_pic50(row[jak1_ic50_col], row[jak2_ic50_col])
            if (has_jak1_ic50 and has_jak2_ic50)
            else None
        )

        if not isinstance(lig_pdbqt_str, str) or not lig_pdbqt_str.strip():
            print(f"[!] Skipping {lig_id}: no PDBQT available.")
            for iso_name in receptor_map:
                raw_results[iso_name].append({
                    "ligand_id": lig_id,
                    "smiles": lig_smiles,
                    "delta_pic50": lig_delta_pic50,
                    "vina_score": None,
                    "residue_interactions": {},
                })
            continue

        for iso_name, rec in receptor_map.items():
            if _checkpoint_key(lig_id, iso_name) in completed_keys:
                continue
            jobs.append(DockJob(
                ligand_id=lig_id,
                ligand_pdbqt=lig_pdbqt_str,
                iso_name=iso_name,
                receptor=rec,
                exhaustiveness=rec.get("exhaustiveness", DEFAULT_EXHAUSTIVENESS),
                cpu_per_job=cpu_per_job,
                smiles=lig_smiles,
                delta_pic50=lig_delta_pic50,
            ))

    print(
        f"Running {len(jobs)} docking jobs "
        f"({len(ligand_df)} ligands x {len(receptor_map)} isoforms) "
        f"with {workers} parallel worker process(es), {cpu_per_job} vina cpu(s) each..."
    )

    if workers <= 1:
        # Sequential path -- also used when the caller explicitly wants the
        # original one-job-at-a-time behavior (e.g. for debugging).
        with tqdm.tqdm(total=len(jobs), desc="DOCKING") as pbar:
            for job in jobs:
                result = _run_one_job(job)
                if result["error"]:
                    print(f"\nERROR docking {result['ligand_id']} x {result['iso_name']}: {result['error']}")
                else:
                    if checkpoint_path is not None:
                        append_checkpoint(checkpoint_path, result)
                    completed_keys.add(_checkpoint_key(result["ligand_id"], result["iso_name"]))
                raw_results[result["iso_name"]].append(result)
                pbar.update(1)
    else:
        # multiprocessing (not threading): PLIP keeps some global config
        # state and relies on OpenBabel, which is safer isolated in separate
        # processes than shared across threads in one interpreter.
        executor = cf.ProcessPoolExecutor(max_workers=workers)
        futures = [executor.submit(_run_one_job, job) for job in jobs]
        try:
            with tqdm.tqdm(total=len(jobs), desc="DOCKING") as pbar:
                for future in cf.as_completed(futures):
                    result = future.result()
                    if result["error"]:
                        print(f"\nERROR docking {result['ligand_id']} x {result['iso_name']}: {result['error']}")
                    else:
                        if checkpoint_path is not None:
                            append_checkpoint(checkpoint_path, result)
                        completed_keys.add(_checkpoint_key(result["ligand_id"], result["iso_name"]))
                    raw_results[result["iso_name"]].append(result)
                    pbar.update(1)
        except KeyboardInterrupt:
            print("\nStopping requested. Completed jobs have already been checkpointed; rerun with --resume to continue.")
            # Cancel work that has not started yet. Jobs already running may
            # finish, but their results will be checkpointed only if returned.
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    wide_results: Dict[str, pd.DataFrame] = {}
    for iso_name, records in raw_results.items():
        # Union of every residue seen across all ligands for this isoform.
        all_residues = sorted({
            res
            for rec in records
            for res in rec["residue_interactions"].keys()
        })

        rows = []
        for rec in records:
            row = {
                "ligand_id": rec["ligand_id"],
                "SMILES": rec.get("smiles"),
                "delta_pIC50": rec.get("delta_pic50"),
                "vina_score": rec["vina_score"],
            }
            res_interactions = rec["residue_interactions"]
            for res in all_residues:
                types = res_interactions.get(res)
                # Exact interaction type(s) instead of a binary 0/1 flag.
                # Blank when this residue did not interact with this ligand.
                row[res] = INTERACTION_TYPE_SEP.join(sorted(types)) if types else ""
            rows.append(row)

        df = pd.DataFrame(rows)
        ordered_cols = ["ligand_id", "SMILES", "delta_pIC50", "vina_score"] + all_residues
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
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of (ligand, isoform) docking jobs to run concurrently. "
             "1 = original sequential behavior. Try os.cpu_count() - 1 for max throughput.",
    )
    parser.add_argument(
        "--cpu-per-job", type=int, default=1,
        help="Threads Vina uses per docking call (its --cpu flag). Keep small "
             "when --workers > 1 to avoid oversubscribing cores.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the persistent JSONL checkpoint in --outdir. Successful jobs are skipped.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "docking_checkpoint.jsonl"

    # A run without --resume is a fresh run: do not accidentally reuse a
    # checkpoint from an older run that may have used different inputs/settings.
    if not args.resume and checkpoint_path.exists():
        checkpoint_path.unlink()

    with open(args.receptors, "r") as f:
        receptor_map = json.load(f)

    ligand_df = pd.read_csv(args.ligands)
    wide_results = dock_and_analyze_all(
        ligand_df,
        receptor_map,
        workers=args.workers,
        cpu_per_job=args.cpu_per_job,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )

    for iso_name, df in wide_results.items():
        out_path = outdir / f"{iso_name}_docking_results.csv"
        df.to_csv(out_path, index=False)
        # -4 for ligand_id, SMILES, delta_pIC50, vina_score
        print(f"Wrote {out_path} ({df.shape[0]} ligands x {df.shape[1] - 4} residue columns)")

if __name__ == "__main__":
    main()