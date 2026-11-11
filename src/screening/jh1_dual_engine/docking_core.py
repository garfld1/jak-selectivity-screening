#!/usr/bin/env python3
"""
Shared docking primitives for the JH1 dual-engine pipeline.

Everything here is engine-level plumbing with no notion of a screening run:
ligand preparation, one Vina call, one rDock call, pose extraction and PLIP
annotation. ``dual_engine_docking.py`` composes these into jobs.

Pose features are strings of the form ``RES123:hydrogen_bond``, which
``plip_tables.py`` splits back into a residue part and an interaction-type
part.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from meeko import MoleculePreparation, PDBQTWriterLegacy
from plip.structure.preparation import PDBComplex
from rdkit import Chem
from rdkit.Chem import AllChem

LIGAND_RESNAME = "LIG"
LIGAND_CHAIN_ID = "Z"
LIGAND_RESSEQ = 1

# Receptor residues PLIP should not see: modified residues it mis-types and
# waters that would create spurious bridges.
EXCLUDE_RESNAMES = {"PTR", "SEP", "TPO", "HOH", "MSE"}

# AutoDock atom type -> element. NA is a nitrogen AutoDock type, not sodium.
AD_TYPE_TO_ELEMENT = {
    "A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N", "O": "O", "OA": "O",
    "OS": "O", "S": "S", "SA": "S", "H": "H", "HD": "H", "HS": "H", "F": "F",
    "CL": "Cl", "BR": "Br", "I": "I", "MG": "Mg", "CA": "Ca", "MN": "Mn",
    "FE": "Fe", "ZN": "Zn", "P": "P", "SI": "Si", "B": "B",
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
# LIGAND PREPARATION
# ============================================================

def generate_3d_conformer(smiles: str) -> Chem.Mol:
    """SMILES -> explicit-H RDKit molecule with one minimized 3D conformer."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    if AllChem.EmbedMolecule(mol, randomSeed=0xF00D) != 0:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            raise RuntimeError("Failed to generate 3D conformer")

    try:
        status = AllChem.MMFFOptimizeMolecule(mol)
        if status not in (0, 1):
            AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception as exc:
            raise RuntimeError(f"Geometry optimization failed: {exc}") from exc

    return mol


def rdkit_mol_to_pdbqt_text(mol: Chem.Mol) -> str:
    """Prepared 3D molecule -> PDBQT via Meeko, tolerating both writer APIs."""
    preparator = MoleculePreparation()
    setups = preparator.prepare(mol) if hasattr(preparator, "prepare") else preparator(mol)
    if not setups:
        raise RuntimeError("Meeko produced no molecule setup for this ligand.")

    result = PDBQTWriterLegacy.write_string(setups[0])
    if isinstance(result, tuple):
        if len(result) == 3:
            pdbqt_string, is_ok, err_msg = result
            if not is_ok:
                raise RuntimeError(f"Meeko failed to write PDBQT: {err_msg}")
            return pdbqt_string
        if len(result) == 1:
            return str(result[0])
    if not isinstance(result, str):
        raise RuntimeError(f"Unexpected Meeko PDBQT writer return type: {type(result)!r}")
    return result


# ============================================================
# VINA
# ============================================================

def dock_with_vina(
    vina_bin: str,
    ligand_pdbqt: str,
    receptor_pdbqt: str,
    out_pdbqt: str,
    log_path: str,
    center: Sequence[float],
    size: Sequence[float],
    exhaustiveness: int,
    cpu: int = 1,
    timeout: int = 1800,
    num_modes: int = 1,
    energy_range: Optional[float] = None,
    min_rmsd: Optional[float] = None,
) -> None:
    """Run one Vina docking.

    Vina 1.2.x removed --log, so stdout and stderr are captured into log_path
    instead; a non-zero exit raises with the tail of that log.
    """
    if not os.path.exists(vina_bin):
        raise RuntimeError(f"Vina binary not found at {vina_bin}")

    cx, cy, cz = center
    sx, sy, sz = size
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
        "--num_modes", str(num_modes),
    ]
    if energy_range is not None:
        cmd.extend(["--energy_range", str(energy_range)])
    if min_rmsd is not None:
        cmd.extend(["--min_rmsd", str(min_rmsd)])

    with open(log_path, "w") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, timeout=timeout, text=True)

    if proc.returncode != 0:
        tail = Path(log_path).read_text(errors="replace")[-1500:]
        raise RuntimeError(f"Vina exited with status {proc.returncode}:\n{tail}")


def split_pdbqt_models(pdbqt_text: str) -> List[str]:
    """Split Vina output into MODEL/ENDMDL blocks; single-pose files yield one block."""
    if "MODEL" not in pdbqt_text:
        return [pdbqt_text] if pdbqt_text.strip() else []

    models: List[str] = []
    current: List[str] = []
    in_model = False
    for line in pdbqt_text.splitlines(keepends=True):
        if line.startswith("MODEL"):
            in_model = True
            current = [line]
            continue
        if not in_model:
            continue
        current.append(line)
        if line.startswith("ENDMDL"):
            models.append("".join(current))
            current = []
            in_model = False
    return models


def vina_score_from_block(block: str) -> float:
    match = re.search(r"REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", block)
    if not match:
        raise RuntimeError("REMARK VINA RESULT missing from Vina pose")
    return float(match.group(1))


def _infer_element(atom_name: str, line: str) -> str:
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


def pdbqt_block_to_ligand_pdb(block: str) -> str:
    """One Vina MODEL block -> HETATM-only ligand PDB text."""
    atom_lines: List[str] = []
    serial = 1
    for line in block.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            atom_name = line[12:16]
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"Could not parse PDBQT atom line: {line.rstrip()}") from exc

        element = _infer_element(atom_name, line)
        atom_lines.append(
            f"HETATM{serial:5d} {atom_name:<4s}{' ':1s}{LIGAND_RESNAME:>3s} "
            f"{LIGAND_CHAIN_ID:1s}{LIGAND_RESSEQ:4d}{' ':1s}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}"
        )
        serial += 1

    if not atom_lines:
        raise RuntimeError("No ligand atoms found in docked pose")
    atom_lines.extend(("TER", "END"))
    return "\n".join(atom_lines) + "\n"


# ============================================================
# RDOCK
# ============================================================

def rdock_env(rdock_root: Path) -> Dict[str, str]:
    rbdock = Path(rdock_root) / "bin" / "rbdock"
    if not rbdock.exists():
        raise RuntimeError(f"rbdock not found at {rbdock}")
    env = dict(os.environ)
    env["RBT_ROOT"] = str(rdock_root)
    env["RBT_HOME"] = str(rdock_root)
    libdir = str(Path(rdock_root) / "lib")
    for var in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        prev = env.get(var, "")
        env[var] = libdir + (os.pathsep + prev if prev else "")
    env["PATH"] = str(Path(rdock_root) / "bin") + os.pathsep + env.get("PATH", "")
    return env


def run_rbdock(
    rdock_root: Path,
    prm_path: Path,
    ligand_sd: Path,
    out_stem: Path,
    n_runs: int,
    protocol: str = "dock.prm",
    timeout: int = 3600,
) -> Path:
    """Run rbdock from the prm directory so relative protocol lookups resolve."""
    proc = subprocess.run(
        [
            str(Path(rdock_root) / "bin" / "rbdock"),
            "-r", str(prm_path),
            "-p", protocol,
            "-i", str(ligand_sd),
            "-o", str(out_stem),
            "-n", str(n_runs),
        ],
        cwd=str(Path(prm_path).parent),
        capture_output=True,
        text=True,
        env=rdock_env(rdock_root),
        timeout=timeout,
    )
    out_sd = Path(f"{out_stem}.sd")
    if proc.returncode != 0 or not out_sd.exists():
        raise RuntimeError(f"rbdock exited {proc.returncode}:\n{(proc.stdout + proc.stderr)[-1500:]}")
    return out_sd


def split_sd(text: str) -> List[str]:
    """Split an SD file into records without dropping a blank molecule title line."""
    records: List[str] = []
    for i, part in enumerate(text.split("$$$$")):
        if "M  END" not in part:
            continue
        if i > 0 and part.startswith("\n"):
            part = part[1:]
        records.append(part.rstrip() + "\n")
    return records


def molblock_for_rdkit(record: str) -> str:
    """Move the V2000 counts line to row 4, which is where RDKit expects it."""
    lines = record.splitlines()
    counts_idx = next(
        (i for i, line in enumerate(lines) if "V2000" in line or "V3000" in line),
        None,
    )
    if counts_idx is None:
        return record if record.endswith("\n") else record + "\n"
    header = lines[:counts_idx]
    while len(header) < 3:
        header.insert(0, "")
    if len(header) > 3:
        header = header[-3:]
    return "\n".join(header + lines[counts_idx:]) + "\n"


def sd_score(record: str) -> float:
    lines = record.splitlines()
    for i, line in enumerate(lines[:-1]):
        if line.strip().startswith(">") and "<SCORE>" in line:
            return float(lines[i + 1].strip())
    raise RuntimeError("SCORE field missing in rDock SD record")


def heavy_xyz(record: str) -> np.ndarray:
    mol = Chem.MolFromMolBlock(molblock_for_rdkit(record), sanitize=False, removeHs=False)
    if mol is None or mol.GetNumConformers() == 0:
        raise RuntimeError("could not parse rDock SD record")
    conf = mol.GetConformer()
    points = [
        conf.GetAtomPosition(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    if not points:
        raise RuntimeError("rDock SD record has no heavy atoms")
    return np.array([(p.x, p.y, p.z) for p in points], dtype=float)


def coord_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        raise RuntimeError("pose atom counts differ; cannot compute RMSD")
    delta = a - b
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def select_diverse(records: Sequence[str], k: int, min_rmsd: float) -> List[Tuple[float, str]]:
    """Best score first; keep a pose only if it is >= min_rmsd from every kept pose."""
    scored = [(sd_score(record), heavy_xyz(record), record) for record in records]
    scored.sort(key=lambda item: item[0])

    kept: List[Tuple[float, np.ndarray, str]] = []
    for score, xyz, record in scored:
        if all(coord_rmsd(xyz, prev) >= min_rmsd for _, prev, _ in kept):
            kept.append((score, xyz, record))
        if len(kept) >= k:
            break
    return [(score, record) for score, _, record in kept]


def mol_to_ligand_pdb(mol: Chem.Mol) -> str:
    conf = mol.GetConformer()
    lines: List[str] = []
    for serial, atom in enumerate(mol.GetAtoms(), start=1):
        pos = conf.GetAtomPosition(atom.GetIdx())
        element = atom.GetSymbol()
        lines.append(
            f"HETATM{serial:5d} {element:<4s}{' ':1s}{LIGAND_RESNAME:>3s} "
            f"{LIGAND_CHAIN_ID:1s}{LIGAND_RESSEQ:4d}{' ':1s}   "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}"
        )
    if not lines:
        raise RuntimeError("rDock pose has no atoms")
    lines.extend(("TER", "END"))
    return "\n".join(lines) + "\n"


def sd_record_to_ligand_pdb(record: str) -> str:
    mol = Chem.MolFromMolBlock(molblock_for_rdkit(record), sanitize=False, removeHs=False)
    if mol is None:
        raise RuntimeError("could not parse rDock SD record")
    return mol_to_ligand_pdb(mol)


def mol_to_sd(mol: Chem.Mol) -> str:
    block = Chem.MolToMolBlock(mol)
    if not block.endswith("\n"):
        block += "\n"
    return block + "$$$$\n"


# ============================================================
# PLIP
# ============================================================

def protein_only_pdb(pdb_text: str) -> str:
    """Keep receptor ATOM/HETATM records except excluded residue names."""
    keep = [
        line
        for line in pdb_text.splitlines()
        if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() not in EXCLUDE_RESNAMES
    ]
    return "\n".join(keep) + "\n"


def write_complex_pdb(protein_pdb_text: str, ligand_pdb_text: str, out_path: str) -> None:
    with open(out_path, "w") as handle:
        handle.write(protein_pdb_text)
        if not protein_pdb_text.rstrip().endswith("TER"):
            handle.write("TER\n")
        handle.write(ligand_pdb_text)


def run_plip_all_residues(complex_pdb: str) -> Dict[str, Set[str]]:
    """Return residue label -> set of exact PLIP interaction types."""
    complex_structure = PDBComplex()
    complex_structure.load_pdb(complex_pdb)
    complex_structure.analyze()

    interacting: Dict[str, Set[str]] = {}
    for interactions in complex_structure.interaction_sets.values():
        for attr, label in PLIP_ATTRS.items():
            entries = getattr(interactions, attr, None)
            if not entries:
                continue
            for entry in entries:
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
                interacting.setdefault(res_label, set()).add(label)
    return interacting


def features_from_plip(residue_map: Dict[str, Set[str]]) -> List[str]:
    """Flatten PLIP output into sorted ``RESIDUE:interaction_type`` labels."""
    return sorted(
        {f"{residue}:{label}" for residue, labels in residue_map.items() for label in labels}
    )


def plip_features(protein_pdb: str, ligand_pdb: str, workdir: Path, tag: str) -> List[str]:
    complex_path = Path(workdir) / f"{tag}_complex.pdb"
    write_complex_pdb(protein_pdb, ligand_pdb, str(complex_path))
    try:
        return features_from_plip(run_plip_all_residues(str(complex_path)))
    finally:
        complex_path.unlink(missing_ok=True)
