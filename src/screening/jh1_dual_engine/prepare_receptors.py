#!/usr/bin/env python3
"""
Step 1: receptor and docking-site preparation for the JAK1/JAK2 JH1 domains.

Two structure sources are supported and produce the same output layout:

  --source crystal    the deposited kinase-domain structures, JAK1 4EI4 and
                      JAK2 4JI9, cleaned straight out of the CIF.
  --source colabfold  the LocalColabFold JH1 models, superposed onto their
                      crystal counterpart by CA atoms so both sources share one
                      receptor frame and one docking box. Construct numbering
                      is shifted onto full-length JAK positions, so PLIP labels
                      the JAK1 hinge Leu959 rather than Leu85 and the two
                      sources stay directly comparable.

For either source the docking site comes from the crystal co-crystal ligand:
the largest bound organic residue (waters, ions, cryoprotectants and modified
amino acids excluded) defines the box center. Vina gets a cubic box of
--box-size edge length; rDock gets an RbtSphereSiteMapper cavity at the same
center with RADIUS = half that edge, so the two engines search the same volume.

Crystallographic waters whose oxygen lies inside that box are kept and given
TIP3P hydrogens. They are written into the PLIP PDB, appended to the Vina
PDBQT, and included in the rDock mol2. ColabFold models have no solvent of
their own, so they inherit the crystal waters after superposition; a water
that sits within 2 A of a model heavy atom is dropped. ``--drop-waters``
rebuilds the previous dry receptors.

Outputs under <outdir>/<source>/:
    prepared_receptors/pdb/<target>.pdb        cleaned receptor plus site waters, also PLIP input
    prepared_receptors/pdbqt/<target>.pdbqt    Vina receptor (Meeko protein, then site waters)
    rdock/<target>_protein.mol2                protonated receptor for rDock
    rdock/<target>.prm                         rDock receptor parameter file
    rdock/<target>.as                          rDock cavity from `rbcavity --was`
    rdock/prep_manifest.json                   rbcavity diagnostics
    receptors.json                             consumed by dual_engine_docking.py

Run:
    python src/screening/jh1_dual_engine/prepare_receptors.py --source crystal
    python src/screening/jh1_dual_engine/prepare_receptors.py --source colabfold
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select, Superimposer

REPO_ROOT = Path(__file__).resolve().parents[3]
CRYSTAL_DIR = REPO_ROOT / "Crystal Structures"
COLABFOLD_DIR = CRYSTAL_DIR / "colabfold_jh1"

DEFAULT_OUTDIR = REPO_ROOT / "docking" / "receptors"
DEFAULT_RDOCK_ROOT = REPO_ROOT / "ensemble_pipeline_new" / "tools" / "src" / "rDock"

TARGETS = ("JAK1", "JAK2")
CRYSTALS: Dict[str, Tuple[str, Path]] = {
    "JAK1": ("4EI4", CRYSTAL_DIR / "JAK1_4EI4.cif"),
    "JAK2": ("4JI9", CRYSTAL_DIR / "JAK2_4JI9.cif"),
}

DEFAULT_BOX_SIZE = 20.0
DEFAULT_EXHAUSTIVENESS = 16
MIN_CAVITY_A3 = 100.0
WATER_CLASH_A = 2.0
WATER_NEIGHBOR_A = 4.0
TIP3P_OH_A = 0.9572
TIP3P_ANGLE_DEG = 104.52
# Meeko types a water oxygen as OA and its hydrogens as HD. Vina scores those
# types; the charges are recorded so the PDBQT matches the rest of the receptor.
WATER_O_CHARGE = -0.412
WATER_H_CHARGE = 0.206

MK_PREPARE_RECEPTOR = "mk_prepare_receptor.py"

# Residue handling during structure cleanup.
KEEP_HET_RESIDUES = {"PTR"}                       # phosphotyrosine, keep
KEEP_IONS = {"ZN", "MG", "MN", "FE", "CA", "K"}   # structural ions, keep
REMOVE_EXPLICIT = {"HOH", "PEG", "PG4", "DTT"}    # bulk solvent and crystallization junk, drop
# Site waters are selected separately and appended after this cleanup.

# Anything in these sets is not a candidate when auto-detecting the bound
# organic ligand that defines the docking box.
WATERS = {"HOH", "DOD"}
COMMON_IONS = {
    "ZN", "MG", "MN", "FE", "CA", "K", "NA", "CL", "CO", "NI", "CU", "CD",
    "BR", "IOD", "SO4", "PO4",
}
CRYO_ADDITIVES = {
    "PEG", "PG4", "PGE", "GOL", "EDO", "DMS", "TRS", "MES", "ACT", "FMT",
    "EPE", "BME", "MPD", "IPA", "MOH", "CIT", "TAM", "1PE", "P6G", "PEO",
    "DIO", "BOG", "OGA", "SIN",
}
MODIFIED_AMINO_ACIDS = {"PTR", "SEP", "TPO", "MSE", "CSO", "KCX", "MLY", "M3L"}
LIGAND_EXCLUDE_RESNAMES = WATERS | COMMON_IONS | CRYO_ADDITIVES | MODIFIED_AMINO_ACIDS

# Residues the ColabFold superposition treats as part of the protein chain.
PROTEIN_LIKE_HET = {"PTR", "SEP", "TPO", "MSE"}

PTR_PHOSPHATE_ATOMS = {"P", "O1P", "O2P", "O3P"}


# ============================================================
# STRUCTURE IO
# ============================================================

def load_structure(path: Path, name: str = "structure"):
    """Parse a CIF or PDB file into a Biopython structure."""
    if path.suffix.lower() == ".cif":
        return MMCIFParser(QUIET=True).get_structure(name, str(path))
    return PDBParser(QUIET=True).get_structure(name, str(path))


class CleanProteinSelect(Select):
    """Keep standard protein residues, drop crystallization junk, keep a
    whitelist of hetero residues and structural ions."""

    def accept_residue(self, residue):
        resname = residue.get_resname().strip().upper()
        if residue.id[0] == " ":
            return 1
        if resname in REMOVE_EXPLICIT:
            return 0
        return 1 if (resname in KEEP_HET_RESIDUES or resname in KEEP_IONS) else 0


def write_structure_pdb(structure, output_path: Path, select: Optional[Select] = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    if select is None:
        io.save(str(output_path))
    else:
        io.save(str(output_path), select)


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


# ============================================================
# DOCKING SITE AUTO-DETECTION
# ============================================================

def find_largest_organic_ligand(structure_path: Path) -> Dict:
    """Return the largest bound organic residue and its centroid.

    Waters, ions, cryoprotectants and modified amino acids are excluded, so
    what remains is the co-crystal inhibitor that defines the ATP site.
    """
    structure = load_structure(structure_path, "receptor")
    candidates: List[Tuple[int, str, str, Tuple[float, float, float]]] = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":
                    continue
                resname = residue.get_resname().strip().upper()
                if resname in LIGAND_EXCLUDE_RESNAMES:
                    continue
                coords = [atom.coord for atom in residue]
                if not coords:
                    continue
                centroid = tuple(float(v) for v in np.array(coords).mean(axis=0))
                candidates.append((len(coords), resname, chain.id, centroid))
        break  # first model only

    if not candidates:
        raise ValueError(
            f"No candidate organic ligand in {structure_path} after excluding "
            "waters, ions, additives and modified residues."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    n_atoms, resname, chain_id, centroid = candidates[0]
    return {
        "resname": resname,
        "chain": chain_id,
        "n_atoms": n_atoms,
        "centroid": centroid,
    }


def audit_hetero_residues(structure_path: Path, label: str) -> None:
    """Print every distinct hetero residue and whether cleanup keeps it, so
    nothing is discarded without the run log showing it."""
    structure = load_structure(structure_path, label)
    print(f"    --- Hetero residue audit for {label} ---")
    seen = set()
    any_hetero = False

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":
                    continue
                any_hetero = True
                resname = residue.get_resname().strip().upper()
                key = (chain.id, resname, residue.id[1])
                if key in seen:
                    continue
                seen.add(key)
                if resname in REMOVE_EXPLICIT:
                    decision = "REMOVED (explicit junk list)"
                elif resname in KEEP_HET_RESIDUES or resname in KEEP_IONS:
                    decision = "KEPT (whitelist)"
                else:
                    decision = "REMOVED (not on keep-list)"
                print(f"        {resname:6s} chain {chain.id} resseq {residue.id[1]:>5}  ->  {decision}")
        break

    if not any_hetero:
        print("        (no hetero residues found)")


# ============================================================
# COLABFOLD MODEL -> CRYSTAL FRAME
# ============================================================

def is_protein_like(residue) -> bool:
    return residue.id[0] == " " or residue.get_resname().strip().upper() in PROTEIN_LIKE_HET


def protein_residues_by_auth(structure) -> Dict[int, object]:
    mapping: Dict[int, object] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_protein_like(residue) or "CA" not in residue:
                    continue
                mapping[int(residue.id[1])] = residue
        break
    return mapping


def collect_ca_pairs(
    model_structure,
    crystal_structure,
    jh1_start_1based: int,
    overlap_start_1based: int,
    overlap_end_1based: int,
) -> Tuple[list, list, List[int]]:
    """Pair model CA atoms with crystal CA atoms over the shared span.

    ColabFold numbers the JH1 construct from 1, the crystal uses full-length
    numbering, so model position i corresponds to full-length position
    jh1_start_1based + i - 1.
    """
    model_res = protein_residues_by_auth(model_structure)
    crystal_res = protein_residues_by_auth(crystal_structure)
    fixed, moving, used = [], [], []
    for full_pos in range(overlap_start_1based, overlap_end_1based + 1):
        model_pos = full_pos - jh1_start_1based + 1
        if model_pos not in model_res or full_pos not in crystal_res:
            continue
        moving.append(model_res[model_pos]["CA"])
        fixed.append(crystal_res[full_pos]["CA"])
        used.append(full_pos)
    if len(fixed) < 20:
        raise ValueError(f"Only {len(fixed)} CA pairs for superposition; need at least 20.")
    return fixed, moving, used


def jh1_placement(target: str) -> Dict:
    """Where the JH1 construct sits in the full-length sequence."""
    for extra in (REPO_ROOT / "src" / "structure", REPO_ROOT / "src" / "training"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    import jak_colabfold_templates as jak  # noqa: PLC0415 - only the ColabFold path needs it

    return jak.assess_job(
        {
            "name": f"{target}_jh1",
            "protein": target,
            "construct": "jh1",
            "sequence": jak.SEQUENCES[target]["jh1"],
            "pdb_id": jak.TEMPLATES[target][0],
            "template_path": jak.TEMPLATES[target][1],
        },
        min_identity=0.90,
        max_mismatch_fraction=0.05,
    )


def renumber_to_full_length(structure, offset: int):
    """Shift construct numbering onto full-length JAK positions.

    ColabFold numbers the JH1 construct from 1, so PLIP would label the JAK1
    hinge Leu85 instead of Leu959. Shifting here means both structure sources
    produce the same residue labels. Residues are detached before renumbering
    so the shift cannot collide with an ID still in the chain.
    """
    if offset == 0:
        return structure
    for model in structure:
        for chain in model:
            residues = list(chain)
            for residue in residues:
                chain.detach_child(residue.id)
            for residue in residues:
                hetflag, resseq, icode = residue.id
                residue.id = (hetflag, resseq + offset, icode)
                chain.add(residue)
        break  # first model only
    return structure


def superpose_model_onto_crystal(model_path: Path, crystal_path: Path, target: str):
    """Superpose a raw ColabFold JH1 model onto its crystal counterpart."""
    info = jh1_placement(target)
    model = load_structure(model_path, "colabfold")
    crystal = load_structure(crystal_path, "crystal")
    fixed, moving, used = collect_ca_pairs(
        model,
        crystal,
        info["jh1_start_1based"],
        info["overlap_full_start_1based"],
        info["overlap_full_end_1based"],
    )
    superimposer = Superimposer()
    superimposer.set_atoms(fixed, moving)
    superimposer.apply(list(model.get_atoms()))
    return model, float(superimposer.rms), len(used)


def resolve_colabfold_model(target: str, pdb_id: str, crystal_path: Path, model_dir: Path):
    """Return (structure, metadata) for a ColabFold JH1 model in the crystal frame.

    A previously aligned model is reused as-is; otherwise the raw rank-1 model
    is superposed here.
    """
    aligned_path = model_dir / f"{target}_jh1_aligned_to_{pdb_id}.pdb"
    raw_path = model_dir / f"{target}_jh1.pdb"

    if aligned_path.is_file():
        print(f"    Using pre-aligned ColabFold model {relpath(aligned_path)}")
        return load_structure(aligned_path, "colabfold"), {
            "model_pdb": relpath(raw_path) if raw_path.is_file() else None,
            "aligned_pdb": relpath(aligned_path),
            "aligned_to": relpath(crystal_path),
            "superposition_ca_rmsd": None,
            "superposition_n_ca": None,
        }

    if not raw_path.is_file():
        raise FileNotFoundError(
            f"No ColabFold JH1 model for {target}. Expected {aligned_path} or {raw_path}. "
            "Run src/structure/colabfold_jh1_docking_prep.py first, or pass --model-dir."
        )

    structure, rms, n_ca = superpose_model_onto_crystal(raw_path, crystal_path, target)
    write_structure_pdb(structure, aligned_path)
    print(f"    Superposed {relpath(raw_path)} onto {pdb_id}: CA RMSD {rms:.3f} A over {n_ca} atoms")
    return structure, {
        "model_pdb": relpath(raw_path),
        "aligned_pdb": relpath(aligned_path),
        "aligned_to": relpath(crystal_path),
        "superposition_ca_rmsd": rms,
        "superposition_n_ca": n_ca,
    }


# ============================================================
# SITE WATERS
# ============================================================

_ACCEPTOR_SIDE = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"},
    "GLN": {"OE1"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "HIS": {"ND1", "NE2"},
}
_DONOR_SIDE = {
    "TRP": {"NE1"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "LYS": {"NZ"},
    "ARG": {"NE", "NH1", "NH2"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
}


def _first_model_residues(structure):
    for model in structure:
        for chain in model:
            for residue in chain:
                yield chain, residue
        break


def _is_hydrogen(atom) -> bool:
    element = (getattr(atom, "element", None) or "").strip().upper()
    if element:
        return element == "H"
    return atom.get_name().strip().upper().startswith("H")


def _water_oxygen(residue):
    for atom in residue:
        if _is_hydrogen(atom):
            continue
        element = (getattr(atom, "element", None) or "").strip().upper()
        if element == "O" or atom.get_name().strip().upper() == "O":
            return atom
    return None


def _pdb_atom_name(name: str) -> str:
    """Four-column PDB atom name. One-letter elements sit in column 14."""
    if len(name) > 4:
        raise ValueError(f"atom name {name!r} is longer than 4 characters")
    if len(name) == 4:
        return name
    return f" {name:<3s}"[:4]


def select_box_waters(structure_path: Path, center: Sequence[float], box_size: float) -> List[Dict]:
    """Crystal waters whose oxygen lies inside the cubic docking box.

    The returned dicts carry the oxygen coordinate and B-factor. Hydrogens are
    added later by ``orient_site_waters``.
    """
    half = float(box_size) / 2.0
    origin = np.asarray(center, dtype=float)
    structure = load_structure(structure_path, "waters")
    waters: List[Dict] = []
    seen = set()
    for chain, residue in _first_model_residues(structure):
        if residue.get_resname().strip().upper() not in WATERS:
            continue
        oxygen = _water_oxygen(residue)
        if oxygen is None:
            continue
        chain_id = (chain.id or "A").strip()[:1] or "A"
        resseq = int(residue.id[1])
        key = (chain_id, resseq)
        if key in seen:
            continue
        coord = np.array(oxygen.coord, dtype=float)
        if np.any(np.abs(coord - origin) > half + 1e-6):
            continue
        seen.add(key)
        waters.append(
            {
                "chain": chain_id,
                "resseq": resseq,
                "oxygen": coord,
                "bfactor": float(oxygen.get_bfactor()),
                "hydrogens": None,
            }
        )
    waters.sort(key=lambda water: (water["chain"], water["resseq"]))
    return waters


def protein_heavy_coordinates(structure) -> np.ndarray:
    """Heavy-atom coordinates of residues the cleaned receptor keeps."""
    selector = CleanProteinSelect()
    coords = []
    for _, residue in _first_model_residues(structure):
        if not selector.accept_residue(residue):
            continue
        for atom in residue:
            if _is_hydrogen(atom):
                continue
            coords.append(np.array(atom.coord, dtype=float))
    if not coords:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(coords)


def drop_clashing_waters(
    waters: Sequence[Dict],
    protein_coords: np.ndarray,
    cutoff: float = WATER_CLASH_A,
) -> Tuple[List[Dict], List[Dict]]:
    """Drop waters whose oxygen is within ``cutoff`` of a protein heavy atom."""
    if len(protein_coords) == 0:
        return list(waters), []
    kept, dropped = [], []
    for water in waters:
        distance = float(np.linalg.norm(protein_coords - water["oxygen"], axis=1).min())
        if distance <= cutoff:
            dropped.append(water)
        else:
            kept.append(water)
    return kept, dropped


def _polar_sites(structure) -> Tuple[np.ndarray, np.ndarray]:
    """Protein acceptor and donor heavy atoms, used to orient water hydrogens."""
    selector = CleanProteinSelect()
    acceptors, donors = [], []
    for _, residue in _first_model_residues(structure):
        if not selector.accept_residue(residue):
            continue
        resname = residue.get_resname().strip().upper()
        for atom in residue:
            if _is_hydrogen(atom):
                continue
            name = atom.get_name().strip().upper()
            coord = np.array(atom.coord, dtype=float)
            if name in {"O", "OXT"} or name in _ACCEPTOR_SIDE.get(resname, ()):
                acceptors.append(coord)
            if (name == "N" and resname != "PRO") or name in _DONOR_SIDE.get(resname, ()):
                donors.append(coord)
    acceptor_xyz = np.vstack(acceptors) if acceptors else np.zeros((0, 3), dtype=float)
    donor_xyz = np.vstack(donors) if donors else np.zeros((0, 3), dtype=float)
    return acceptor_xyz, donor_xyz


def _nearby(points: np.ndarray, origin: np.ndarray, cutoff: float) -> np.ndarray:
    if len(points) == 0:
        return points
    distance = np.linalg.norm(points - origin, axis=1)
    return points[distance <= cutoff]


def _tip3p_hydrogens(origin: np.ndarray, direction: np.ndarray, twist: float) -> Tuple[np.ndarray, np.ndarray]:
    """Two TIP3P hydrogens. ``direction`` is the O-H1 unit vector, ``twist`` spins H2."""
    unit = direction / np.linalg.norm(direction)
    helper = np.array([1.0, 0.0, 0.0]) if abs(unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perpendicular = np.cross(unit, helper)
    perpendicular /= np.linalg.norm(perpendicular)
    theta = math.radians(TIP3P_ANGLE_DEG)
    base = math.cos(theta) * unit + math.sin(theta) * perpendicular
    cosine, sine = math.cos(twist), math.sin(twist)
    spun = base * cosine + np.cross(unit, base) * sine + unit * np.dot(unit, base) * (1.0 - cosine)
    return origin + TIP3P_OH_A * unit, origin + TIP3P_OH_A * spun


def _orientation_score(
    origin: np.ndarray,
    hydrogen_a: np.ndarray,
    hydrogen_b: np.ndarray,
    acceptors: np.ndarray,
    donors: np.ndarray,
) -> float:
    score = 0.0
    if len(acceptors):
        for hydrogen in (hydrogen_a, hydrogen_b):
            distance = float(np.linalg.norm(acceptors - hydrogen, axis=1).min())
            score += math.exp(-0.5 * ((distance - 1.8) / 0.35) ** 2)
    if len(donors):
        bisector = (hydrogen_a - origin) + (hydrogen_b - origin)
        norm = float(np.linalg.norm(bisector))
        if norm > 1e-8:
            lone_pair = -bisector / norm
            vectors = donors - origin
            distance = np.linalg.norm(vectors, axis=1)
            nearest = int(distance.argmin())
            direction = vectors[nearest] / distance[nearest]
            alignment = max(0.0, float(np.dot(direction, lone_pair)))
            score += alignment * math.exp(-0.5 * ((float(distance[nearest]) - 2.8) / 0.4) ** 2)
    return score


def orient_site_waters(waters: Sequence[Dict], structure) -> None:
    """Place TIP3P hydrogens so they point at nearby protein acceptors.

    The search grid is fixed, and ties keep the first sample, so the same
    water and protein always produce the same hydrogens.
    """
    acceptors, donors = _polar_sites(structure)
    polar_steps = np.linspace(0.0, math.pi, 12, endpoint=False)
    azimuth_steps = np.linspace(0.0, 2.0 * math.pi, 18, endpoint=False)
    twist_steps = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    for water in waters:
        origin = water["oxygen"]
        local_acceptors = _nearby(acceptors, origin, WATER_NEIGHBOR_A)
        local_donors = _nearby(donors, origin, WATER_NEIGHBOR_A)
        best_score = -1.0
        best = None
        for polar in polar_steps:
            sin_polar = math.sin(polar)
            for azimuth in azimuth_steps:
                direction = np.array(
                    [sin_polar * math.cos(azimuth), sin_polar * math.sin(azimuth), math.cos(polar)],
                    dtype=float,
                )
                for twist in twist_steps:
                    hydrogens = _tip3p_hydrogens(origin, direction, float(twist))
                    score = _orientation_score(origin, hydrogens[0], hydrogens[1], local_acceptors, local_donors)
                    if best is None or score > best_score + 1e-12:
                        best_score = score
                        best = hydrogens
        water["hydrogens"] = best


def water_json(water: Dict) -> Dict:
    return {
        "chain": water["chain"],
        "resseq": int(water["resseq"]),
        "bfactor": round(float(water["bfactor"]), 2),
        "xyz": [round(float(value), 3) for value in water["oxygen"]],
    }


def _format_pdb_hetatm(serial, name, chain, resseq, xyz, bfactor, element) -> str:
    x, y, z = (float(value) for value in xyz)
    return (
        f"HETATM{serial:5d} {_pdb_atom_name(name)} HOH "
        f"{chain:1s}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{float(bfactor):6.2f}"
        f"          {element:>2s}"
    )


def _format_pdbqt_hetatm(serial, name, chain, resseq, xyz, charge, ad_type) -> str:
    x, y, z = (float(value) for value in xyz)
    return (
        f"HETATM{serial:5d} {_pdb_atom_name(name)} HOH "
        f"{chain:1s}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    {charge:+6.3f} {ad_type:<2s}"
    )


def _atom_lines(waters: Sequence[Dict], start_serial: int, pdbqt: bool) -> List[str]:
    lines: List[str] = []
    serial = start_serial
    for water in waters:
        if water.get("hydrogens") is None:
            raise RuntimeError(f"water {water['chain']}{water['resseq']} has no hydrogens")
        chain = water["chain"]
        resseq = int(water["resseq"])
        hydrogen_a, hydrogen_b = water["hydrogens"]
        atoms = (("O", water["oxygen"]), ("H1", hydrogen_a), ("H2", hydrogen_b))
        for name, xyz in atoms:
            if pdbqt:
                charge = WATER_O_CHARGE if name == "O" else WATER_H_CHARGE
                ad_type = "OA" if name == "O" else "HD"
                lines.append(_format_pdbqt_hetatm(serial, name, chain, resseq, xyz, charge, ad_type))
            else:
                element = "O" if name == "O" else "H"
                lines.append(
                    _format_pdb_hetatm(serial, name, chain, resseq, xyz, water["bfactor"], element)
                )
            serial += 1
    return lines


def last_atom_serial(path: Path) -> int:
    last = 0
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            last = max(last, int(line[6:11]))
        except ValueError:
            continue
    return last


def insert_before_end(path: Path, lines: Sequence[str]) -> None:
    if not lines:
        return
    rows = path.read_text().splitlines()
    end_at = next((i for i, row in enumerate(rows) if row.startswith("END")), len(rows))
    rows[end_at:end_at] = list(lines)
    path.write_text("\n".join(rows) + "\n")


def append_site_waters(path: Path, waters: Sequence[Dict], pdbqt: bool) -> None:
    insert_before_end(path, _atom_lines(waters, last_atom_serial(path) + 1, pdbqt))


def total_cavity_volume(stdout: str) -> float:
    for line in stdout.splitlines():
        if "Total volume" not in line:
            continue
        token = line.split("Total volume", 1)[1].strip().split()[0]
        return float(token)
    raise RuntimeError(f"rbcavity did not report a cavity volume:\n{stdout[-1500:]}")


# ============================================================
# VINA RECEPTOR (MEEKO)
# ============================================================

def ensure_meeko_on_path() -> None:
    if shutil.which(MK_PREPARE_RECEPTOR):
        return
    candidates = [
        Path.home() / "miniforge3" / "envs" / "jak_docking" / "bin",
        Path.home() / "miniforge3" / "envs" / "jak_ensemble" / "bin",
        Path.home() / "miniforge3" / "bin",
    ]
    for directory in candidates:
        if (directory / MK_PREPARE_RECEPTOR).is_file():
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            return
    raise RuntimeError(
        f"{MK_PREPARE_RECEPTOR} was not found. Activate the docking environment "
        "or add Meeko to PATH."
    )


def prepare_receptor_pdbqt_with_meeko(cleaned_pdb: Path, output_pdbqt: Path) -> None:
    """Cleaned receptor PDB -> PDBQT via Meeko's mk_prepare_receptor.py.

    Meeko names its output from -o, so prepare into the matching basename and
    rename if the tool picked a different suffix.
    """
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    basename = str(output_pdbqt.with_suffix(""))
    result = subprocess.run(
        [MK_PREPARE_RECEPTOR, "--read_pdb", str(cleaned_pdb), "-o", basename, "-p"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{MK_PREPARE_RECEPTOR} failed:\n{result.stderr}")

    written = Path(f"{basename}.pdbqt")
    if not written.exists():
        raise RuntimeError(
            f"{MK_PREPARE_RECEPTOR} did not produce {written}.\n{result.stdout}\n{result.stderr}"
        )
    if written != output_pdbqt:
        written.rename(output_pdbqt)


def validate_receptor_pdbqt(pdbqt_path: Path) -> Tuple[bool, str]:
    if not pdbqt_path.exists():
        return False, "File does not exist."
    n_atoms = sum(
        1 for line in pdbqt_path.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM"))
    )
    if n_atoms == 0:
        return False, "File contains no ATOM/HETATM records."
    return True, f"Atoms: {n_atoms}"


# ============================================================
# RDOCK RECEPTOR AND CAVITY
# ============================================================

def convert_ptr(pdb_text: str, mode: str = "tyr") -> str:
    """rDock has no phosphotyrosine type; strip the phosphate and call it TYR."""
    if mode == "keep":
        return pdb_text
    if mode != "tyr":
        raise ValueError(f"unknown ptr mode {mode!r}")
    out: List[str] = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[17:20] == "PTR":
            if line[12:16].strip() in PTR_PHOSPHATE_ATOMS:
                continue
            line = "ATOM  " + line[6:17] + "TYR" + line[20:]
        out.append(line)
    return "\n".join(out) + "\n"


def protein_pdb_to_mol2(pdb_text: str, ph: float = 7.4) -> str:
    """PDB -> rDock-readable mol2 with unique H names and a SUBSTRUCTURE block."""
    from openbabel import pybel  # noqa: PLC0415 - only the rDock path needs Open Babel

    mol = pybel.readstring("pdb", pdb_text)
    mol.OBMol.AddHydrogens(False, True, ph)
    mol2 = mol.write("mol2")

    pre, rest = mol2.split("@<TRIPOS>ATOM", 1)
    atoms_txt, post = rest.split("@<TRIPOS>BOND", 1)
    out_atoms: List[str] = []
    first: Dict[Tuple[int, str], int] = {}
    for line in atoms_txt.splitlines():
        fields = line.split()
        if len(fields) >= 9 and fields[0].isdigit():
            if fields[5] == "H" and fields[1] == "H":
                aid = int(fields[0])
                line = (
                    f"{aid:7d} "
                    f"{('H' + fields[0]):<8s}"
                    f"{float(fields[2]):10.4f}"
                    f"{float(fields[3]):10.4f}"
                    f"{float(fields[4]):10.4f} "
                    f"{fields[5]:<5s}"
                    f"{int(fields[6]):4d}"
                    f"  "
                    f"{fields[7]:<8s}"
                    f"{float(fields[8]):10.4f}"
                )
                fields = line.split()
            first.setdefault((int(fields[6]), fields[7]), int(fields[0]))
        out_atoms.append(line)

    bonds_txt = post.split("@<TRIPOS>UNITY_ATOM_ATTR")[0].rstrip() + "\n"
    substructure = "@<TRIPOS>SUBSTRUCTURE\n" + "".join(
        f"{i:7d} {n:<8s}{a:7d} RESIDUE           1 A     {n[:3]:<4s}     0 ****  ****    0 ROOT\n"
        for (i, n), a in sorted(first.items())
    )
    return pre + "@<TRIPOS>ATOM" + "\n".join(out_atoms) + "\n@<TRIPOS>BOND" + bonds_txt + substructure


def receptor_prm_text(
    target: str,
    mol2_path: str,
    center: Sequence[float],
    box_size: float,
    small_sphere: float = 1.0,
    large_sphere: float = 4.0,
    min_volume: float = 100.0,
) -> str:
    """Sphere mapper whose mapping cube side (2 * RADIUS) equals the Vina box edge."""
    cx, cy, cz = (float(v) for v in center)
    return "\n".join(
        [
            "RBT_PARAMETER_FILE_V1.00",
            f"TITLE {target} Vina-matched sphere cavity",
            f"RECEPTOR_FILE {mol2_path}",
            "SECTION MAPPER",
            "    SITE_MAPPER RbtSphereSiteMapper",
            f"    CENTER {cx},{cy},{cz}",
            f"    RADIUS {box_size / 2.0:g}",
            f"    SMALL_SPHERE {small_sphere:g}",
            f"    LARGE_SPHERE {large_sphere:g}",
            f"    MIN_VOLUME {min_volume:g}",
            "    MAX_CAVITIES 1",
            "    VOL_INCR 0.0",
            "    GRIDSTEP 0.5",
            "END_SECTION",
            "SECTION CAVITY",
            "    SCORING_FUNCTION RbtCavityGridSF",
            "    WEIGHT 1.0",
            "END_SECTION",
            "",
        ]
    )


def rdock_env(rdock_root: Path) -> Dict[str, str]:
    if not (rdock_root / "bin" / "rbcavity").exists():
        raise RuntimeError(f"rbcavity not found under {rdock_root}")
    env = dict(os.environ)
    env["RBT_ROOT"] = str(rdock_root)
    env["RBT_HOME"] = str(rdock_root)
    libdir = str(rdock_root / "lib")
    for var in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        prev = env.get(var, "")
        env[var] = libdir + (os.pathsep + prev if prev else "")
    env["PATH"] = str(rdock_root / "bin") + os.pathsep + env.get("PATH", "")
    return env


def prepare_rdock_site(
    target: str,
    cleaned_pdb: Path,
    center: Sequence[float],
    box_size: float,
    rdock_dir: Path,
    rdock_root: Path,
    ptr_mode: str = "tyr",
    ph: float = 7.4,
) -> Dict:
    rdock_dir.mkdir(parents=True, exist_ok=True)
    mol2_path = rdock_dir / f"{target}_protein.mol2"
    prm_path = rdock_dir / f"{target}.prm"
    as_path = rdock_dir / f"{target}.as"

    mol2_path.write_text(protein_pdb_to_mol2(convert_ptr(cleaned_pdb.read_text(), ptr_mode), ph))
    prm_path.write_text(
        receptor_prm_text(target, str(mol2_path.resolve()), center, box_size)
    )

    proc = subprocess.run(
        [str(rdock_root / "bin" / "rbcavity"), "-r", str(prm_path), "--was"],
        cwd=str(prm_path.parent),
        capture_output=True,
        text=True,
        env=rdock_env(rdock_root),
        timeout=600,
    )
    if proc.returncode != 0 or not as_path.exists():
        raise RuntimeError(f"rbcavity failed for {target}:\n{(proc.stdout + proc.stderr)[-1500:]}")

    volume = total_cavity_volume(proc.stdout)
    if volume < MIN_CAVITY_A3:
        raise RuntimeError(
            f"rbcavity volume for {target} is {volume:.1f} A^3, below {MIN_CAVITY_A3:.0f} A^3. "
            "Site waters may have filled the cavity."
        )

    volume_lines = [
        line for line in proc.stdout.splitlines()
        if "Cavity #" in line or "Total volume" in line
    ]
    n_warnings = sum(1 for line in proc.stdout.splitlines() if "WARNING" in line)
    print(f"    [{target}] rbcavity ok; warnings={n_warnings}")
    for line in volume_lines:
        print(f"        {line}")

    return {
        "target": target,
        "mol2": relpath(mol2_path),
        "prm": relpath(prm_path),
        "as": relpath(as_path),
        "ptr_mode": ptr_mode,
        "center": [float(v) for v in center],
        "radius": box_size / 2.0,
        "mapper": "RbtSphereSiteMapper",
        "rbcavity_volume_lines": volume_lines,
        "n_rbcavity_warnings": n_warnings,
    }


# ============================================================
# MAIN
# ============================================================

def prepare_target(
    target: str,
    args: argparse.Namespace,
    source_dir: Path,
) -> Tuple[Dict, Optional[Dict]]:
    pdb_id, crystal_path = CRYSTALS[target]
    if not crystal_path.is_file():
        raise FileNotFoundError(f"{target} crystal structure not found: {crystal_path}")

    print(f"\n=== {target} ({args.source}, box from {pdb_id}) ===")

    site = find_largest_organic_ligand(crystal_path)
    print(
        f"    Docking-site ligand: {site['resname']} (chain {site['chain']}, "
        f"{site['n_atoms']} atoms)"
    )
    print(f"    Box center: {site['centroid']}  edge: {args.box_size} A")
    audit_hetero_residues(crystal_path, f"{target} {pdb_id}")

    provenance: Dict = {
        "source": args.source,
        "crystal": relpath(crystal_path),
        "residue_numbering": "full_length",
        "residue_offset_applied": 0,
    }
    if args.source == "crystal":
        structure = load_structure(crystal_path, target)
    else:
        structure, model_meta = resolve_colabfold_model(
            target, pdb_id, crystal_path, Path(args.model_dir).expanduser()
        )
        provenance.update(model_meta)
        if args.keep_construct_numbering:
            provenance["residue_numbering"] = "jh1_construct"
        else:
            offset = jh1_placement(target)["jh1_start_1based"] - 1
            renumber_to_full_length(structure, offset)
            provenance["residue_offset_applied"] = offset
            print(f"    Renumbered JH1 construct to full-length numbering (+{offset})")

    pdb_dir = source_dir / "prepared_receptors" / "pdb"
    pdbqt_dir = source_dir / "prepared_receptors" / "pdbqt"
    cleaned_pdb = pdb_dir / f"{target}.pdb"
    pdbqt_path = pdbqt_dir / f"{target}.pdbqt"

    write_structure_pdb(structure, cleaned_pdb, CleanProteinSelect())
    print(f"[>] Converting {target} cleaned PDB -> PDBQT (Meeko) ...")
    prepare_receptor_pdbqt_with_meeko(cleaned_pdb, pdbqt_path)

    if args.drop_waters:
        waters, dropped = [], []
        water_source = "none"
    else:
        waters = select_box_waters(crystal_path, site["centroid"], args.box_size)
        dropped = []
        if args.source == "colabfold":
            waters, dropped = drop_clashing_waters(waters, protein_heavy_coordinates(structure))
        orient_site_waters(waters, structure)
        append_site_waters(cleaned_pdb, waters, pdbqt=False)
        append_site_waters(pdbqt_path, waters, pdbqt=True)
        water_source = "crystal_box"
        print(f"    Site waters kept: {len(waters)}")
        if dropped:
            dropped_ids = ", ".join(f"{water['chain']}{water['resseq']}" for water in dropped)
            print(f"    Dropped {len(dropped)} waters within {WATER_CLASH_A:.1f} A of the model: {dropped_ids}")

    ok, message = validate_receptor_pdbqt(pdbqt_path)
    print(f"[{'OK' if ok else 'FAIL'}] {target}: {message}")
    if not ok:
        raise RuntimeError(f"{target} PDBQT validation failed: {message}")

    record = {
        "pdb": relpath(cleaned_pdb),
        "pdbqt": relpath(pdbqt_path),
        "center": list(site["centroid"]),
        "size": [args.box_size] * 3,
        "exhaustiveness": args.exhaustiveness,
        "detected_ligand_resname": site["resname"],
        "detected_ligand_chain": site["chain"],
        "detected_ligand_natoms": site["n_atoms"],
        "water_source": water_source,
        "n_waters": len(waters),
        "waters": [water_json(water) for water in waters],
        "waters_dropped_clash": [water_json(water) for water in dropped],
        **provenance,
    }

    rdock_info = None
    if not args.skip_rdock:
        rdock_info = prepare_rdock_site(
            target,
            cleaned_pdb,
            site["centroid"],
            args.box_size,
            source_dir / "rdock",
            Path(args.rdock_root).expanduser(),
            ptr_mode=args.ptr_mode,
            ph=args.ph,
        )
        record.update(
            {
                "rdock_prm": rdock_info["prm"],
                "rdock_as": rdock_info["as"],
                "rdock_mol2": rdock_info["mol2"],
            }
        )

    return record, rdock_info


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        choices=("crystal", "colabfold"),
        default="crystal",
        help="Crystal structures (4EI4/4JI9) or the aligned ColabFold JH1 models.",
    )
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help=f"Parent directory; a <source> subdirectory is created. Default: {relpath(DEFAULT_OUTDIR)}",
    )
    parser.add_argument(
        "--model-dir",
        default=str(COLABFOLD_DIR),
        help=f"Where ColabFold JH1 models live. Default: {relpath(COLABFOLD_DIR)}",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=DEFAULT_BOX_SIZE,
        help=f"Cubic Vina box edge in A; rDock RADIUS is half this. Default: {DEFAULT_BOX_SIZE}",
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=DEFAULT_EXHAUSTIVENESS,
        help=f"Default Vina exhaustiveness recorded in receptors.json. Default: {DEFAULT_EXHAUSTIVENESS}",
    )
    parser.add_argument("--rdock-root", default=str(DEFAULT_RDOCK_ROOT), help="rDock installation prefix.")
    parser.add_argument(
        "--ptr-mode",
        choices=("tyr", "keep"),
        default="tyr",
        help="Phosphotyrosine handling for the rDock mol2. Default: tyr",
    )
    parser.add_argument("--ph", type=float, default=7.4, help="pH used when Open Babel adds hydrogens.")
    parser.add_argument(
        "--keep-construct-numbering",
        action="store_true",
        help="Leave ColabFold models numbered from 1 instead of shifting onto full-length "
        "JAK positions. PLIP labels then will not match the crystal receptors.",
    )
    parser.add_argument("--skip-rdock", action="store_true", help="Prepare Vina receptors only.")
    parser.add_argument(
        "--drop-waters",
        action="store_true",
        help="Leave the receptor dry. By default, crystallographic waters inside the docking box are kept.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.box_size <= 0:
        raise ValueError("--box-size must be positive")
    if args.exhaustiveness < 1:
        raise ValueError("--exhaustiveness must be >= 1")

    source_dir = Path(args.outdir).expanduser().resolve() / args.source
    source_dir.mkdir(parents=True, exist_ok=True)
    ensure_meeko_on_path()

    receptors: Dict[str, Dict] = {}
    rdock_manifest: List[Dict] = []
    for target in args.targets:
        record, rdock_info = prepare_target(target, args, source_dir)
        receptors[target] = record
        if rdock_info is not None:
            rdock_manifest.append(rdock_info)

    receptors_json = source_dir / "receptors.json"
    receptors_json.write_text(json.dumps(receptors, indent=2) + "\n")
    print(f"\nWrote {receptors_json}")

    if rdock_manifest:
        manifest_path = source_dir / "rdock" / "prep_manifest.json"
        manifest_path.write_text(json.dumps(rdock_manifest, indent=2) + "\n")
        print(f"Wrote {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
