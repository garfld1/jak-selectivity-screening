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

Outputs under <outdir>/<source>/:
    prepared_receptors/pdb/<target>.pdb        cleaned receptor, also PLIP input
    prepared_receptors/pdbqt/<target>.pdbqt    Vina receptor (Meeko)
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

MK_PREPARE_RECEPTOR = "mk_prepare_receptor.py"

# Residue handling during structure cleanup.
KEEP_HET_RESIDUES = {"PTR"}                       # phosphotyrosine, keep
KEEP_IONS = {"ZN", "MG", "MN", "FE", "CA", "K"}   # structural ions, keep
REMOVE_EXPLICIT = {"HOH", "PEG", "PG4", "DTT"}    # crystallization junk, drop

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
