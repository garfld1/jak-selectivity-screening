#!/usr/bin/env python3
"""
Fold JAK1/JAK2 JH1 with LocalColabFold, then prepare Vina and rDock receptors.

ColabFold models are not in the crystal frame and do not contain the bound
inhibitor. This script:

  1) Runs jak_colabfold_templates.py for the JH1 constructs.
  2) Copies the rank-1 models into Crystal Structures/colabfold_jh1/.
  3) Superposes each JH1 model onto 4EI4 / 4JI9 by CA atoms.
  4) Takes the docking-box center from the crystal ligand (protein_prep.py).
  5) Writes Vina PDBQT/PDB files and a new receptors.json under
     docking/docking_prep_colabfold/.
  6) Builds rDock mol2/prm/as files with the same sphere-mapper logic.

Usage
-----
python src/structure/colabfold_jh1_docking_prep.py
python src/structure/colabfold_jh1_docking_prep.py --skip-fold
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Superimposer

REPO_ROOT = Path(__file__).resolve().parents[2]
STRUCTURE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = REPO_ROOT / "src" / "training"
ENSEMBLE_SRC = REPO_ROOT / "ensemble_pipeline_new" / "src"

for extra in (STRUCTURE_DIR, TRAINING_DIR, ENSEMBLE_SRC):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import jak_colabfold_templates as jak
import protein_prep
from jak_ens.prep import rdock_prep

CRYSTAL_DIR = REPO_ROOT / "Crystal Structures"
STRUCTURE_OUT_DIR = CRYSTAL_DIR / "colabfold_jh1"
DOCKING_DIR = REPO_ROOT / "docking" / "docking_prep_colabfold"
DEFAULT_FOLD_ROOT = REPO_ROOT / "results" / "colabfold_jak"

PROTEIN_LIKE_HET = {"PTR", "SEP", "TPO", "MSE"}

CRYSTALS = {
    "JAK1": CRYSTAL_DIR / "JAK1_4EI4.cif",
    "JAK2": CRYSTAL_DIR / "JAK2_4JI9.cif",
}


def ensure_meeko_on_path() -> None:
    if shutil.which(protein_prep.MK_PREPARE_RECEPTOR):
        return
    candidates = [
        Path.home() / "miniforge3" / "envs" / "jak_docking" / "bin",
        Path.home() / "miniforge3" / "envs" / "jak_ensemble" / "bin",
        Path.home() / "miniforge3" / "bin",
    ]
    for directory in candidates:
        if (directory / protein_prep.MK_PREPARE_RECEPTOR).is_file():
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            return
    raise RuntimeError(
        f"{protein_prep.MK_PREPARE_RECEPTOR} was not found. "
        "Activate the jak_docking environment or add Meeko to PATH."
    )


def load_structure(path: Path, name: str = "structure"):
    suffix = path.suffix.lower()
    if suffix == ".cif":
        return MMCIFParser(QUIET=True).get_structure(name, str(path))
    return PDBParser(QUIET=True).get_structure(name, str(path))


def is_protein_like(residue) -> bool:
    resname = residue.get_resname().strip().upper()
    hetflag = residue.id[0]
    return hetflag == " " or resname in PROTEIN_LIKE_HET


def protein_residues_by_auth(structure) -> dict[int, object]:
    mapping: dict[int, object] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_protein_like(residue):
                    continue
                if "CA" not in residue:
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
) -> tuple[list, list, list[int]]:
    model_res = protein_residues_by_auth(model_structure)
    crystal_res = protein_residues_by_auth(crystal_structure)
    fixed = []
    moving = []
    used = []
    for full_pos in range(overlap_start_1based, overlap_end_1based + 1):
        model_pos = full_pos - jh1_start_1based + 1
        if model_pos not in model_res or full_pos not in crystal_res:
            continue
        moving.append(model_res[model_pos]["CA"])
        fixed.append(crystal_res[full_pos]["CA"])
        used.append(full_pos)
    if len(fixed) < 20:
        raise ValueError(
            f"Only {len(fixed)} CA pairs for superposition; need at least 20."
        )
    return fixed, moving, used


def superpose_model_onto_crystal(
    model_path: Path,
    crystal_path: Path,
    jh1_start_1based: int,
    overlap_start_1based: int,
    overlap_end_1based: int,
) -> tuple[object, float, int]:
    model = load_structure(model_path, "colabfold")
    crystal = load_structure(crystal_path, "crystal")
    fixed, moving, used = collect_ca_pairs(
        model,
        crystal,
        jh1_start_1based,
        overlap_start_1based,
        overlap_end_1based,
    )
    superimposer = Superimposer()
    superimposer.set_atoms(fixed, moving)
    superimposer.apply(list(model.get_atoms()))
    return model, float(superimposer.rms), len(used)


def write_structure_pdb(structure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))


def write_cleaned_pdb(structure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path), protein_prep.CleanProteinSelect())


def select_ranked_model(output_dir: Path) -> Path:
    pdbs = list(output_dir.rglob("*.pdb"))
    ranked = [path for path in pdbs if "rank" in path.name.lower()]
    if not ranked:
        raise FileNotFoundError(f"No ranked PDB models in {output_dir}")

    def sort_key(path: Path):
        match = re.search(r"rank_?0*(\d+)", path.name, re.I)
        rank = int(match.group(1)) if match else 999
        relaxed = 0 if "relax" in path.name.lower() else 1
        return (rank, relaxed, path.name)

    return sorted(ranked, key=sort_key)[0]


def extract_auth_span_renumbered(
    crystal_path: Path,
    start_1based: int,
    end_1based: int,
    output_path: Path,
) -> Path:
    """Write crystal residues start..end as a ColabFold-like PDB numbered 1..N."""
    crystal = load_structure(crystal_path, "crystal")
    residues = []
    for model in crystal:
        for chain in model:
            for residue in list(chain):
                if not is_protein_like(residue):
                    chain.detach_child(residue.id)
                    continue
                auth = int(residue.id[1])
                if start_1based <= auth <= end_1based:
                    residues.append((chain, residue))
                else:
                    chain.detach_child(residue.id)
        break

    # Keep JH1-relative numbering, including gaps for missing crystal residues,
    # so residue i is full-length position start_1based + i - 1.
    for chain, residue in residues:
        auth = int(residue.id[1])
        chain.detach_child(residue.id)
        residue.id = (residue.id[0], 10_000 + (auth - start_1based + 1), " ")
        chain.add(residue)
    for chain, residue in residues:
        rel = int(residue.id[1]) - 10_000
        chain.detach_child(residue.id)
        residue.id = (" ", rel, " ")
        chain.add(residue)

    write_structure_pdb(crystal, output_path)
    return output_path


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def run_colabfold_jh1(
    output_root: Path,
    overwrite: bool,
    num_recycle: int,
    num_models: int,
) -> int:
    argv = [
        "--protein",
        "JAK1",
        "JAK2",
        "--construct",
        "jh1",
        "--output-root",
        str(output_root),
        "--num-recycle",
        str(num_recycle),
        "--num-models",
        str(num_models),
    ]
    if overwrite:
        argv.append("--overwrite")
    return jak.main(argv)


def publish_ranked_models(fold_root: Path, structure_dir: Path) -> dict[str, Path]:
    structure_dir.mkdir(parents=True, exist_ok=True)
    published = {}
    for protein in ("JAK1", "JAK2"):
        job_name = f"{protein}_jh1"
        ranked = select_ranked_model(fold_root / job_name / "colabfold_output")
        dest = structure_dir / f"{job_name}.pdb"
        shutil.copy2(ranked, dest)
        published[protein] = dest
        print(f"Copied {ranked.name} -> {dest}")
    return published


def prepare_vina_and_receptors(
    published: dict[str, Path],
    docking_dir: Path,
    structure_dir: Path,
) -> dict:
    pdb_dir = docking_dir / "prepared_receptors" / "pdb"
    pdbqt_dir = docking_dir / "prepared_receptors" / "pdbqt"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    pdbqt_dir.mkdir(parents=True, exist_ok=True)

    receptor_map = {}
    alignment_records = {}

    for protein, model_path in published.items():
        job = {
            "name": f"{protein}_jh1",
            "protein": protein,
            "construct": "jh1",
            "sequence": jak.SEQUENCES[protein]["jh1"],
            "pdb_id": jak.TEMPLATES[protein][0],
            "template_path": jak.TEMPLATES[protein][1],
        }
        info = jak.assess_job(job, min_identity=0.90, max_mismatch_fraction=0.05)
        crystal_path = Path(job["template_path"])
        site_info = protein_prep.find_largest_organic_ligand(str(crystal_path))
        protein_prep.audit_hetero_residues(str(crystal_path), protein)

        aligned, rms, n_pairs = superpose_model_onto_crystal(
            model_path,
            crystal_path,
            info["jh1_start_1based"],
            info["overlap_full_start_1based"],
            info["overlap_full_end_1based"],
        )
        aligned_path = structure_dir / f"{protein}_jh1_aligned_to_{job['pdb_id']}.pdb"
        write_structure_pdb(aligned, aligned_path)

        cleaned_pdb = pdb_dir / f"{protein}.pdb"
        write_cleaned_pdb(aligned, cleaned_pdb)
        pdbqt_path = pdbqt_dir / f"{protein}.pdbqt"
        print(f"[>] Converting {protein} aligned JH1 PDB -> PDBQT (Meeko) ...")
        protein_prep.prepare_receptor_pdbqt_with_meeko(cleaned_pdb, pdbqt_path)
        ok, msg = protein_prep.validate_receptor_pdbqt(pdbqt_path)
        print(f"[{'OK' if ok else 'FAIL'}] {protein}: {msg}")
        if not ok:
            raise RuntimeError(f"{protein} PDBQT validation failed: {msg}")

        receptor_map[protein] = {
            "pdbqt": relpath(pdbqt_path),
            "pdb": relpath(cleaned_pdb),
            "center": list(site_info["centroid"]),
            "size": list(protein_prep.DEFAULT_BOX_SIZE),
            "exhaustiveness": protein_prep.DEFAULT_EXHAUSTIVENESS,
            "detected_ligand_resname": site_info["resname"],
            "detected_ligand_chain": site_info["chain"],
            "detected_ligand_natoms": site_info["n_atoms"],
            "source": "colabfold_jh1",
            "model_pdb": relpath(model_path),
            "aligned_pdb": relpath(aligned_path),
            "aligned_to": relpath(crystal_path),
            "superposition_ca_rmsd": rms,
            "superposition_n_ca": n_pairs,
        }
        alignment_records[protein] = {
            "model_pdb": relpath(model_path),
            "aligned_pdb": relpath(aligned_path),
            "crystal": relpath(crystal_path),
            "jh1": f"{info['jh1_start_1based']}-{info['jh1_end_1based']}",
            "overlap": (
                f"{info['overlap_full_start_1based']}-"
                f"{info['overlap_full_end_1based']}"
            ),
            "ca_rmsd": rms,
            "n_ca": n_pairs,
            "ligand": site_info,
        }
        print(
            f"{protein}: ligand {site_info['resname']} center={site_info['centroid']} "
            f"CA RMSD={rms:.3f} A over {n_pairs} atoms"
        )

    receptors_json = docking_dir / "receptors.json"
    receptors_json.write_text(json.dumps(receptor_map, indent=2) + "\n")
    (structure_dir / "alignment_report.json").write_text(
        json.dumps(alignment_records, indent=2) + "\n"
    )
    print(f"Wrote {receptors_json}")
    return receptor_map


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fold JAK JH1 models with LocalColabFold and prepare Vina/rDock "
            "receptors using the crystal bound-ligand boxes."
        )
    )
    parser.add_argument(
        "--fold-root",
        default=str(DEFAULT_FOLD_ROOT),
        help="ColabFold job directory. Default: results/colabfold_jak",
    )
    parser.add_argument(
        "--structure-dir",
        default=str(STRUCTURE_OUT_DIR),
        help="Where rank-1 JH1 models are copied. Default: Crystal Structures/colabfold_jh1",
    )
    parser.add_argument(
        "--docking-dir",
        default=str(DOCKING_DIR),
        help="Where receptors.json and prepared receptors are written.",
    )
    parser.add_argument("--num-recycle", type=int, default=12)
    parser.add_argument("--num-models", type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-fold",
        action="store_true",
        help="Use already-published or already-folded JH1 models.",
    )
    parser.add_argument(
        "--skip-rdock",
        action="store_true",
        help="Write Vina receptors.json only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fold_root = Path(args.fold_root).expanduser().resolve()
    structure_dir = Path(args.structure_dir).expanduser().resolve()
    docking_dir = Path(args.docking_dir).expanduser().resolve()

    if not args.skip_fold:
        rc = run_colabfold_jh1(
            fold_root,
            overwrite=args.overwrite,
            num_recycle=args.num_recycle,
            num_models=args.num_models,
        )
        if rc != 0:
            return rc
        published = publish_ranked_models(fold_root, structure_dir)
    else:
        published = {}
        for protein in ("JAK1", "JAK2"):
            dest = structure_dir / f"{protein}_jh1.pdb"
            if dest.is_file():
                published[protein] = dest
                continue
            ranked_dir = fold_root / f"{protein}_jh1" / "colabfold_output"
            if ranked_dir.is_dir():
                ranked = select_ranked_model(ranked_dir)
                structure_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ranked, dest)
                published[protein] = dest
                continue
            raise FileNotFoundError(
                f"No {protein} JH1 model at {dest} or {ranked_dir}. "
                "Run without --skip-fold after installing LocalColabFold."
            )

    ensure_meeko_on_path()
    receptor_map = prepare_vina_and_receptors(published, docking_dir, structure_dir)

    if not args.skip_rdock:
        rdock_dir = docking_dir / "rdock"
        rdock_prep.prepare_all(receptors_json=docking_dir / "receptors.json", prep_dir=rdock_dir)
        for protein, rec in receptor_map.items():
            rec["rdock_mol2"] = relpath(rdock_dir / f"{protein}_protein.mol2")
            rec["rdock_prm"] = relpath(rdock_dir / f"{protein}.prm")
            rec["rdock_as"] = relpath(rdock_dir / f"{protein}.as")
        (docking_dir / "receptors.json").write_text(
            json.dumps(receptor_map, indent=2) + "\n"
        )
        print(f"Updated {docking_dir / 'receptors.json'} with rDock paths")

    return 0


if __name__ == "__main__":
    sys.exit(main())
