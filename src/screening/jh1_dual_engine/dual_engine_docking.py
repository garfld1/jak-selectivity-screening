#!/usr/bin/env python3
"""
Step 2: dock a compound set against JAK1/JAK2 JH1 with both Vina and rDock.

Each (ligand, target, engine) triple is one independent job. Vina and rDock
search the same box prepared by ``prepare_receptors.py``, so their poses live
in one receptor frame and stay directly comparable. PLIP runs on every
retained pose, and the per-pose interaction lists are what
``plip_tables.py`` turns into the single-pose and multi-pose tables.

Vina
    exhaustiveness is customizable (``--exhaustiveness``, default 16), with
    ``--num-poses`` modes, ``--vina-energy-range`` and ``--vina-min-rmsd``.
rDock
    ``--rdock-runs`` docking runs (default 50), then a heavy-atom diversity
    filter at ``--rdock-cluster-rmsd`` keeping at most ``--num-poses`` poses.

Each retained pose is rescored with MM/GBSA. Receptor and ligand hydrogens
are relaxed, and the complex energy is a single evaluation. More negative
energy ranks first, which is the same direction as the engine scores. The
engine score of that pose is what the tables still call ``vina_score`` or
``rdock_score``; ``mmgbsa_score`` is the value that set the order.
``--skip-mmgbsa`` restores engine-score ordering. ``--num-poses`` sets how
many poses are retained and so caps the interaction-frequency windows;
``--frequency-windows`` then chooses which windows get a table, by default
the top 5 and the top 10.

A config-aware JSONL checkpoint keyed by (ligand_id, target, engine) makes
runs resumable. ``--resume`` reuses a finished job that already has an
MM/GBSA score, including one scored by the previous hydrogen-minimization
protocol, and retries failures. Tables are rebuilt in full from the
checkpoint at the end of every run, so a resumed run still produces complete
outputs.

Run:
    python src/screening/jh1_dual_engine/dual_engine_docking.py \
        --input data/qsar_df.csv \
        --receptors docking/receptors/crystal/receptors.json \
        --outdir results/jh1_dual_engine_crystal \
        --exhaustiveness 16 --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plip_tables
from mmgbsa import METHOD_ID, PREVIOUS_METHOD_ID, cpu_threads_for_workers
from docking_core import (
    apply_pdbqt_heavy_coords,
    apply_sd_coords,
    dock_with_vina,
    generate_3d_conformer,
    heavy_index_in_pdbqt_order,
    mol_to_sd,
    pdbqt_block_to_ligand_pdb,
    plip_features,
    protein_only_pdb,
    rdkit_mol_to_pdbqt_text,
    run_rbdock,
    sd_record_to_ligand_pdb,
    select_diverse,
    split_pdbqt_models,
    split_sd,
    vina_score_from_block,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RECEPTORS = REPO_ROOT / "docking" / "receptors" / "crystal" / "receptors.json"
DEFAULT_OUTDIR = REPO_ROOT / "results" / "jh1_dual_engine"
DEFAULT_VINA_BIN = Path("~/bin/vina").expanduser()
DEFAULT_RDOCK_ROOT = REPO_ROOT / "ensemble_pipeline_new" / "tools" / "src" / "rDock"

CHECKPOINT_NAME = "docking_checkpoint.jsonl"
CONFIG_VERSION = 2

TARGETS = ("JAK1", "JAK2")
ENGINES = ("vina", "rdock")

DEFAULT_EXHAUSTIVENESS = 16
DEFAULT_NUM_POSES = 10
DEFAULT_VINA_ENERGY_RANGE = 10.0
DEFAULT_VINA_MIN_RMSD = 1.0
DEFAULT_VINA_TIMEOUT = 1800
DEFAULT_RDOCK_RUNS = 50
DEFAULT_RDOCK_CLUSTER_RMSD = 1.0
DEFAULT_RDOCK_TIMEOUT = 3600
RDOCK_PROTOCOL = "dock.prm"


@dataclass
class DockJob:
    ligand_id: str
    smiles: str
    carry: Dict[str, object]
    input_order: int
    target: str
    engine: str
    receptor_pdb: str
    receptor_pdbqt: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    exhaustiveness: int
    rdock_prm: str
    rdock_root: str
    vina_bin: str
    vina_cpu: int
    num_poses: int
    vina_energy_range: float
    vina_min_rmsd: float
    vina_timeout: int
    rdock_runs: int
    rdock_cluster_rmsd: float
    rdock_timeout: int
    config_hash: str
    openmm_threads: int = 1
    skip_mmgbsa: bool = False
    pose_dir: Optional[str] = None


# ============================================================
# INPUT
# ============================================================

def resolve_smiles_column(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(f"SMILES column {requested!r} not in input. Columns: {list(df.columns)}")
        return requested
    for name in ("smiles", "canonical_smiles", "SMILES"):
        if name in df.columns:
            return name
    raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")


def resolve_id_column(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(f"ID column {requested!r} not in input. Columns: {list(df.columns)}")
        return requested
    for name in ("ligand_id", "identifier", "zinc_id", "coconut_id"):
        if name in df.columns:
            return name
    return None


def _safe_value(value):
    """JSON-safe scalar; NaN and pandas NA become None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def load_ligands(
    path: Path,
    smiles_col: Optional[str],
    id_col: Optional[str],
    carry_cols: Sequence[str],
    limit: Optional[int],
) -> pd.DataFrame:
    """Load [ligand_id, smiles, carry...] deduped by SMILES, first occurrence wins."""
    df = pd.read_csv(path)
    smiles_name = resolve_smiles_column(df, smiles_col)
    id_name = resolve_id_column(df, id_col)
    missing = [col for col in carry_cols if col not in df.columns]
    if missing:
        raise ValueError(f"--carry-cols not in input: {', '.join(missing)}")

    rows: List[dict] = []
    seen = set()
    for _, row in df.iterrows():
        raw = row[smiles_name]
        if pd.isna(raw):
            continue
        smiles = str(raw).strip()
        if not smiles:
            continue
        if id_name is not None and not pd.isna(row[id_name]) and str(row[id_name]).strip():
            ligand_id = str(row[id_name]).strip()
        else:
            ligand_id = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
        if ligand_id in seen:
            continue
        seen.add(ligand_id)
        rows.append(
            {
                "ligand_id": ligand_id,
                "smiles": smiles,
                **{col: _safe_value(row[col]) for col in carry_cols},
            }
        )

    out = pd.DataFrame(rows, columns=["ligand_id", "smiles", *carry_cols])
    if limit is not None:
        out = out.head(limit).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No usable ligands in {path}")
    return out


# ============================================================
# RECEPTORS
# ============================================================

def resolve_receptors(
    receptors_json: Path,
    targets: Sequence[str],
    engines: Sequence[str],
) -> Dict[str, dict]:
    """Load receptors.json and make every referenced path absolute and present."""
    receptors_json = Path(receptors_json).expanduser().resolve()
    raw = json.loads(receptors_json.read_text())

    def absolute(value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        for base in (REPO_ROOT, receptors_json.parent):
            candidate = (base / path).resolve()
            if candidate.exists():
                return candidate
        return (REPO_ROOT / path).resolve()

    resolved: Dict[str, dict] = {}
    for target in targets:
        if target not in raw:
            raise KeyError(f"{target} not in {receptors_json}. Found: {list(raw)}")
        rec = dict(raw[target])
        for key in ("pdb", "pdbqt", "center", "size"):
            if key not in rec:
                raise ValueError(f"Receptor {target} is missing required key {key!r}")
        for key in ("pdb", "pdbqt"):
            path = absolute(rec[key])
            if not path.exists():
                raise FileNotFoundError(f"{target} {key} not found: {path}")
            rec[key] = str(path)
        if "rdock" in engines:
            if "rdock_prm" not in rec:
                raise ValueError(
                    f"Receptor {target} has no rdock_prm. Re-run prepare_receptors.py "
                    "without --skip-rdock, or drop rdock from --engines."
                )
            prm = absolute(rec["rdock_prm"])
            if not prm.exists():
                raise FileNotFoundError(f"{target} rDock prm not found: {prm}")
            rec["rdock_prm"] = str(prm)
        resolved[target] = rec
    return resolved


def preflight(args: argparse.Namespace, receptors: Dict[str, dict]) -> None:
    missing = []
    required = ["meeko", "plip", "rdkit"]
    if not args.skip_mmgbsa:
        required.extend(["openmm", "openff.toolkit", "openmmforcefields", "openff.nagl"])
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(f"Missing Python dependencies: {', '.join(missing)}")
    if "vina" in args.engines and not Path(args.vina).expanduser().exists():
        raise FileNotFoundError(f"Vina binary not found: {args.vina}")
    if "rdock" in args.engines:
        rbdock = Path(args.rdock_root).expanduser() / "bin" / "rbdock"
        if not rbdock.exists():
            raise FileNotFoundError(f"rbdock not found: {rbdock}")
    if not receptors:
        raise RuntimeError("No receptors resolved.")


# ============================================================
# DOCKING
# ============================================================

def _write_pose_archive(job: DockJob, blocks: Sequence[str], suffix: str) -> Optional[str]:
    if job.pose_dir is None:
        return None
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in job.ligand_id)[:80]
    path = Path(job.pose_dir) / job.target / f"{safe}_{job.engine}{suffix}.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write("".join(blocks))
    return str(path)


def _order_scored_rows(job: DockJob, rows: Sequence[tuple]) -> List[tuple]:
    """Sort ``(engine_score, features, payload, mol)`` by MM/GBSA, then drop non-finite energies.

    With ``--skip-mmgbsa`` the engine score is the sort key and the MM/GBSA
    slot stays None. Otherwise a pose whose energy is not finite is removed.
    """
    if not rows:
        return []
    if job.skip_mmgbsa:
        ordered = [(score, None, features, payload) for score, features, payload, _mol in rows]
        ordered.sort(key=lambda item: item[0])
        return ordered

    from mmgbsa import mmgbsa_pose_energies

    energies = mmgbsa_pose_energies(job.receptor_pdb, [row[3] for row in rows])
    if len(energies) != len(rows):
        raise RuntimeError("MM/GBSA returned a different number of energies than poses")
    kept = []
    for (score, features, payload, _mol), energy in zip(rows, energies):
        if not math.isfinite(energy):
            continue
        kept.append((score, float(energy), features, payload))
    if not kept:
        raise RuntimeError("MM/GBSA energies were non-finite for every retained pose")
    kept.sort(key=lambda item: item[1])
    return kept


def dock_vina(job: DockJob, mol, workdir: Path) -> Tuple[List[Tuple[float, Optional[float], List[str]]], Optional[str]]:
    lig_path = workdir / "lig.pdbqt"
    out_path = workdir / "docked.pdbqt"
    log_path = workdir / "vina.log"
    pdbqt_text = rdkit_mol_to_pdbqt_text(mol)
    lig_path.write_text(pdbqt_text)
    heavy_index = heavy_index_in_pdbqt_order(mol, pdbqt_text)

    dock_with_vina(
        vina_bin=job.vina_bin,
        ligand_pdbqt=str(lig_path),
        receptor_pdbqt=job.receptor_pdbqt,
        out_pdbqt=str(out_path),
        log_path=str(log_path),
        center=job.center,
        size=job.size,
        exhaustiveness=job.exhaustiveness,
        cpu=job.vina_cpu,
        timeout=job.vina_timeout,
        num_modes=job.num_poses,
        energy_range=job.vina_energy_range,
        min_rmsd=job.vina_min_rmsd,
    )

    protein = protein_only_pdb(Path(job.receptor_pdb).read_text())
    rows = []
    last_error: Optional[Exception] = None
    for i, block in enumerate(split_pdbqt_models(out_path.read_text())):
        try:
            score = vina_score_from_block(block)
            features = plip_features(protein, pdbqt_block_to_ligand_pdb(block), workdir, f"vina_{i}")
            pose_mol = apply_pdbqt_heavy_coords(mol, block, heavy_index)
        except Exception as exc:
            last_error = exc
            continue
        rows.append((score, features, block, pose_mol))

    if not rows and last_error is not None:
        raise RuntimeError(f"Vina poses failed PLIP: {last_error}") from last_error

    ordered = _order_scored_rows(job, rows)[: job.num_poses]
    archive = _write_pose_archive(job, [payload for _score, _mm, _features, payload in ordered], ".pdbqt")
    return [(score, mmgbsa, features) for score, mmgbsa, features, _payload in ordered], archive


def dock_rdock(job: DockJob, mol, workdir: Path) -> Tuple[List[Tuple[float, Optional[float], List[str]]], Optional[str]]:
    lig_path = workdir / "lig.sd"
    lig_path.write_text(mol_to_sd(mol))

    out_sd = run_rbdock(
        rdock_root=Path(job.rdock_root),
        prm_path=Path(job.rdock_prm),
        ligand_sd=lig_path,
        out_stem=workdir / "out",
        n_runs=job.rdock_runs,
        protocol=RDOCK_PROTOCOL,
        timeout=job.rdock_timeout,
    )
    kept = select_diverse(split_sd(out_sd.read_text()), job.num_poses, job.rdock_cluster_rmsd)
    if not kept:
        raise RuntimeError("rbdock produced no poses")

    protein = protein_only_pdb(Path(job.receptor_pdb).read_text())
    rows = []
    last_error: Optional[Exception] = None
    for i, (score, record) in enumerate(kept):
        try:
            features = plip_features(protein, sd_record_to_ligand_pdb(record), workdir, f"rdock_{i}")
            pose_mol = apply_sd_coords(mol, record)
        except Exception as exc:
            last_error = exc
            continue
        payload = record if record.endswith("$$$$\n") else record + "$$$$\n"
        rows.append((score, features, payload, pose_mol))

    if not rows and last_error is not None:
        raise RuntimeError(f"rDock poses failed PLIP: {last_error}") from last_error

    ordered = _order_scored_rows(job, rows)
    archive = _write_pose_archive(job, [payload for _score, _mm, _features, payload in ordered], ".sd")
    return [(score, mmgbsa, features) for score, mmgbsa, features, _payload in ordered], archive


def run_job(job: DockJob) -> dict:
    os.environ["OPENMM_DEFAULT_PLATFORM"] = "CPU"
    os.environ["OPENMM_CPU_THREADS"] = str(max(1, job.openmm_threads))
    record = {
        "ligand_id": job.ligand_id,
        "smiles": job.smiles,
        "carry": job.carry,
        "input_order": job.input_order,
        "target": job.target,
        "engine": job.engine,
        "config_hash": job.config_hash,
    }
    try:
        mol = generate_3d_conformer(job.smiles)
        prefix = f"{job.target}_{job.engine}_{job.ligand_id[:12]}_"
        with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
            workdir = Path(tmp)
            if job.engine == "vina":
                poses, archive = dock_vina(job, mol, workdir)
            elif job.engine == "rdock":
                poses, archive = dock_rdock(job, mol, workdir)
            else:
                raise RuntimeError(f"unknown engine {job.engine}")
        if not poses:
            raise RuntimeError(f"{job.engine} produced no PLIP-valid poses")
        mmgbsa_scores = [score for _engine, score, _features in poses]
        record.update(
            {
                "top_score": poses[0][0],
                "pose_scores": [score for score, _mmgbsa, _features in poses],
                "mmgbsa_score": mmgbsa_scores[0],
                "mmgbsa_pose_scores": [score for score in mmgbsa_scores if score is not None],
                "pose_features": [features for _score, _mmgbsa, features in poses],
                "pose_archive": archive,
                "mmgbsa_protocol": None if job.skip_mmgbsa else METHOD_ID,
                "error": None,
            }
        )
    except Exception as exc:
        record.update(
            {
                "top_score": None,
                "pose_scores": [],
                "mmgbsa_score": None,
                "mmgbsa_pose_scores": [],
                "pose_features": [],
                "pose_archive": None,
                "error": str(exc),
            }
        )
    return record


# ============================================================
# CHECKPOINT
# ============================================================

def checkpoint_key(record: dict) -> Tuple[str, str, str]:
    return (str(record["ligand_id"]), str(record["target"]), str(record["engine"]))


def _finite_mmgbsa(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def adopt_scored_successes(path: Path, config_hash: str) -> int:
    """Keep finished MM/GBSA jobs when the scoring protocol changes.

    A success with a finite ``mmgbsa_score`` is re-stamped with the current
    hash so ``--resume`` does not dock that compound again. Failures are left
    alone and get retried. A dry-run success has no MM/GBSA score, so it is
    not carried into a wet run.
    """
    if not path.exists():
        return 0
    successes, _ = plip_tables.read_checkpoint(path)
    adopted = 0
    for record in successes:
        if record.get("config_hash") == config_hash:
            continue
        if not _finite_mmgbsa(record.get("mmgbsa_score")):
            continue
        cloned = dict(record)
        cloned["config_hash"] = config_hash
        cloned["mmgbsa_protocol"] = record.get("mmgbsa_protocol") or PREVIOUS_METHOD_ID
        append_checkpoint(path, cloned)
        adopted += 1
    return adopted


def load_successes(path: Path, config_hash: str) -> Dict[Tuple[str, str, str], dict]:
    done: Dict[Tuple[str, str, str], dict] = {}
    if not path.exists():
        return done
    successes, _ = plip_tables.read_checkpoint(path, config_hash)
    for record in successes:
        done[checkpoint_key(record)] = record
    return done


def append_checkpoint(path: Path, record: dict) -> None:
    with open(path, "a") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def input_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _water_identity(receptors: Dict[str, dict]) -> Dict[str, list]:
    identity = {}
    for target, rec in receptors.items():
        identity[target] = [
            {
                "chain": water.get("chain"),
                "resseq": water.get("resseq"),
                "xyz": water.get("xyz"),
            }
            for water in rec.get("waters") or []
        ]
    return identity


def config_hash(args: argparse.Namespace, receptors: Dict[str, dict]) -> str:
    mmgbsa = None if args.skip_mmgbsa else {
        "method": METHOD_ID,
        "hydrogen_minimize": "receptor-and-ligand",
        "complex_evaluation": "single_point",
    }
    payload = {
        "version": CONFIG_VERSION,
        "input": str(Path(args.input).expanduser().resolve()),
        "limit": args.limit,
        "targets": list(args.targets),
        "engines": list(args.engines),
        "num_poses": args.num_poses,
        "exhaustiveness": args.exhaustiveness,
        "vina_energy_range": args.vina_energy_range,
        "vina_min_rmsd": args.vina_min_rmsd,
        "rdock_runs": args.rdock_runs,
        "rdock_cluster_rmsd": args.rdock_cluster_rmsd,
        "boxes": {
            target: {"center": list(rec["center"]), "size": list(rec["size"])}
            for target, rec in receptors.items()
        },
        "waters": _water_identity(receptors),
        "mmgbsa": mmgbsa,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ============================================================
# JOBS
# ============================================================

def build_jobs(
    ligands: pd.DataFrame,
    receptors: Dict[str, dict],
    args: argparse.Namespace,
    carry_cols: Sequence[str],
    cfg_hash: str,
    done: Dict[Tuple[str, str, str], dict],
    pose_dir: Optional[Path],
) -> List[DockJob]:
    jobs: List[DockJob] = []
    for order, row in ligands.iterrows():
        ligand_id = str(row["ligand_id"])
        for target in args.targets:
            rec = receptors[target]
            exhaustiveness = (
                args.exhaustiveness
                if args.exhaustiveness is not None
                else int(rec.get("exhaustiveness", DEFAULT_EXHAUSTIVENESS))
            )
            for engine in args.engines:
                if (ligand_id, target, engine) in done:
                    continue
                jobs.append(
                    DockJob(
                        ligand_id=ligand_id,
                        smiles=str(row["smiles"]),
                        carry={col: _safe_value(row[col]) for col in carry_cols},
                        input_order=int(order),
                        target=target,
                        engine=engine,
                        receptor_pdb=rec["pdb"],
                        receptor_pdbqt=rec["pdbqt"],
                        center=tuple(float(v) for v in rec["center"]),
                        size=tuple(float(v) for v in rec["size"]),
                        exhaustiveness=exhaustiveness,
                        rdock_prm=rec.get("rdock_prm", ""),
                        rdock_root=str(Path(args.rdock_root).expanduser()),
                        vina_bin=str(Path(args.vina).expanduser()),
                        vina_cpu=args.cpu_per_job,
                        num_poses=args.num_poses,
                        vina_energy_range=args.vina_energy_range,
                        vina_min_rmsd=args.vina_min_rmsd,
                        vina_timeout=args.vina_timeout,
                        rdock_runs=args.rdock_runs,
                        rdock_cluster_rmsd=args.rdock_cluster_rmsd,
                        rdock_timeout=args.rdock_timeout,
                        config_hash=cfg_hash,
                        openmm_threads=args.openmm_threads,
                        skip_mmgbsa=args.skip_mmgbsa,
                        pose_dir=str(pose_dir) if pose_dir is not None else None,
                    )
                )
    return jobs


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Compound CSV with a SMILES column.")
    parser.add_argument("--receptors", default=str(DEFAULT_RECEPTORS), help="receptors.json from prepare_receptors.py.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Run directory for tables and checkpoint.")
    parser.add_argument("--smiles-col", default=None, help="SMILES column name. Default: autodetect.")
    parser.add_argument("--id-col", default=None, help="Ligand ID column. Default: autodetect, else SHA-256 of SMILES.")
    parser.add_argument(
        "--carry-cols",
        nargs="*",
        default=[],
        help="Extra input columns to copy into every output table, e.g. delta_pIC50.",
    )
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument("--engines", nargs="+", default=list(ENGINES), choices=list(ENGINES))
    parser.add_argument("--limit", type=int, default=None, help="Dock only the first N distinct ligands.")

    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=None,
        help=f"Vina exhaustiveness. Default: the value in receptors.json, else {DEFAULT_EXHAUSTIVENESS}.",
    )
    parser.add_argument(
        "--num-poses",
        type=int,
        default=DEFAULT_NUM_POSES,
        help=f"Poses retained per engine. Default: {DEFAULT_NUM_POSES}",
    )
    parser.add_argument(
        "--frequency-windows",
        nargs="+",
        type=int,
        default=list(plip_tables.DEFAULT_FREQUENCY_WINDOWS),
        help="Pose windows for the interaction-frequency tables, on top of the top-1 table. "
        f"Default: {' '.join(str(w) for w in plip_tables.DEFAULT_FREQUENCY_WINDOWS)}",
    )
    parser.add_argument("--vina-energy-range", type=float, default=DEFAULT_VINA_ENERGY_RANGE)
    parser.add_argument("--vina-min-rmsd", type=float, default=DEFAULT_VINA_MIN_RMSD)
    parser.add_argument("--vina-timeout", type=int, default=DEFAULT_VINA_TIMEOUT)
    parser.add_argument("--vina", default=str(DEFAULT_VINA_BIN), help="AutoDock Vina executable.")

    parser.add_argument("--rdock-runs", type=int, default=DEFAULT_RDOCK_RUNS, help="rbdock -n value.")
    parser.add_argument("--rdock-cluster-rmsd", type=float, default=DEFAULT_RDOCK_CLUSTER_RMSD)
    parser.add_argument("--rdock-timeout", type=int, default=DEFAULT_RDOCK_TIMEOUT)
    parser.add_argument("--rdock-root", default=str(DEFAULT_RDOCK_ROOT), help="rDock installation prefix.")

    parser.add_argument("--workers", type=int, default=1, help="Process pool size.")
    parser.add_argument("--cpu-per-job", type=int, default=1, help="Vina --cpu inside each worker.")
    parser.add_argument("--resume", action="store_true", help="Reuse matching-config successes in the checkpoint.")
    parser.add_argument("--save-poses", action="store_true", help="Write retained poses to <outdir>/poses/.")
    parser.add_argument(
        "--skip-mmgbsa",
        action="store_true",
        help="Rank retained poses by the docking-engine score instead of MM/GBSA.",
    )
    parser.add_argument("--preflight-only", action="store_true", help="Validate inputs and binaries, then exit.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.workers < 1 or args.cpu_per_job < 1:
        raise ValueError("--workers and --cpu-per-job must be >= 1")
    if args.num_poses < 1:
        raise ValueError("--num-poses must be >= 1")
    if any(window < 1 for window in args.frequency_windows):
        raise ValueError("--frequency-windows must all be >= 1")
    too_wide = [window for window in args.frequency_windows if window > args.num_poses]
    if too_wide:
        raise ValueError(
            f"--frequency-windows {too_wide} exceed --num-poses {args.num_poses}; "
            "raise --num-poses so there are enough poses to aggregate."
        )
    if args.exhaustiveness is not None and args.exhaustiveness < 1:
        raise ValueError("--exhaustiveness must be >= 1")
    if args.rdock_runs < 1:
        raise ValueError("--rdock-runs must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")

    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    receptors = resolve_receptors(Path(args.receptors), args.targets, args.engines)
    preflight(args, receptors)
    if args.preflight_only:
        print("Preflight passed.")
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    pose_dir = outdir / "poses" if args.save_poses else None
    carry_cols = list(args.carry_cols)
    ligands = load_ligands(input_path, args.smiles_col, args.id_col, carry_cols, args.limit)
    print(f"Loaded {len(ligands)} distinct ligands from {input_path.name}.")

    cfg_hash = config_hash(args, receptors)
    checkpoint = outdir / CHECKPOINT_NAME
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(
            f"{checkpoint} already exists. Pass --resume to continue that run, "
            "or choose a new --outdir."
        )
    adopted = adopt_scored_successes(checkpoint, cfg_hash) if args.resume else 0
    done = load_successes(checkpoint, cfg_hash) if args.resume else {}
    if adopted:
        print(f"Keeping {adopted} finished MM/GBSA jobs from the previous protocol.")
    if done:
        print(f"Resuming: {len(done)} successes already checkpointed.")

    openmm_threads = cpu_threads_for_workers(args.workers)
    args.openmm_threads = openmm_threads
    os.environ["OPENMM_DEFAULT_PLATFORM"] = "CPU"
    os.environ["OPENMM_CPU_THREADS"] = str(openmm_threads)

    jobs = build_jobs(ligands, receptors, args, carry_cols, cfg_hash, done, pose_dir)
    total = len(ligands) * len(args.targets) * len(args.engines)
    print(
        f"Running {len(jobs)} of {total} ligand/target/engine jobs "
        f"with {args.workers} worker(s), {openmm_threads} OpenMM thread(s) each."
    )

    def accept(record: dict) -> None:
        append_checkpoint(checkpoint, record)
        if record["error"]:
            print(f"\nERROR {record['ligand_id'][:16]} {record['target']} {record['engine']}: {record['error']}")
        else:
            done[checkpoint_key(record)] = record

    if jobs:
        if args.workers == 1:
            for job in tqdm.tqdm(jobs, desc="DOCKING"):
                accept(run_job(job))
        else:
            executor = cf.ProcessPoolExecutor(max_workers=args.workers)
            futures = [executor.submit(run_job, job) for job in jobs]
            try:
                for future in tqdm.tqdm(cf.as_completed(futures), total=len(futures), desc="DOCKING"):
                    accept(future.result())
            except KeyboardInterrupt:
                print("\nStopping. Successful jobs are checkpointed; rerun with --resume.")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)

    successes, failures = plip_tables.read_checkpoint(checkpoint, cfg_hash)
    coverage = plip_tables.write_all_tables(
        successes, failures, outdir, args.targets, args.engines, carry_cols,
        args.frequency_windows,
    )

    manifest = {
        "config_hash": cfg_hash,
        "config_version": CONFIG_VERSION,
        "input": str(input_path),
        "input_sha256": input_sha256(input_path),
        "receptors_json": str(Path(args.receptors).expanduser().resolve()),
        "receptor_source": {t: rec.get("source") for t, rec in receptors.items()},
        "boxes": {
            target: {"center": list(rec["center"]), "size": list(rec["size"])}
            for target, rec in receptors.items()
        },
        "n_ligands": int(len(ligands)),
        "limit": args.limit,
        "targets": list(args.targets),
        "engines": list(args.engines),
        "carry_columns": carry_cols,
        "num_poses": args.num_poses,
        "frequency_windows": sorted(set(args.frequency_windows)),
        "vina_exhaustiveness": {
            target: (
                args.exhaustiveness
                if args.exhaustiveness is not None
                else int(rec.get("exhaustiveness", DEFAULT_EXHAUSTIVENESS))
            )
            for target, rec in receptors.items()
        },
        "vina_energy_range": args.vina_energy_range,
        "vina_min_rmsd": args.vina_min_rmsd,
        "rdock_runs": args.rdock_runs,
        "rdock_cluster_rmsd": args.rdock_cluster_rmsd,
        "rdock_protocol": RDOCK_PROTOCOL,
        "saved_poses": bool(args.save_poses),
        "mmgbsa": None if args.skip_mmgbsa else METHOD_ID,
        "mmgbsa_reused_previous_protocol": adopted,
        "openmm_platform": None if args.skip_mmgbsa else "CPU",
        "openmm_threads_per_worker": None if args.skip_mmgbsa else openmm_threads,
        "waters": _water_identity(receptors),
        "n_successful_jobs": len(successes),
        "n_failed_jobs": len(failures),
        "coverage": coverage,
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {outdir / 'run_manifest.json'}")
    print(f"Docking complete: {len(successes)} successful jobs, {len(failures)} failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
