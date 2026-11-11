#!/usr/bin/env python3
"""
Step 3: turn docked poses into PLIP interaction tables.

Every (target, engine) pair gets four tables, spanning two independent choices.

Pose scope
    top1        the single best-scoring pose, 0/1 occupancy
    frequency   a window of the N best-scoring poses, each cell the fraction
                of that window showing the interaction (denominator
                ``n_poses``). One table per window in ``frequency_windows``,
                by default the top 5 and the top 10.

Feature resolution
    type        exact PLIP interaction types, one column per
                ``RESIDUE:interaction``, e.g. ``ARG980:hydrogen_bond``
    residue     binary PLIP contact map, one column per ``RESIDUE``, set when
                that residue makes any interaction at all

So the files written per target and engine are::

    <engine>_<target>_type_top1.csv
    <engine>_<target>_type_frequency_top5.csv
    <engine>_<target>_type_frequency_top10.csv
    <engine>_<target>_residue_top1.csv
    <engine>_<target>_residue_frequency_top5.csv
    <engine>_<target>_residue_frequency_top10.csv

A window only ever narrows what a ligand already has: a ligand with 7 retained
poses contributes 5 to the top-5 table and 7 to the top-10 table, and its
``n_poses`` column says which.

plus ``docking_scores.csv``, ``docking_coverage_summary.csv`` and
``failures.csv`` across all targets and engines.

``dual_engine_docking.py`` calls ``write_all_tables`` at the end of a run. This
module is also a CLI, so tables can be rebuilt from an existing checkpoint
without re-docking:

    python src/screening/jh1_dual_engine/plip_tables.py --outdir results/jh1_dual_engine_crystal
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CHECKPOINT_NAME = "docking_checkpoint.jsonl"
RESIDUE_LABEL_RE = re.compile(r"^(.*?)(-?\d+)$")
DEFAULT_FREQUENCY_WINDOWS = (5, 10)


# ============================================================
# FEATURE PARSING AND COLUMN ORDER
# ============================================================

def parse_feature(feature: str) -> Tuple[str, str]:
    """``ARG980:hydrogen_bond`` -> ``("ARG980", "hydrogen_bond")``."""
    residue, _, interaction = feature.partition(":")
    return residue, interaction


def residue_sort_key(residue: str) -> Tuple[int, str]:
    """Order residue columns by sequence position, then name."""
    match = RESIDUE_LABEL_RE.match(residue)
    if not match:
        return (10**9, residue)
    return (int(match.group(2)), match.group(1))


def collect_feature_columns(records: Sequence[dict], resolution: str) -> List[str]:
    """Union of interaction columns across records, in a stable order."""
    features = {
        feature
        for record in records
        for pose in record.get("pose_features") or []
        for feature in pose
    }
    if resolution == "type":
        return sorted(features, key=lambda f: (*residue_sort_key(parse_feature(f)[0]), parse_feature(f)[1]))
    if resolution == "residue":
        return sorted({parse_feature(f)[0] for f in features}, key=residue_sort_key)
    raise ValueError(f"unknown resolution {resolution!r}")


def pose_labels(pose: Iterable[str], resolution: str) -> set:
    """Project one pose's features onto the requested resolution."""
    if resolution == "type":
        return set(pose)
    return {parse_feature(feature)[0] for feature in pose}


# ============================================================
# TABLE BUILDERS
# ============================================================

def score_column(engine: str) -> str:
    return f"{engine}_score"


def window_records(records: Sequence[dict], window: Optional[int]) -> List[dict]:
    """Truncate every record to its ``window`` best-scoring poses.

    Poses are stored best-first, so this is a plain slice. ``window=None``
    keeps everything.
    """
    if window is None:
        return list(records)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    windowed = []
    for record in records:
        trimmed = dict(record)
        trimmed["pose_features"] = (record.get("pose_features") or [])[:window]
        trimmed["pose_scores"] = (record.get("pose_scores") or [])[:window]
        windowed.append(trimmed)
    return windowed


def build_table(
    records: Sequence[dict],
    engine: str,
    resolution: str,
    scope: str,
    carry_cols: Sequence[str] = (),
    window: Optional[int] = None,
) -> pd.DataFrame:
    """One interaction table.

    ``scope="top1"`` scores the best pose only and writes 0/1. ``scope =
    "frequency"`` writes the fraction of the pose window carrying the feature.
    Columns come from the windowed poses, so a top-5 table never carries an
    all-zero column for an interaction only pose 7 made.
    """
    if scope not in ("top1", "frequency"):
        raise ValueError(f"unknown scope {scope!r}")

    records = window_records(records, 1 if scope == "top1" else window)
    records = sorted(records, key=lambda rec: rec.get("input_order", 10**18))
    columns = collect_feature_columns(records, resolution)
    id_cols = ["ligand_id", "smiles", *carry_cols, score_column(engine)]
    if scope == "frequency":
        id_cols.append("n_poses")

    rows: List[dict] = []
    for record in records:
        poses = record.get("pose_features") or []
        if not poses:
            continue
        carry = record.get("carry") or {}
        row = {
            "ligand_id": record.get("ligand_id"),
            "smiles": record.get("smiles"),
            **{col: carry.get(col) for col in carry_cols},
            score_column(engine): record.get("top_score"),
        }
        if scope == "top1":
            present = pose_labels(poses[0], resolution)
            for column in columns:
                row[column] = 1 if column in present else 0
        else:
            row["n_poses"] = len(poses)
            projected = [pose_labels(pose, resolution) for pose in poses]
            for column in columns:
                row[column] = sum(column in pose for pose in projected) / len(projected)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=id_cols + columns)
    return frame.reindex(columns=id_cols + columns)


def build_score_table(records: Sequence[dict], carry_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Flat per (ligand, target, engine) score summary across all retained poses."""
    rows: List[dict] = []
    for record in sorted(records, key=lambda rec: (rec.get("target", ""), rec.get("engine", ""), rec.get("input_order", 10**18))):
        scores = record.get("pose_scores") or []
        carry = record.get("carry") or {}
        rows.append(
            {
                "ligand_id": record.get("ligand_id"),
                "smiles": record.get("smiles"),
                **{col: carry.get(col) for col in carry_cols},
                "target": record.get("target"),
                "engine": record.get("engine"),
                "top_score": record.get("top_score"),
                "n_poses": len(scores),
                "pose_scores": ";".join(f"{score:.3f}" for score in scores),
            }
        )
    columns = [
        "ligand_id", "smiles", *carry_cols, "target", "engine",
        "top_score", "n_poses", "pose_scores",
    ]
    return pd.DataFrame(rows, columns=columns)


# ============================================================
# WRITING
# ============================================================

def write_all_tables(
    successes: Sequence[dict],
    failures: Sequence[dict],
    outdir: Path,
    targets: Sequence[str],
    engines: Sequence[str],
    carry_cols: Sequence[str] = (),
    frequency_windows: Sequence[int] = DEFAULT_FREQUENCY_WINDOWS,
) -> List[dict]:
    """Write every table for every (target, engine) and return coverage rows."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    windows = sorted({int(w) for w in frequency_windows})
    if any(w < 1 for w in windows):
        raise ValueError(f"frequency windows must be >= 1, got {frequency_windows}")
    coverage: List[dict] = []

    for target in targets:
        for engine in engines:
            records = [
                record
                for record in successes
                if record.get("target") == target and record.get("engine") == engine
            ]
            n_feature_cols: Dict[Tuple[str, str], int] = {}
            for resolution in ("type", "residue"):
                specs = [("top1", None, f"{resolution}_top1")]
                specs += [
                    ("frequency", window, f"{resolution}_frequency_top{window}")
                    for window in windows
                ]
                for scope, window, label in specs:
                    table = build_table(records, engine, resolution, scope, carry_cols, window)
                    table.to_csv(outdir / f"{engine}_{target.lower()}_{label}.csv", index=False)
                    n_id_cols = len(carry_cols) + (3 if scope == "top1" else 4)
                    n_feature_cols[(resolution, label)] = max(table.shape[1] - n_id_cols, 0)

            pose_counts = [len(record.get("pose_features") or []) for record in records]
            row = {
                "target": target,
                "engine": engine,
                "n_ligands": len(records),
                "n_poses_mean": float(np.mean(pose_counts)) if pose_counts else 0.0,
                "n_poses_median": float(np.median(pose_counts)) if pose_counts else 0.0,
                "n_poses_min": int(np.min(pose_counts)) if pose_counts else 0,
                "n_poses_max": int(np.max(pose_counts)) if pose_counts else 0,
            }
            for (resolution, label), count in n_feature_cols.items():
                row[f"n_{resolution}_columns_{label.split('_', 1)[1]}"] = count
            coverage.append(row)

            widest = max(windows)
            print(
                f"Wrote {engine}_{target.lower()}_*.csv "
                f"({len(records)} ligands; top1 and frequency windows "
                f"{', '.join(f'top{w}' for w in windows)}; "
                f"{n_feature_cols[('type', f'type_frequency_top{widest}')]} typed "
                f"/ {n_feature_cols[('residue', f'residue_frequency_top{widest}')]} residue "
                f"columns at top{widest})"
            )

    build_score_table(successes, carry_cols).to_csv(outdir / "docking_scores.csv", index=False)
    pd.DataFrame(failures).reindex(
        columns=["ligand_id", "smiles", "target", "engine", "error"]
    ).to_csv(outdir / "failures.csv", index=False)
    pd.DataFrame(coverage).to_csv(outdir / "docking_coverage_summary.csv", index=False)
    return coverage


# ============================================================
# CHECKPOINT REPLAY
# ============================================================

def read_checkpoint(path: Path, config_hash: Optional[str] = None) -> Tuple[List[dict], List[dict]]:
    """Return (successes, failures) from a JSONL checkpoint.

    Later lines win for a given (ligand, target, engine), so a resumed run that
    retried a failure reports the retry.
    """
    latest: Dict[Tuple[str, str, str], dict] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not all(key in record for key in ("ligand_id", "target", "engine")):
            continue
        if config_hash is not None and record.get("config_hash") != config_hash:
            continue
        latest[(record["ligand_id"], record["target"], record["engine"])] = record

    successes = [rec for rec in latest.values() if rec.get("error") is None]
    failures = [rec for rec in latest.values() if rec.get("error") is not None]
    return successes, failures


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", required=True, help="Run directory holding docking_checkpoint.jsonl.")
    parser.add_argument("--checkpoint", default=None, help="Override the checkpoint path.")
    parser.add_argument(
        "--config-hash",
        default=None,
        help="Only replay lines with this config hash. Default: whatever run_manifest.json records.",
    )
    parser.add_argument(
        "--frequency-windows",
        nargs="+",
        type=int,
        default=None,
        help="Pose windows for the frequency tables. Default: whatever run_manifest.json records, "
        f"else {' '.join(str(w) for w in DEFAULT_FREQUENCY_WINDOWS)}.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir).expanduser().resolve()
    checkpoint = Path(args.checkpoint) if args.checkpoint else outdir / CHECKPOINT_NAME
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint}")

    manifest_path = outdir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    config_hash = args.config_hash or manifest.get("config_hash")
    carry_cols = manifest.get("carry_columns", [])

    successes, failures = read_checkpoint(checkpoint, config_hash)
    targets = manifest.get("targets") or sorted({rec["target"] for rec in successes})
    engines = manifest.get("engines") or sorted({rec["engine"] for rec in successes})
    windows = (
        args.frequency_windows
        or manifest.get("frequency_windows")
        or list(DEFAULT_FREQUENCY_WINDOWS)
    )

    coverage = write_all_tables(
        successes, failures, outdir, targets, engines, carry_cols, windows
    )
    print(f"Rebuilt tables from {len(successes)} successful jobs ({len(failures)} failures).")
    print(json.dumps(coverage, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
