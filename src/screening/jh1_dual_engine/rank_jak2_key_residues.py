#!/usr/bin/env python3
"""
Rank JAK2 ligands by dual-engine consensus on key residues.

For each key residue the consensus frequency is the fraction of top poses
both engines support:

    c_r = min(Vina frequency, rDock frequency)

A contact that only one engine sees contributes 0. The compound score is the
sum of ``c_r`` over the key residues, so a strong contact on a few residues
outranks a weak contact on many. Ties break by the count of residues with
``c_r >= 0.5``, then by ``predicted_delta_pIC50``.

``--selectivity`` ranks the same compounds for JAK2-over-JAK1 contact on
matched residues. For each pair the JAK2 frequency is the minimum of the two
engines, and the JAK1 frequency is the maximum. The pair score is JAK2 minus
JAK1, and the compound score is the sum of those pair scores. Using the
maximum on JAK1 means the matched contact is low only when both engines are
low. Ties break by the number of pairs where JAK2 exceeds JAK1, then by the
JAK2 consensus sum, then by ``predicted_delta_pIC50``.

Input tables are the binary residue-frequency files (any PLIP contact), not
the interaction-type tables. Frequencies are already divided by ``n_poses``.

    python src/screening/jh1_dual_engine/rank_jak2_key_residues.py
    python src/screening/jh1_dual_engine/rank_jak2_key_residues.py --selectivity
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOCKING_DIR = REPO_ROOT / "results" / "docking" / "jh1_colabfold_coconut"
KEY_RESIDUES = ("TYR931", "ASN859", "ASP939", "GLN853", "TYR934", "LYS857", "SER862")
# JAK2 residue, then the JAK1 residue treated as its counterpart.
SELECTIVITY_PAIRS = (
    ("TYR931", "PHE958"),
    ("ASN859", "HIS885"),
    ("ASP939", "GLU966"),
    ("GLN853", "ARG879"),
)
ID_COLUMNS = ("ligand_id", "smiles", "predicted_delta_pIC50", "QSAR_rank")


def rank_key_residues(
    vina: pd.DataFrame,
    rdock: pd.DataFrame,
    residues: Sequence[str] = KEY_RESIDUES,
) -> pd.DataFrame:
    """Inner-join the two frequency tables and rank by the sum of per-residue mins."""
    residues = tuple(residues)
    missing = {
        "vina": [residue for residue in residues if residue not in vina.columns],
        "rdock": [residue for residue in residues if residue not in rdock.columns],
    }
    missing = {engine: cols for engine, cols in missing.items() if cols}
    if missing:
        raise ValueError(f"frequency tables are missing key residues: {missing}")
    for column in (*ID_COLUMNS, "n_poses"):
        if column not in vina.columns:
            raise ValueError(f"Vina table is missing {column}")
    for column in ("ligand_id", "rdock_score", "n_poses"):
        if column not in rdock.columns:
            raise ValueError(f"rDock table is missing {column}")
    if "vina_score" not in vina.columns:
        raise ValueError("Vina table is missing vina_score")

    left = vina[list(ID_COLUMNS) + ["vina_score", "n_poses", *residues]].rename(
        columns={"n_poses": "n_poses_vina", **{residue: f"{residue}_vina" for residue in residues}}
    )
    right = rdock[["ligand_id", "rdock_score", "n_poses", *residues]].rename(
        columns={"n_poses": "n_poses_rdock", **{residue: f"{residue}_rdock" for residue in residues}}
    )
    merged = left.merge(right, on="ligand_id", how="inner")
    if merged.empty:
        raise ValueError("no ligands are present in both frequency tables")

    # Pose frequencies are k/n with n <= 10. Summing those binary fractions is
    # not exact, so round to 6 decimals before the threshold counts and the
    # sort. That keeps distinct pose fractions apart and lets equal sums tie.
    consensus_cols = []
    for residue in residues:
        column = f"{residue}_consensus"
        merged[column] = np.round(
            np.minimum(
                merged[f"{residue}_vina"].to_numpy(),
                merged[f"{residue}_rdock"].to_numpy(),
            ),
            6,
        )
        consensus_cols.append(column)

    consensus = merged[consensus_cols]
    merged["consensus_sum"] = np.round(consensus.sum(axis=1).to_numpy(), 6)
    merged["n_consensus_ge_0.5"] = (consensus >= 0.5).sum(axis=1).astype(int)
    merged["n_consensus_ge_0.3"] = (consensus >= 0.3).sum(axis=1).astype(int)

    merged = merged.sort_values(
        ["consensus_sum", "n_consensus_ge_0.5", "predicted_delta_pIC50", "ligand_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    merged.insert(0, "rank", np.arange(1, len(merged) + 1))

    per_residue = [
        column
        for residue in residues
        for column in (f"{residue}_consensus", f"{residue}_vina", f"{residue}_rdock")
    ]
    columns = [
        "rank",
        *ID_COLUMNS,
        "vina_score",
        "rdock_score",
        "n_poses_vina",
        "n_poses_rdock",
        "consensus_sum",
        "n_consensus_ge_0.5",
        "n_consensus_ge_0.3",
        *per_residue,
    ]
    return merged[columns]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing {missing}")


def _frequency_frame(
    frame: pd.DataFrame,
    id_cols: Sequence[str],
    score_col: str,
    score_out: str,
    n_out: str,
    residues: Sequence[str],
    engine: str,
) -> pd.DataFrame:
    """Keep one engine's score, pose count, and residue frequencies."""
    subset = frame[list(id_cols) + [score_col, "n_poses", *residues]].copy()
    return subset.rename(
        columns={
            score_col: score_out,
            "n_poses": n_out,
            **{residue: f"{residue}_{engine}" for residue in residues},
        }
    )


def rank_selective_residues(
    vina_jak2: pd.DataFrame,
    rdock_jak2: pd.DataFrame,
    vina_jak1: pd.DataFrame,
    rdock_jak1: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]] = SELECTIVITY_PAIRS,
) -> pd.DataFrame:
    """Rank ligands by JAK2 consensus frequency minus the matched JAK1 frequency.

    JAK2 consensus is ``min(Vina, rDock)``. The matched JAK1 frequency is
    ``max(Vina, rDock)``, so the JAK1 contact is low only when both engines
    are low. The compound score sums ``JAK2 consensus - JAK1 upper`` over pairs.
    """
    pairs = tuple(pairs)
    jak2_residues = [jak2 for jak2, _ in pairs]
    jak1_residues = [jak1 for _, jak1 in pairs]
    _require_columns(vina_jak2, [*ID_COLUMNS, "vina_score", "n_poses", *jak2_residues], "Vina JAK2")
    _require_columns(rdock_jak2, ["ligand_id", "rdock_score", "n_poses", *jak2_residues], "rDock JAK2")
    _require_columns(vina_jak1, ["ligand_id", "vina_score", "n_poses", *jak1_residues], "Vina JAK1")
    _require_columns(rdock_jak1, ["ligand_id", "rdock_score", "n_poses", *jak1_residues], "rDock JAK1")

    merged = _frequency_frame(
        vina_jak2, ID_COLUMNS, "vina_score", "vina_score_jak2", "n_poses_vina_jak2", jak2_residues, "vina"
    )
    for part in (
        _frequency_frame(rdock_jak2, ["ligand_id"], "rdock_score", "rdock_score_jak2", "n_poses_rdock_jak2", jak2_residues, "rdock"),
        _frequency_frame(vina_jak1, ["ligand_id"], "vina_score", "vina_score_jak1", "n_poses_vina_jak1", jak1_residues, "vina"),
        _frequency_frame(rdock_jak1, ["ligand_id"], "rdock_score", "rdock_score_jak1", "n_poses_rdock_jak1", jak1_residues, "rdock"),
    ):
        merged = merged.merge(part, on="ligand_id", how="inner")
    if merged.empty:
        raise ValueError("no ligands are present in all four frequency tables")

    # Same rounding as the JAK2-only score: pose fractions are k/n with n <= 10,
    # and sums of those tenths are not exact in binary floating point.
    diff_cols = []
    jak2_cols = []
    for jak2, jak1 in pairs:
        jak2_col = f"{jak2}_consensus"
        jak1_col = f"{jak1}_upper"
        diff_col = f"{jak2}_minus_{jak1}"
        merged[jak2_col] = np.round(
            np.minimum(merged[f"{jak2}_vina"].to_numpy(), merged[f"{jak2}_rdock"].to_numpy()),
            6,
        )
        merged[jak1_col] = np.round(
            np.maximum(merged[f"{jak1}_vina"].to_numpy(), merged[f"{jak1}_rdock"].to_numpy()),
            6,
        )
        merged[diff_col] = np.round(merged[jak2_col].to_numpy() - merged[jak1_col].to_numpy(), 6)
        diff_cols.append(diff_col)
        jak2_cols.append(jak2_col)

    merged["selectivity_sum"] = np.round(merged[diff_cols].sum(axis=1).to_numpy(), 6)
    merged["jak2_consensus_sum"] = np.round(merged[jak2_cols].sum(axis=1).to_numpy(), 6)
    merged["n_pairs_jak2_gt_jak1"] = (merged[diff_cols] > 0).sum(axis=1).astype(int)

    merged = merged.sort_values(
        ["selectivity_sum", "n_pairs_jak2_gt_jak1", "jak2_consensus_sum", "predicted_delta_pIC50", "ligand_id"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    merged.insert(0, "rank", np.arange(1, len(merged) + 1))

    per_pair = [
        column
        for jak2, jak1 in pairs
        for column in (
            f"{jak2}_consensus",
            f"{jak2}_vina",
            f"{jak2}_rdock",
            f"{jak1}_upper",
            f"{jak1}_vina",
            f"{jak1}_rdock",
            f"{jak2}_minus_{jak1}",
        )
    ]
    columns = [
        "rank",
        *ID_COLUMNS,
        "vina_score_jak2",
        "rdock_score_jak2",
        "vina_score_jak1",
        "rdock_score_jak1",
        "n_poses_vina_jak2",
        "n_poses_rdock_jak2",
        "n_poses_vina_jak1",
        "n_poses_rdock_jak1",
        "selectivity_sum",
        "jak2_consensus_sum",
        "n_pairs_jak2_gt_jak1",
        *per_pair,
    ]
    return merged[columns]


def _parse_pairs(raw: Optional[Sequence[str]]) -> Tuple[Tuple[str, str], ...]:
    if not raw:
        return SELECTIVITY_PAIRS
    pairs = []
    for item in raw:
        jak2, separator, jak1 = item.partition(":")
        if not separator or not jak2 or not jak1:
            raise ValueError(f"pair must be JAK2:JAK1, got {item!r}")
        pairs.append((jak2, jak1))
    return tuple(pairs)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--vina",
        type=Path,
        default=DEFAULT_DOCKING_DIR / "vina_jak2_residue_frequency_top10.csv",
        help="Vina JAK2 binary residue-frequency table for the top poses.",
    )
    parser.add_argument(
        "--rdock",
        type=Path,
        default=DEFAULT_DOCKING_DIR / "rdock_jak2_residue_frequency_top10.csv",
        help="rDock JAK2 binary residue-frequency table for the top poses.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ranked table. Default depends on --selectivity.",
    )
    parser.add_argument(
        "--residues",
        nargs="+",
        default=list(KEY_RESIDUES),
        help="Key JAK2 residue labels for the JAK2-only ranking.",
    )
    parser.add_argument(
        "--selectivity",
        action="store_true",
        help="Rank by JAK2 consensus minus the matched JAK1 frequency.",
    )
    parser.add_argument(
        "--vina-jak1",
        type=Path,
        default=DEFAULT_DOCKING_DIR / "vina_jak1_residue_frequency_top10.csv",
        help="Vina JAK1 binary residue-frequency table for the top poses.",
    )
    parser.add_argument(
        "--rdock-jak1",
        type=Path,
        default=DEFAULT_DOCKING_DIR / "rdock_jak1_residue_frequency_top10.csv",
        help="rDock JAK1 binary residue-frequency table for the top poses.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Matched residues as JAK2:JAK1. Default: TYR931:PHE958 ASN859:HIS885 ASP939:GLU966 GLN853:ARG879.",
    )
    parser.add_argument("--top", type=int, default=10, help="How many ranks to print. Default: 10.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.selectivity:
        pairs = _parse_pairs(args.pairs)
        ranked = rank_selective_residues(
            pd.read_csv(args.vina),
            pd.read_csv(args.rdock),
            pd.read_csv(args.vina_jak1),
            pd.read_csv(args.rdock_jak1),
            pairs,
        )
        output = args.output or (DEFAULT_DOCKING_DIR / "jak2_vs_jak1_key_residue_consensus.csv")
        preview_cols = ["rank", "ligand_id", "selectivity_sum", "n_pairs_jak2_gt_jak1", "jak2_consensus_sum"]
        preview_cols += [f"{jak2}_minus_{jak1}" for jak2, jak1 in pairs]
    else:
        ranked = rank_key_residues(
            pd.read_csv(args.vina),
            pd.read_csv(args.rdock),
            args.residues,
        )
        output = args.output or (DEFAULT_DOCKING_DIR / "jak2_key_residue_consensus.csv")
        preview_cols = ["rank", "ligand_id", "consensus_sum", "n_consensus_ge_0.5", "n_consensus_ge_0.3"]
        preview_cols += [f"{residue}_consensus" for residue in args.residues]

    output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output, index=False)
    print(f"Wrote {len(ranked)} ranked ligands to {output}")
    print(ranked.head(args.top)[preview_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
