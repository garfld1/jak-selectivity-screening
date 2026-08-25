#!/usr/bin/env python3
"""
Annotate existing binary PLIP docking-result CSVs with exact PLIP interaction types.

Expected layout:
    results/
    ├── saved_complexes/
    │   └── *.pdb
    └── plip results/
        ├── JAK1_docking_results_jak1_selective.csv
        ├── JAK1_docking_results_jak2_selective.csv
        ├── JAK1_docking_results_nonselective.csv
        ├── JAK1_docking_results_validation.csv
        ├── JAK2_docking_results_jak1_selective.csv
        ├── JAK2_docking_results_jak2_selective.csv
        ├── JAK2_docking_results_nonselective.csv
        └── JAK2_docking_results_validation.csv

Ligand-ID sources:
    docking/docking_prep/ligands_pdbqt.csv
    docking/docking_prep/validation_set_pdbqt.csv

What this script does:
1. Loads ligand_id, SMILES, and PDBQT from both ligand-source CSVs.
2. Indexes every saved complex in results/saved_complexes.
3. Uses the first 8 characters of ligand_id to find candidate complex files,
   but DOES NOT trust the prefix alone when duplicates exist.
4. Resolves the exact ligand by comparing the saved complex's LIG atom
   names/elements against the original ligand PDBQT atom names/elements.
   This is robust to docked-coordinate changes because the atom ordering and
   names are preserved by the complex-writing code.
5. For each ligand+isoform, selects the first matching complex after sorting
   filenames alphabetically. Duplicate complexes are therefore ignored after
   exact matching.
6. Runs PLIP once for each selected (ligand_id, JAK1/JAK2) pair and caches it.
7. Replaces binary residue values of 1 with the exact PLIP interaction type(s).
   Zeros remain zero.
8. Writes annotated copies of all input result CSVs and a mapping/audit CSV.

If a binary 1 cannot be resolved to an interaction type, the original 1 is
left in place and a warning is written to the audit CSV rather than inventing
an interaction.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from plip.structure.preparation import PDBComplex


# ---------------------------------------------------------------------------
# PLIP interaction attributes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LigandSource:
    ligand_id: str
    smiles: str
    pdbqt: str


@dataclass(frozen=True)
class ComplexCandidate:
    path: Path
    ligand_prefix: str
    isoform: str


# ---------------------------------------------------------------------------
# PDBQT / PDB atom-signature helpers
# ---------------------------------------------------------------------------

def normalize_element(raw: str) -> str:
    """Normalize an element string such as C, CL, Cl -> canonical form."""
    s = re.sub(r"[^A-Za-z]", "", raw or "").strip()
    if not s:
        return ""
    if len(s) == 1:
        return s.upper()
    return s[0].upper() + s[1:].lower()


def infer_element_from_pdbqt_line(line: str) -> str:
    """
    Get element from the PDBQT atom-type token in columns 67+.

    Mirrors the mapping used by docking_analysis_fixed.py.
    """
    ad_type_to_element = {
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

    tail = line[66:].strip().split()
    if tail:
        token = re.sub(r"[^A-Za-z]", "", tail[-1]).upper()
        if token in ad_type_to_element:
            return ad_type_to_element[token]

    # Fallback to the PDB element column if present.
    if len(line) >= 78:
        elem = normalize_element(line[76:78])
        if elem:
            return elem

    return ""


def pdbqt_atom_signature(pdbqt_text: str) -> Tuple[Tuple[str, str], ...]:
    """
    Extract ordered (atom_name, element) tuples from the first PDBQT pose.

    The existing docking script preserves atom order and atom names when it
    converts the PDBQT ligand into the saved complex PDB, so this signature is
    coordinate-independent and can distinguish duplicate 8-character prefixes.
    """
    atoms: List[Tuple[str, str]] = []

    for line in pdbqt_text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue

        atom_name = line[12:16].strip()
        element = infer_element_from_pdbqt_line(line)
        atoms.append((atom_name, element))

    if not atoms:
        raise ValueError("No ATOM/HETATM records found in PDBQT text.")

    return tuple(atoms)


def complex_ligand_signature(pdb_path: Path) -> Tuple[Tuple[str, str], ...]:
    """
    Extract the ordered (atom_name, element) signature of the saved LIG block.

    The user's current docking script writes the docked ligand as:
        HETATM ... LIG Z 1 ...
    """
    atoms: List[Tuple[str, str]] = []

    with pdb_path.open("r", errors="replace") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                chain = line[21:22].strip()
                try:
                    resseq = int(line[22:26].strip())
                except ValueError:
                    resseq = None

                if resname != "LIG":
                    continue
                if chain and chain != "Z":
                    continue
                if resseq is not None and resseq != 1:
                    continue

                atom_name = line[12:16].strip()
                element = normalize_element(line[76:78]) if len(line) >= 78 else ""
                atoms.append((atom_name, element))

            # The ligand is appended at the end of the complex by the current
            # docking script, so once the ligand TER is reached we can stop.
            elif line.startswith("TER") and atoms:
                break

    if not atoms:
        raise ValueError(f"No LIG atoms found in {pdb_path}")

    return tuple(atoms)


# ---------------------------------------------------------------------------
# Ligand-source loading
# ---------------------------------------------------------------------------

def load_ligand_sources(csv_paths: Sequence[Path]) -> Dict[str, LigandSource]:
    """
    Load the first occurrence of each ligand_id across the two PDBQT CSVs.

    The first occurrence is deterministic according to csv_paths order and
    row order within each file.
    """
    sources: Dict[str, LigandSource] = {}

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"Ligand source CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, usecols=lambda c: c in {"ligand_id", "smiles", "PDBQT"})

        required = {"ligand_id", "PDBQT"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

        for _, row in df.iterrows():
            ligand_id = str(row["ligand_id"]).strip()
            if not ligand_id or ligand_id.lower() == "nan":
                continue

            if ligand_id in sources:
                continue

            pdbqt = row["PDBQT"]
            if pd.isna(pdbqt) or not str(pdbqt).strip():
                continue

            smiles = "" if "smiles" not in df.columns or pd.isna(row.get("smiles")) else str(row["smiles"]).strip()
            sources[ligand_id] = LigandSource(
                ligand_id=ligand_id,
                smiles=smiles,
                pdbqt=str(pdbqt),
            )

    return sources


# ---------------------------------------------------------------------------
# Saved-complex indexing
# ---------------------------------------------------------------------------

COMPLEX_FILENAME_RE = re.compile(
    r"^(?P<prefix>[^_]+)_(?P<iso>JAK1|JAK2)_(?P<suffix>[^_]+)_complex\.pdb$",
    re.IGNORECASE,
)


def index_saved_complexes(complex_dir: Path) -> Dict[Tuple[str, str], List[ComplexCandidate]]:
    """
    Index complexes by (first-8-character ligand prefix, isoform).

    Candidate lists are sorted by filename so "first instance" is deterministic.
    """
    if not complex_dir.exists():
        raise FileNotFoundError(f"Saved complex directory not found: {complex_dir}")

    index: Dict[Tuple[str, str], List[ComplexCandidate]] = defaultdict(list)

    for path in sorted(complex_dir.glob("*.pdb"), key=lambda p: p.name):
        m = COMPLEX_FILENAME_RE.match(path.name)
        if not m:
            continue

        prefix = m.group("prefix")
        iso = m.group("iso").upper()

        # The current writer uses the first 8 characters of ligand_id.
        # Require at least 8 to avoid accidental matches on short names.
        if len(prefix) < 8:
            continue

        index[(prefix[:8], iso)].append(
            ComplexCandidate(path=path, ligand_prefix=prefix[:8], isoform=iso)
        )

    return index


# ---------------------------------------------------------------------------
# Exact ligand matching
# ---------------------------------------------------------------------------

class ComplexMatcher:
    """
    Resolve a full ligand_id to saved JAK1/JAK2 complexes.

    Matching strategy:
      1. Prefix-match on ligand_id[:8] to reduce candidates.
      2. Compare the ordered ligand atom signature.
      3. If multiple candidates have the same exact signature, select the first
         filename alphabetically and report the duplicate count.
    """

    def __init__(
        self,
        source_map: Dict[str, LigandSource],
        complex_index: Dict[Tuple[str, str], List[ComplexCandidate]],
    ) -> None:
        self.source_map = source_map
        self.complex_index = complex_index
        self._complex_signature_cache: Dict[Path, Tuple[Tuple[str, str], ...]] = {}

    def get_complex_signature(self, path: Path) -> Tuple[Tuple[str, str], ...]:
        if path not in self._complex_signature_cache:
            self._complex_signature_cache[path] = complex_ligand_signature(path)
        return self._complex_signature_cache[path]

    def match_one(
        self,
        ligand_id: str,
        isoform: str,
    ) -> Tuple[Optional[Path], int, str]:
        """
        Returns:
            selected_path
            number_of_exact_signature_matches
            status
        """
        source = self.source_map.get(ligand_id)
        if source is None:
            return None, 0, "missing_ligand_source"

        prefix = ligand_id[:8]
        expected = pdbqt_atom_signature(source.pdbqt)

        candidates = self.complex_index.get((prefix, isoform.upper()), [])
        if not candidates:
            return None, 0, "no_prefix_candidates"

        exact_matches: List[Path] = []

        for candidate in candidates:
            try:
                observed = self.get_complex_signature(candidate.path)
            except Exception:
                continue

            if observed == expected:
                exact_matches.append(candidate.path)

        if not exact_matches:
            return None, 0, "no_exact_atom_signature_match"

        exact_matches.sort(key=lambda p: p.name)

        if len(exact_matches) == 1:
            return exact_matches[0], 1, "unique_exact_match"

        return exact_matches[0], len(exact_matches), "duplicate_exact_matches_first_selected"


# ---------------------------------------------------------------------------
# PLIP analysis
# ---------------------------------------------------------------------------

def residue_key_from_entry(entry) -> Optional[str]:
    """
    Convert a PLIP interaction entry into the same residue key format used by
    the original binary CSVs, e.g. ARG1007.
    """
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

    if resnr is None or restype is None:
        return None

    try:
        return f"{str(restype).upper()}{int(resnr)}"
    except (TypeError, ValueError):
        return None


def run_plip_interactions(complex_pdb: Path) -> Dict[str, Set[str]]:
    """
    Run PLIP on one saved complex.

    Returns:
        residue -> set(interaction_type)
    """
    pc = PDBComplex()
    pc.load_pdb(str(complex_pdb))
    pc.analyze()

    residue_to_types: Dict[str, Set[str]] = defaultdict(set)

    for _, interactions in pc.interaction_sets.items():
        for attr, label in PLIP_ATTRS.items():
            entries = getattr(interactions, attr, None)
            if not entries:
                continue

            for entry in entries:
                residue = residue_key_from_entry(entry)
                if residue:
                    residue_to_types[residue].add(label)

    return residue_to_types


def format_interaction_types(types: Iterable[str]) -> str:
    """
    Deterministic display order for multiple interaction types.
    """
    preferred_order = [
        "hydrogen_bond",
        "hydrophobic",
        "pi_stacking",
        "pi_cation",
        "salt_bridge",
        "halogen_bond",
        "water_bridge",
        "metal_complex",
    ]
    rank = {name: i for i, name in enumerate(preferred_order)}
    ordered = sorted(set(types), key=lambda x: (rank.get(x, 999), x))
    return "; ".join(ordered)


# ---------------------------------------------------------------------------
# CSV annotation
# ---------------------------------------------------------------------------

def is_binary_positive(value) -> bool:
    if pd.isna(value):
        return False
    try:
        return float(value) != 0.0
    except (TypeError, ValueError):
        return str(value).strip().lower() in {"1", "true", "yes"}


def annotate_result_csv(
    input_csv: Path,
    output_csv: Path,
    isoform: str,
    ligand_to_interactions: Dict[str, Dict[str, Set[str]]],
    unresolved_log: List[Dict[str, str]],
) -> None:
    df = pd.read_csv(input_csv)

    if "ligand_id" not in df.columns:
        raise ValueError(f"{input_csv} does not contain a ligand_id column.")

    metadata_cols = {"ligand_id", "vina_score"}
    residue_cols = [c for c in df.columns if c not in metadata_cols]

    for idx, row in df.iterrows():
        ligand_id = str(row["ligand_id"]).strip()
        interaction_map = ligand_to_interactions.get(ligand_id, {})

        for residue in residue_cols:
            if not is_binary_positive(row[residue]):
                continue

            types = interaction_map.get(residue)
            if types:
                df.at[idx, residue] = format_interaction_types(types)
            else:
                # Keep the original 1 instead of silently fabricating an
                # interaction. Record it so the user can inspect the case.
                unresolved_log.append({
                    "file": str(input_csv),
                    "ligand_id": ligand_id,
                    "isoform": isoform,
                    "residue": residue,
                    "reason": "binary_1_but_plip_returned_no_matching_interaction",
                })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace binary PLIP residue columns with exact PLIP interaction types."
    )
    parser.add_argument(
        "--complex-dir",
        default="results/saved_complexes",
        help="Directory containing saved complex PDB files.",
    )
    parser.add_argument(
        "--results-dir",
        default="results/plip results",
        help="Directory containing the 8 binary PLIP CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plip results_annotated",
        help="Directory for annotated CSVs and audit files.",
    )
    parser.add_argument(
        "--ligand-csv",
        default="docking/docking_prep/ligands_pdbqt.csv",
        help="Training/reference ligand PDBQT CSV.",
    )
    parser.add_argument(
        "--validation-ligand-csv",
        default="docking/docking_prep/validation_set_pdbqt.csv",
        help="Validation ligand PDBQT CSV.",
    )
    args = parser.parse_args()

    complex_dir = Path(args.complex_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    ligand_csvs = [
        Path(args.ligand_csv),
        Path(args.validation_ligand_csv),
    ]

    expected_result_files = [
        "JAK1_docking_results_jak1_selective.csv",
        "JAK1_docking_results_jak2_selective.csv",
        "JAK1_docking_results_nonselective.csv",
        "JAK1_docking_results_validation.csv",
        "JAK2_docking_results_jak1_selective.csv",
        "JAK2_docking_results_jak2_selective.csv",
        "JAK2_docking_results_nonselective.csv",
        "JAK2_docking_results_validation.csv",
    ]

    print("Loading ligand PDBQT source files...")
    source_map = load_ligand_sources(ligand_csvs)
    print(f"  Loaded {len(source_map):,} unique ligand IDs.")

    print("Indexing saved complexes...")
    complex_index = index_saved_complexes(complex_dir)
    n_complexes = sum(len(v) for v in complex_index.values())
    print(f"  Indexed {n_complexes:,} candidate complex PDBs.")

    matcher = ComplexMatcher(source_map, complex_index)

    # Collect every ligand actually present in the eight result CSVs.
    result_paths: List[Path] = []
    ligand_ids_needed: Set[str] = set()

    for filename in expected_result_files:
        path = results_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected result CSV not found: {path}")
        result_paths.append(path)

        ids = pd.read_csv(path, usecols=["ligand_id"])["ligand_id"].astype(str).str.strip()
        ligand_ids_needed.update(ids.tolist())

    print(f"Found {len(ligand_ids_needed):,} unique ligand IDs in the 8 result CSVs.")

    # Match every ligand to one JAK1 and one JAK2 complex.
    mapping_rows: List[Dict[str, str]] = []
    selected_complexes: Dict[Tuple[str, str], Optional[Path]] = {}

    for count, ligand_id in enumerate(sorted(ligand_ids_needed), start=1):
        if count == 1 or count % 250 == 0 or count == len(ligand_ids_needed):
            print(f"Matching complexes: {count:,}/{len(ligand_ids_needed):,}")

        for isoform in ("JAK1", "JAK2"):
            path, n_exact, status = matcher.match_one(ligand_id, isoform)
            selected_complexes[(ligand_id, isoform)] = path

            mapping_rows.append({
                "ligand_id": ligand_id,
                "isoform": isoform,
                "complex_file": "" if path is None else str(path),
                "exact_signature_matches": str(n_exact),
                "status": status,
            })

    # Run PLIP exactly once for each selected ligand+isoform pair.
    ligand_to_interactions: Dict[Tuple[str, str], Dict[str, Set[str]]] = {}
    plip_failures: List[Dict[str, str]] = []

    total_plip = sum(path is not None for path in selected_complexes.values())
    completed_plip = 0

    print(f"Running PLIP on {total_plip:,} selected complexes...")

    for key in sorted(selected_complexes):
        ligand_id, isoform = key
        complex_path = selected_complexes[key]

        if complex_path is None:
            ligand_to_interactions[key] = {}
            continue

        try:
            ligand_to_interactions[key] = run_plip_interactions(complex_path)
        except Exception as exc:
            ligand_to_interactions[key] = {}
            plip_failures.append({
                "ligand_id": ligand_id,
                "isoform": isoform,
                "complex_file": str(complex_path),
                "error": repr(exc),
            })

        completed_plip += 1
        if completed_plip == 1 or completed_plip % 250 == 0 or completed_plip == total_plip:
            print(f"PLIP: {completed_plip:,}/{total_plip:,}")

    # Re-key by ligand for convenient CSV annotation.
    interaction_lookup: Dict[Tuple[str, str], Dict[str, Set[str]]] = ligand_to_interactions

    unresolved_log: List[Dict[str, str]] = []

    print("Writing annotated CSVs...")

    for input_csv in result_paths:
        name_upper = input_csv.name.upper()
        isoform = "JAK1" if name_upper.startswith("JAK1_") else "JAK2"

        # Preserve the original filename.
        output_csv = output_dir / input_csv.name

        # Build a ligand -> interactions map specifically for this isoform.
        per_ligand_map = {
            ligand_id: interaction_lookup.get((ligand_id, isoform), {})
            for ligand_id in ligand_ids_needed
        }

        annotate_result_csv(
            input_csv=input_csv,
            output_csv=output_csv,
            isoform=isoform,
            ligand_to_interactions=per_ligand_map,
            unresolved_log=unresolved_log,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = output_dir / "complex_mapping.csv"
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False)

    failures_path = output_dir / "plip_failures.csv"
    if plip_failures:
        pd.DataFrame(plip_failures).to_csv(failures_path, index=False)
    else:
        pd.DataFrame(
            columns=["ligand_id", "isoform", "complex_file", "error"]
        ).to_csv(failures_path, index=False)

    unresolved_path = output_dir / "unresolved_binary_interactions.csv"
    if unresolved_log:
        pd.DataFrame(unresolved_log).to_csv(unresolved_path, index=False)
    else:
        pd.DataFrame(
            columns=["file", "ligand_id", "isoform", "residue", "reason"]
        ).to_csv(unresolved_path, index=False)

    print("\nDone.")
    print(f"Annotated CSVs:                 {output_dir}")
    print(f"Complex mapping:                {mapping_path}")
    print(f"PLIP failures:                  {failures_path}")
    print(f"Unresolved binary interactions: {unresolved_path}")

    # A few useful summary counts.
    duplicate_selected = sum(
        1
        for row in mapping_rows
        if row["status"] == "duplicate_exact_matches_first_selected"
    )
    no_match = sum(
        1
        for row in mapping_rows
        if row["status"] not in {
            "unique_exact_match",
            "duplicate_exact_matches_first_selected",
        }
    )

    print(f"\nDuplicate exact-match groups (first selected): {duplicate_selected:,}")
    print(f"Unresolved ligand/isoform mappings:             {no_match:,}")
    print(f"PLIP failures:                                  {len(plip_failures):,}")
    print(f"Unresolved binary 1 cells:                      {len(unresolved_log):,}")


if __name__ == "__main__":
    main()