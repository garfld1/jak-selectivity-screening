#!/usr/bin/env python3
"""
Run LocalColabFold for JAK1 and JAK2 using the crystal structures already in
this repository as custom templates.

Jobs
----
JAK1_full   full-length JAK1, template 4EI4 (Crystal Structures/JAK1_4EI4.cif)
JAK1_jh1    JH1 domain, same 4EI4 template
JAK2_full   full-length JAK2, template 4JI9 (Crystal Structures/JAK2_4JI9.cif)
JAK2_jh1    JH1 domain, same 4JI9 template

4EI4 and 4JI9 are kinase-domain crystals, not full-length structures. For the
full-length jobs the crystal constrains the JH1 region and the rest of the
chain is predicted from the MSA. For the JH1 jobs the query is the domain
sequence and the crystal covers that whole query.

The query sequence is written unchanged. Crystal residues are used to place
the template and to report mismatches. They are not copied into the FASTA.

4EI4 differs from full-length JAK1 at the first crystallized residue
(Pro853Gly). That proline is kept in JAK1_full. It lies outside the JH1
query, so JAK1_jh1 has no sequence mismatch. 4JI9 matches full-length JAK2
exactly across the crystallized span. Both crystals include a short stretch
outside the JH1 boundaries used here, and both contain phosphotyrosine in the
activation loop. Phosphotyrosine is compared as tyrosine.

Usage
-----
python src/structure/jak_colabfold_templates.py --dry-run
python src/structure/jak_colabfold_templates.py --protein JAK1 --construct jh1
python src/structure/jak_colabfold_templates.py --overwrite
"""

from __future__ import annotations

import argparse
import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
    "MSE": "M",
    "PTR": "Y",  # phosphotyrosine, present in 4EI4 and 4JI9
    "SEP": "S",
    "TPO": "T",
    "TYS": "Y",
    "UNK": "X",
    "ASX": "B",
    "GLX": "Z",
}

PARENT_OF_MODIFIED = {
    "MSE": "M",
    "PTR": "Y",
    "SEP": "S",
    "TPO": "T",
    "TYS": "Y",
}


def _seq(text: str) -> str:
    return "".join(text.split())


# Full-length sequences match UniProt P23458 (JAK1) and O60674 (JAK2).
# JH1 spans are the sequences supplied for these models:
#   JAK1 875-1153, JAK2 849-1124 (1-based, inclusive).
JAK1_FULL = _seq("""
MQYLNIKEDCNAMAFCAKMRSSKKTEVNLEAPEPGVEVIFYLSDREPLRLGSGEYTAEELCIRAAQACRI
SPLCHNLFALYDENTKLWYAPNRTITVDDKMSLRLHYRMRFYFTNWHGTNDNEQSVWRHSPKKQKNGYEK
KKIPDATPLLDASSLEYLFAQGQYDLVKCLAPIRDPKTEQDGHDIENECLGMAVLAISHYAMMKKMQLPE
LPKDISYKRYIPETLNKSIRQRNLLTRMRINNVFKDFLKEFNNKTICDSSVSTHDLKVKYLATLETLTKH
YGAEIFETSMLLISSENEMNWFHSNDGGNVLYYEVMVTGNLGIQWRHKPNVVSVEKEKNKLKRKKLENKH
KKDEEKNKIREEWNNFSYFPEITHIVIKESVVSINKQDNKKMELKLSSHEEALSFVSLVDGYFRLTADAH
HYLCTDVAPPLIVHNIQNGCHGPICTEYAINKLRQEGSEEGMYVLRWSCTDFDNILMTVTCFEKSEQVQG
AQKQFKNFQIEVQKGRYSLHGSDRSFPSLGDLMSHLKKQILRTDNISFMLKRCCQPKPREISNLLVATKK
AQEWQPVYPMSQLSFDRILKKDLVQGEHLGRGTRTHIYSGTLMDYKDDEGTSEEKKIKVILKVLDPSHRD
ISLAFFEAASMMRQVSHKHIVYLYGVCVRDVENIMVEEFVEGGPLDLFMHRKSDVLTTPWKFKVAKQLAS
ALSYLEDKDLVHGNVCTKNLLLAREGIDSECGPFIKLSDPGIPITVLSRQECIERIPWIAPECVEDSKNL
SVAADKWSFGTTLWEICYNGEIPLKDKTLIEKERFYESRCRPVTPSCKELADLMTRCMNYDPNQRPFFRA
IMRDINKLEEQNPDIVSEKKPATEVDPTHFEKRFLKRIRDLGEGHFGKVELCRYDPEGDNTGEQVAVKSL
KPESGGNHIADLKKEIEILRNLYHENIVKYKGICTEDGGNGIKLIMEFLPSGSLKEYLPKNKNKINLKQQ
LKYAVQICKGMDYLGSRQYVHRDLAARNVLVESEHQVKIGDFGLTKAIETDKEYYTVKDDRDSPVFWYAP
ECLMQSKFYIASDVWSFGVTLHELLTYCDSDSSPMALFLKMIGPTHGQMTVTRLVNTLKEGKRLPCPPNC
PDEVYQLMRKCWEFQPSNRTSFQNLIEGFEALLK
""")

JAK1_JH1 = _seq("""
LKRIRDLGEGHFGKVELCRYDPEGDNTGEQVAVKSLKPESGGNHIADLKKEIEILRNLYHENIVKYKGIC
TEDGGNGIKLIMEFLPSGSLKEYLPKNKNKINLKQQLKYAVQICKGMDYLGSRQYVHRDLAARNVLVESE
HQVKIGDFGLTKAIETDKEYYTVKDDRDSPVFWYAPECLMQSKFYIASDVWSFGVTLHELLTYCDSDSSP
MALFLKMIGPTHGQMTVTRLVNTLKEGKRLPCPPNCPDEVYQLMRKCWEFQPSNRTSFQNLIEGFEALL
""")

JAK2_FULL = _seq("""
MGMACLTMTEMEGTSTSSIYQNGDISGNANSMKQIDPVLQVYLYHSLGKSEADYLTFPSGEYVAEEICIA
ASKACGITPVYHNMFALMSETERIWYPPNHVFHIDESTRHNVLYRIRFYFPRWYCSGSNRAYRHGISRGA
EAPLLDDFVMSYLFAQWRHDFVHGWIKVPVTHETQEECLGMAVLDMMRIAKENDQTPLAIYNSISYKTFL
PKCIRAKIQDYHILTRKRIRYRFRRFIQQFSQCKATARNLKLKYLINLETLQSAFYTEKFEVKEPGSGPS
GEEIFATIIITGNGGIQWSRGKHKESETLTEQDLQLYCDFPNIIDVSIKQANQEGSNESRVVTIHKQDGK
NLEIELSSLREALSFVSLIDGYYRLTADAHHYLCKEVAPPAVLENIQSNCHGPISMDFAISKLKKAGNQT
GLYVLRCSPKDFNKYFLTFAVERENVIEYKHCLITKNENEEYNLSGTKKNFSSLKDLLNCYQMETVRSDN
IIFQFTKCCPPKPKDKSNLLVFRTNGVSDVPTSPTLQRPTHMNQMVFHKIRNEDLIFNESLGQGTFTKIF
KGVRREVGDYGQLHETEVLLKVLDKAHRNYSESFFEAASMMSKLSHKHLVLNYGVCVCGDENILVQEFVK
FGSLDTYLKKNKNCINILWKLEVAKQLAWAMHFLEENTLIHGNVCAKNILLIREEDRKTGNPPFIKLSDP
GISITVLPKDILQERIPWVPPECIENPKNLNLATDKWSFGTTLWEICSGGDKPLSALDSQRKLQFYEDRH
QLPAPKWAELANLINNCMDYEPDFRPSFRAIIRDLNSLFTPDYELLTENDMLPNMRIGALGFSGAFEDRD
PTQFEERHLKFLQQLGKGNFGSVEMCRYDPLQDNTGEVVAVKKLQHSTEEHLRDFEREIEILKSLQHDNI
VKYKGVCYSAGRRNLKLIMEYLPYGSLRDYLQKHKERIDHIKLLQYTSQICKGMEYLGTKRYIHRDLATR
NILVENENRVKIGDFGLTKVLPQDKEYYKVKEPGESPIFWYAPESLTESKFSVASDVWSFGVVLYELFTY
IEKSKSPPAEFMRMIGNDKQGQMIVFHLIELLKNNGRLPRPDGCPDEIYMIMTECWNNNVNQRPSFRDLA
LRVDQIRDNMAG
""")

JAK2_JH1 = _seq("""
LKFLQQLGKGNFGSVEMCRYDPLQDNTGEVVAVKKLQHSTEEHLRDFEREIEILKSLQHDNIVKYKGVCY
SAGRRNLKLIMEYLPYGSLRDYLQKHKERIDHIKLLQYTSQICKGMEYLGTKRYIHRDLATRNILVENEN
RVKIGDFGLTKVLPQDKEYYKVKEPGESPIFWYAPESLTESKFSVASDVWSFGVVLYELFTYIEKSKSPP
AEFMRMIGNDKQGQMIVFHLIELLKNNGRLPRPDGCPDEIYMIMTECWNNNVNQRPSFRDLALRVD
""")

SEQUENCES = {
    "JAK1": {"full": JAK1_FULL, "jh1": JAK1_JH1},
    "JAK2": {"full": JAK2_FULL, "jh1": JAK2_JH1},
}

TEMPLATES = {
    "JAK1": ("4EI4", REPO_ROOT / "Crystal Structures" / "JAK1_4EI4.cif"),
    "JAK2": ("4JI9", REPO_ROOT / "Crystal Structures" / "JAK2_4JI9.cif"),
}


def validate_builtin_sequences() -> None:
    for protein, seqs in SEQUENCES.items():
        for name, seq in seqs.items():
            if not re.fullmatch(r"[A-Z]+", seq):
                raise ValueError(f"{protein} {name} sequence is empty or contains non-letters")
        if seqs["jh1"] not in seqs["full"]:
            raise ValueError(f"{protein} JH1 sequence is not contained in the full-length sequence")


def find_colabfold_batch() -> str | None:
    """Find LocalColabFold's colabfold_batch, or return None."""
    on_path = shutil.which("colabfold_batch")
    if on_path:
        return on_path

    candidates = [
        Path.home() / "localcolabfold" / ".pixi" / "envs" / "default" / "bin" / "colabfold_batch",
        Path.home() / "localcolabfold" / "colabfold-conda" / "bin" / "colabfold_batch",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def tokenize_cif_line(line: str) -> list[str]:
    """Tokenize a simple mmCIF row, including quoted strings."""
    tokens: list[str] = []
    current: list[str] = []
    quote = None

    for ch in line:
        if quote:
            if ch == quote:
                tokens.append("".join(current))
                current = []
                quote = None
            else:
                current.append(ch)
        elif ch in ("'", '"'):
            quote = ch
        elif ch.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(ch)

    if current:
        tokens.append("".join(current))
    return tokens


def extract_template_sequences_from_entity_poly_seq(
    cif_path: Path,
) -> tuple[dict[str, str], list[dict]]:
    """
    Read protein sequences from the _entity_poly_seq loop.

    Modified residues in PARENT_OF_MODIFIED are translated to their parent
    amino acid so they can be compared with the query sequence.
    """
    lines = cif_path.read_text(errors="ignore").splitlines()
    sequences: dict[str, list[str]] = {}
    modifications: list[dict] = []
    i = 0

    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue

        i += 1
        tags: list[str] = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            tags.append(lines[i].strip())
            i += 1

        if "_entity_poly_seq.entity_id" not in tags or "_entity_poly_seq.mon_id" not in tags:
            continue

        entity_idx = tags.index("_entity_poly_seq.entity_id")
        mon_idx = tags.index("_entity_poly_seq.mon_id")

        while i < len(lines):
            row = lines[i].strip()
            if not row:
                i += 1
                continue
            if row.startswith("#") or row == "loop_" or row.startswith("_"):
                break

            tokens = tokenize_cif_line(row)
            if len(tokens) > max(entity_idx, mon_idx):
                entity_id = tokens[entity_idx]
                mon_id = tokens[mon_idx].upper()
                one_letter = THREE_TO_ONE.get(mon_id, "X")
                residues = sequences.setdefault(entity_id, [])
                residues.append(one_letter)
                if mon_id in PARENT_OF_MODIFIED:
                    modifications.append({
                        "entity_id": entity_id,
                        "template_position_1based": len(residues),
                        "mon_id": mon_id,
                        "compared_as": one_letter,
                    })
            i += 1

        joined = {entity: "".join(residues) for entity, residues in sequences.items()}
        return joined, modifications

    return {}, []


def extract_template_sequences_from_entity_poly(cif_path: Path) -> list[str]:
    """Fallback reader for _entity_poly.pdbx_seq_one_letter_code_can."""
    text = cif_path.read_text(errors="ignore")
    pattern = re.compile(
        r"_entity_poly\.pdbx_seq_one_letter_code_can\s*\n;(.*?)\n;",
        re.DOTALL,
    )
    sequences = []
    for match in pattern.finditer(text):
        seq = re.sub(r"[^A-Za-z]", "", match.group(1)).upper()
        if seq:
            sequences.append(seq)
    return sequences


def get_template_sequence(cif_path: Path) -> tuple[str, str, list[dict]]:
    """
    Return (label, sequence, modifications) for the longest protein entity.

    Preference is the _entity_poly_seq loop, then the canonical one-letter
    sequence. The longest sequence is kept when a file has several entities.
    """
    entity_seqs, modifications = extract_template_sequences_from_entity_poly_seq(cif_path)
    candidates: list[tuple[str, str]] = []

    for entity_id, seq in entity_seqs.items():
        clean = re.sub(r"[^A-Z]", "", seq.upper())
        if clean:
            candidates.append((f"entity_{entity_id}", clean))

    if not candidates:
        for idx, seq in enumerate(extract_template_sequences_from_entity_poly(cif_path), start=1):
            candidates.append((f"entity_poly_fallback_{idx}", seq))
        modifications = []

    if not candidates:
        raise ValueError(f"Could not extract a protein sequence from {cif_path}")

    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    label, sequence = candidates[0]
    entity_id = label.removeprefix("entity_")
    kept = [item for item in modifications if item["entity_id"] == entity_id]
    return label, sequence, kept


def find_best_ungapped_overlap(seq_a: str, seq_b: str) -> dict:
    """
    Place the shorter sequence inside the longer one with no gaps.

    Returns the overlap with the most exact matches. X on either side is
    ignored in the identity. The first placement wins when match counts tie.
    """
    seq_a = seq_a.upper()
    seq_b = seq_b.upper()
    if not seq_a or not seq_b:
        raise ValueError("Cannot align an empty sequence")

    if len(seq_b) <= len(seq_a):
        windows = [(start, 0, len(seq_b)) for start in range(len(seq_a) - len(seq_b) + 1)]
    else:
        windows = [(0, start, len(seq_a)) for start in range(len(seq_b) - len(seq_a) + 1)]

    best = None
    for a_start, b_start, overlap in windows:
        a_slice = seq_a[a_start:a_start + overlap]
        b_slice = seq_b[b_start:b_start + overlap]
        matches = mismatches = comparable = 0
        for left, right in zip(a_slice, b_slice):
            if left == "X" or right == "X":
                continue
            comparable += 1
            if left == right:
                matches += 1
            else:
                mismatches += 1
        result = {
            "a_start": a_start,
            "a_end": a_start + overlap,
            "b_start": b_start,
            "b_end": b_start + overlap,
            "overlap": overlap,
            "matches": matches,
            "mismatches": mismatches,
            "comparable": comparable,
            "identity": (matches / comparable) if comparable else 0.0,
        }
        if best is None or result["matches"] > best["matches"]:
            best = result

    if best is None:
        raise ValueError("No ungapped overlap could be scored")
    return best


def _mismatch_fraction(placement: dict) -> float:
    if not placement["comparable"]:
        return 1.0
    return placement["mismatches"] / placement["comparable"]


def _collect_mismatches(seq_a: str, seq_b: str, placement: dict) -> list[dict]:
    """List ungapped mismatches. seq_a is the full-length protein; seq_b is the template."""
    mismatches = []
    for offset in range(placement["overlap"]):
        full_pos = placement["a_start"] + offset
        template_pos = placement["b_start"] + offset
        full_res = seq_a[full_pos]
        template_res = seq_b[template_pos]
        if full_res == "X" or template_res == "X" or full_res == template_res:
            continue
        context_start = max(0, full_pos - 5)
        context_end = min(len(seq_a), full_pos + 6)
        mismatches.append({
            "full_position_1based": full_pos + 1,
            "template_position_1based": template_pos + 1,
            "full_residue": full_res,
            "template_residue": template_res,
            "full_context": seq_a[context_start:context_end],
            "inside_query": False,
        })
    return mismatches


def assess_job(
    job: dict,
    min_identity: float,
    max_mismatch_fraction: float,
) -> dict:
    """
    Place the crystal sequence on the full-length protein and score the part
    that overlaps the sequence being folded.
    """
    template_path = Path(job["template_path"])
    if not template_path.is_file():
        raise ValueError(f"Template CIF does not exist: {template_path}")

    label, template_seq, modifications = get_template_sequence(template_path)
    full_seq = SEQUENCES[job["protein"]]["full"]
    query_seq = job["sequence"]
    jh1_seq = SEQUENCES[job["protein"]]["jh1"]
    jh1_start = full_seq.find(jh1_seq)
    if jh1_start < 0:
        raise ValueError(f"{job['protein']} JH1 sequence was not found in the full-length sequence")

    placement = find_best_ungapped_overlap(full_seq, template_seq)
    mismatch_fraction = _mismatch_fraction(placement)
    if placement["identity"] < min_identity:
        raise ValueError(
            f"{job['name']}: template/full-length identity is {placement['identity']:.3f}, "
            f"below {min_identity:.3f}. The crystal may be the wrong chain, or an "
            f"insertion/deletion may prevent this ungapped alignment."
        )
    if mismatch_fraction > max_mismatch_fraction:
        raise ValueError(
            f"{job['name']}: template mismatch fraction is {mismatch_fraction:.3f}, "
            f"above {max_mismatch_fraction:.3f}."
        )

    if job["construct"] == "full":
        query_full_start = 0
        query_full_end = len(full_seq)
    elif job["construct"] == "jh1":
        query_full_start = jh1_start
        query_full_end = jh1_start + len(jh1_seq)
    else:
        raise ValueError(f"Unknown construct: {job['construct']}")

    template_full_start = placement["a_start"]
    template_full_end = placement["a_end"]
    overlap_start = max(query_full_start, template_full_start)
    overlap_end = min(query_full_end, template_full_end)
    if overlap_end <= overlap_start:
        raise ValueError(
            f"{job['name']}: crystal template does not overlap the {job['construct']} query"
        )

    overlap_len = overlap_end - overlap_start
    query_coverage = overlap_len / len(query_seq)
    if job["construct"] == "jh1" and query_coverage < 0.90:
        raise ValueError(
            f"{job['name']}: crystal covers only {query_coverage:.1%} of the JH1 query"
        )

    mismatches = _collect_mismatches(full_seq, template_seq, placement)
    for item in mismatches:
        pos0 = item["full_position_1based"] - 1
        item["inside_query"] = query_full_start <= pos0 < query_full_end
        item["query_position_1based"] = (pos0 - query_full_start + 1) if item["inside_query"] else None

    in_query = [item for item in mismatches if item["inside_query"]]
    comparable_in_query = 0
    for offset in range(overlap_len):
        full_res = full_seq[overlap_start + offset]
        template_index = (overlap_start - template_full_start) + placement["b_start"] + offset
        template_res = template_seq[template_index]
        if full_res == "X" or template_res == "X":
            continue
        comparable_in_query += 1
    query_identity = (
        (comparable_in_query - len(in_query)) / comparable_in_query
        if comparable_in_query else 0.0
    )
    query_mismatch_fraction = (len(in_query) / comparable_in_query) if comparable_in_query else 1.0
    if query_identity < min_identity or query_mismatch_fraction > max_mismatch_fraction:
        raise ValueError(
            f"{job['name']}: identity inside the folded query is {query_identity:.3f} "
            f"with mismatch fraction {query_mismatch_fraction:.3f}."
        )

    return {
        "template_label": label,
        "template_length": len(template_seq),
        "template_seq": template_seq,
        "modifications": modifications,
        "full_length": len(full_seq),
        "query_length": len(query_seq),
        "jh1_start_1based": jh1_start + 1,
        "jh1_end_1based": jh1_start + len(jh1_seq),
        "query_full_start_1based": query_full_start + 1,
        "query_full_end_1based": query_full_end,
        "template_full_start_1based": template_full_start + 1,
        "template_full_end_1based": template_full_end,
        "overlap_full_start_1based": overlap_start + 1,
        "overlap_full_end_1based": overlap_end,
        "query_coverage": query_coverage,
        "placement": placement,
        "mismatch_fraction": mismatch_fraction,
        "query_identity": query_identity,
        "query_mismatch_fraction": query_mismatch_fraction,
        "mismatches": mismatches,
        "mismatches_in_query": in_query,
        "residues_changed": 0,
    }


def write_fasta(header: str, sequence: str, output_path: Path, width: int = 80) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        handle.write(f">{header}\n")
        for start in range(0, len(sequence), width):
            handle.write(sequence[start:start + width] + "\n")


def write_report(job: dict, info: dict, fasta_path: Path, template_copy: Path, report_path: Path) -> None:
    placement = info["placement"]
    in_query = info["mismatches_in_query"]
    flank = [item for item in info["mismatches"] if not item["inside_query"]]

    with report_path.open("w") as report:
        report.write("JAK ColabFold template report\n")
        report.write("=" * 80 + "\n")
        report.write("The folded sequence is the supplied query, written unchanged into the FASTA.\n")
        report.write("The crystal is passed to ColabFold as a custom template for the overlapping region.\n")
        report.write("\n")
        report.write(f"Job: {job['name']}\n")
        report.write(f"Protein: {job['protein']}\n")
        report.write(f"Construct: {job['construct']}\n")
        report.write(f"Template PDB: {job['pdb_id']}\n")
        report.write(f"Template CIF: {job['template_path']}\n")
        report.write(f"Template copied to: {template_copy}\n")
        report.write(f"Template entity selected: {info['template_label']}\n")
        report.write(f"Query FASTA: {fasta_path}\n")
        report.write(f"Query header: {job['name']}\n")
        report.write(f"Full-length sequence: {info['full_length']} aa\n")
        report.write(
            f"JH1 definition on the full-length sequence: "
            f"{info['jh1_start_1based']}-{info['jh1_end_1based']}\n"
        )
        report.write(
            f"Folded query on the full-length sequence: "
            f"{info['query_full_start_1based']}-{info['query_full_end_1based']} "
            f"({info['query_length']} aa)\n"
        )
        report.write(f"Template sequence length: {info['template_length']} aa\n")
        report.write(
            f"Template placement on the full-length sequence: "
            f"{info['template_full_start_1based']}-{info['template_full_end_1based']}\n"
        )
        report.write(
            f"Overlap used for this query: "
            f"{info['overlap_full_start_1based']}-{info['overlap_full_end_1based']}\n"
        )
        report.write(f"Fraction of the query covered by the crystal: {info['query_coverage']:.4f}\n")
        report.write(f"Comparable positions in the full-length placement: {placement['comparable']}\n")
        report.write(f"Matches in the full-length placement: {placement['matches']}\n")
        report.write(f"Mismatches in the full-length placement: {placement['mismatches']}\n")
        report.write(f"Identity of the full-length placement: {placement['identity']:.4f}\n")
        report.write(f"Mismatch fraction of the full-length placement: {info['mismatch_fraction']:.4f}\n")
        report.write(f"Identity inside the folded query: {info['query_identity']:.4f}\n")
        report.write(f"Mismatches inside the folded query: {len(in_query)}\n")
        report.write(f"Residues changed in the query FASTA: {info['residues_changed']}\n")
        report.write("\n")

        if info["modifications"]:
            report.write("Nonstandard crystal residues compared as their parent amino acid\n")
            report.write("-" * 80 + "\n")
            report.write("template_pos_1based\tmon_id\tcompared_as\tfull_pos_1based\n")
            origin = info["template_full_start_1based"]
            for item in info["modifications"]:
                full_pos = origin + item["template_position_1based"] - 1
                report.write(
                    f"{item['template_position_1based']}\t{item['mon_id']}\t"
                    f"{item['compared_as']}\t{full_pos}\n"
                )
            report.write(
                "\nPhosphotyrosine is scored as tyrosine. ColabFold's template reader uses "
                "standard amino acids, so the activation-loop phosphotyrosines can be left "
                "out of the template coordinates. The query still has tyrosine there.\n\n"
            )

        if in_query:
            report.write("Mismatches inside the folded query; the query residue is kept\n")
            report.write("-" * 80 + "\n")
            report.write(
                "full_pos_1based\tquery_pos_1based\ttemplate_pos_1based\t"
                "query_residue\ttemplate_residue\tcontext\n"
            )
            for item in in_query:
                report.write(
                    f"{item['full_position_1based']}\t{item['query_position_1based']}\t"
                    f"{item['template_position_1based']}\t{item['full_residue']}\t"
                    f"{item['template_residue']}\t{item['full_context']}\n"
                )
            report.write("\n")
        else:
            report.write("No mismatches inside the folded query.\n\n")

        if flank:
            report.write("Mismatches in crystal positions outside this query\n")
            report.write("-" * 80 + "\n")
            report.write("full_pos_1based\ttemplate_pos_1based\tfull_residue\ttemplate_residue\tcontext\n")
            for item in flank:
                report.write(
                    f"{item['full_position_1based']}\t{item['template_position_1based']}\t"
                    f"{item['full_residue']}\t{item['template_residue']}\t{item['full_context']}\n"
                )
            report.write("\n")


def prepare_template_dir(
    template_file: Path,
    template_dir: Path,
    pdb_id: str | None = None,
) -> Path:
    """Replace custom_template/ and copy the crystal CIF as <pdbid>.cif.

    ColabFold's template reader looks up files by 4-character PDB ID, so
    JAK1_4EI4.cif must be stored as 4ei4.cif, not jak1_4ei4.cif.
    """
    if template_dir.exists():
        shutil.rmtree(template_dir)
    template_dir.mkdir(parents=True, exist_ok=True)
    stem = (pdb_id or template_file.stem).lower()
    match = re.search(r"([0-9][a-z0-9]{3})", stem)
    pdb_stem = match.group(1) if match else stem
    template_copy = template_dir / f"{pdb_stem}.cif"
    shutil.copy2(template_file, template_copy)
    return template_copy


def prepare_job_inputs(job: dict, job_dir: Path, info: dict, template_dir_name: str) -> dict:
    """Write the query FASTA, the alignment report, and the custom-template folder."""
    job_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = job_dir / f"{job['name']}.fasta"
    report_path = job_dir / "template_alignment_report.txt"
    write_fasta(job["name"], job["sequence"], fasta_path)
    template_copy = prepare_template_dir(
        Path(job["template_path"]),
        job_dir / template_dir_name,
        job.get("pdb_id"),
    )
    write_report(job, info, fasta_path, template_copy, report_path)
    return {
        "fasta": fasta_path,
        "report": report_path,
        "template_copy": template_copy,
        "template_dir": job_dir / template_dir_name,
    }


def build_colabfold_command(
    colabfold_bin: str,
    fasta_path: Path,
    output_dir: Path,
    template_dir: Path,
    num_recycle: int,
    num_models: int,
    amber: bool,
    use_gpu_relax: bool,
) -> list[str]:
    cmd = [
        colabfold_bin,
        "--templates",
        "--custom-template-path",
        str(template_dir),
        "--num-recycle",
        str(num_recycle),
        "--num-models",
        str(num_models),
    ]
    if amber:
        cmd.append("--amber")
    if amber and use_gpu_relax:
        cmd.append("--use-gpu-relax")
    cmd.extend([str(fasta_path), str(output_dir)])
    return cmd


def run_command(cmd: list[str], log_file: Path, dry_run: bool = False) -> int:
    command_text = " ".join(str(part) for part in cmd)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as log:
        log.write("=" * 80 + "\n")
        log.write(f"Started: {datetime.datetime.now()}\n")
        log.write(f"Command:\n{command_text}\n\n")
        if dry_run:
            log.write("DRY RUN: command not executed.\n")
            return 0

        process = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"\nFinished: {datetime.datetime.now()}\n")
        log.write(f"Return code: {process.returncode}\n")
        return process.returncode


def output_has_ranked_models(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    patterns = ["*rank*.pdb", "*rank*.cif", "*rank*.json", "ranking_debug.json"]
    return any(list(output_dir.rglob(pattern)) for pattern in patterns)


def log_has_obvious_error(log_file: Path) -> bool:
    """Catch ColabFold tracebacks that still exit with return code 0."""
    if not log_file.exists():
        return False
    error_terms = [
        "Traceback",
        "FileNotFoundError",
        "Could not get MSA/templates",
        "No such file or directory",
        "OSError",
        "ERROR",
    ]
    text = log_file.read_text(errors="ignore")
    return any(term in text for term in error_terms)


def jobs_to_run(proteins: list[str], constructs: list[str]) -> list[dict]:
    jobs = []
    for protein in proteins:
        pdb_id, template_path = TEMPLATES[protein]
        for construct in constructs:
            jobs.append({
                "name": f"{protein}_{construct}",
                "protein": protein,
                "construct": construct,
                "sequence": SEQUENCES[protein][construct],
                "pdb_id": pdb_id,
                "template_path": template_path,
            })
    return jobs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fold JAK1 and JAK2 full-length and JH1 sequences with LocalColabFold, "
            "using 4EI4 and 4JI9 as custom templates."
        )
    )
    parser.add_argument(
        "--protein",
        nargs="+",
        choices=["JAK1", "JAK2"],
        default=["JAK1", "JAK2"],
        help="Proteins to fold. Default: JAK1 JAK2",
    )
    parser.add_argument(
        "--construct",
        nargs="+",
        choices=["full", "jh1"],
        default=["full", "jh1"],
        help="Constructs to fold. Default: full jh1",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "results" / "colabfold_jak"),
        help="Directory where one folder is created per job. Default: results/colabfold_jak",
    )
    parser.add_argument(
        "--colabfold-bin",
        default=None,
        help="Path to colabfold_batch. Default: auto-detect LocalColabFold",
    )
    parser.add_argument(
        "--template-dir-name",
        default="custom_template",
        help="Custom-template folder created inside each job folder.",
    )
    parser.add_argument(
        "--min-template-identity",
        type=float,
        default=0.90,
        help="Minimum identity between the crystal sequence and the full-length protein. Default: 0.90",
    )
    parser.add_argument(
        "--max-mismatch-fraction",
        type=float,
        default=0.05,
        help="Maximum mismatch fraction. Default: 0.05",
    )
    parser.add_argument("--num-recycle", type=int, default=12, help="ColabFold recycles. Default: 12")
    parser.add_argument(
        "--num-models",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="Number of models. Default: 5",
    )
    parser.add_argument("--amber", action="store_true", help="Enable AMBER relaxation.")
    parser.add_argument(
        "--use-gpu-relax",
        action="store_true",
        help="Relax on GPU. Used only together with --amber.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing ColabFold output folders.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write FASTA files, reports, and template folders, and log the ColabFold command without running it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_builtin_sequences()

    if args.use_gpu_relax and not args.amber:
        print("WARNING: --use-gpu-relax has no effect without --amber.")

    colabfold_bin = args.colabfold_bin or find_colabfold_batch()
    if not args.dry_run and not colabfold_bin:
        print(
            "ERROR: colabfold_batch was not found.\n"
            "Install LocalColabFold, or pass --colabfold-bin /path/to/colabfold_batch.\n"
            "To write the FASTA files, alignment reports, and template folders without folding:\n"
            "  python src/structure/jak_colabfold_templates.py --dry-run",
            file=sys.stderr,
        )
        return 1
    if colabfold_bin is None:
        colabfold_bin = "colabfold_batch"

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = jobs_to_run(args.protein, args.construct)

    print(f"Using colabfold_batch: {colabfold_bin}")
    print(f"Output root: {output_root}")
    print(f"Jobs: {', '.join(job['name'] for job in jobs)}")
    print("The query FASTA keeps the supplied sequence. Crystal residues are templates only.")
    print()

    failed: list[str] = []
    skipped: list[str] = []
    completed: list[str] = []

    for job in jobs:
        print(f"\n--- {job['name']} ({job['pdb_id']}) ---")
        job_dir = output_root / job["name"]
        output_dir = job_dir / "colabfold_output"
        log_file = job_dir / "localcolabfold_run.log"

        try:
            info = assess_job(job, args.min_template_identity, args.max_mismatch_fraction)
        except ValueError as exc:
            print(f"FAILED: {job['name']}: {exc}")
            failed.append(job["name"])
            continue

        if output_dir.exists():
            if args.overwrite:
                print(f"Removing existing output folder: {output_dir}")
                shutil.rmtree(output_dir)
            else:
                print(f"SKIP: output already exists: {output_dir}")
                print("Use --overwrite to rerun this job.")
                skipped.append(job["name"])
                continue

        written = prepare_job_inputs(job, job_dir, info, args.template_dir_name)
        cmd = build_colabfold_command(
            colabfold_bin=colabfold_bin,
            fasta_path=written["fasta"],
            output_dir=output_dir,
            template_dir=written["template_dir"],
            num_recycle=args.num_recycle,
            num_models=args.num_models,
            amber=args.amber,
            use_gpu_relax=args.use_gpu_relax,
        )

        print(f"Query:     {written['fasta'].name} ({info['query_length']} aa)")
        print(
            f"Query span on full-length sequence: "
            f"{info['query_full_start_1based']}-{info['query_full_end_1based']}"
        )
        print(
            f"Template:  {job['pdb_id']} covers full-length "
            f"{info['template_full_start_1based']}-{info['template_full_end_1based']} "
            f"(identity {info['placement']['identity']:.4f})"
        )
        print(f"Query coverage: {info['query_coverage']:.4f}")
        print(f"Mismatches inside query: {len(info['mismatches_in_query'])}")
        print(f"Residues changed: {info['residues_changed']}")
        print(f"Report:    {written['report']}")
        print(f"Output:    {output_dir}")

        rc = run_command(cmd, log_file, dry_run=args.dry_run)
        if args.dry_run:
            print(f"DRY RUN DONE: {job['name']}")
            completed.append(job["name"])
            continue

        produced_models = output_has_ranked_models(output_dir)
        obvious_log_error = log_has_obvious_error(log_file)
        if rc == 0 and produced_models and not obvious_log_error:
            print(f"DONE: {job['name']}")
            completed.append(job["name"])
        elif rc == 0 and produced_models and obvious_log_error:
            print(f"WARNING: {job['name']} produced ranked files, but the log contains an error.")
            print(f"Check log manually: {log_file}")
            failed.append(job["name"])
        elif rc == 0 and not produced_models:
            print(f"FAILED: {job['name']}")
            print("ColabFold exited with return code 0, but no ranked model files were found.")
            print(f"Check log: {log_file}")
            failed.append(job["name"])
        else:
            print(f"FAILED: {job['name']}. See log: {log_file}")
            failed.append(job["name"])

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for label, names in ("Completed", completed), ("Skipped", skipped), ("Failed", failed):
        print(f"{label}: {len(names)}")
        for name in names:
            print(f"  - {name}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
