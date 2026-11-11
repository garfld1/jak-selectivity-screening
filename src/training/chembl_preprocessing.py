"""
JAK1 / JAK2 selectivity analysis from ChEMBL IC50 exports.

Usage (from a JupyterLab terminal, in the folder containing this file):

    python jak_selectivity_analysis.py \
        --upload-dir ./data \
        --work-dir ./output \
        --jak1-file chembl_jak1.tsv \
        --jak2-file chembl_jak2.tsv

All arguments are optional; defaults are shown above. Run with -h for help:

    python jak_selectivity_analysis.py -h
"""

import argparse
import csv
import math
import os
from collections import defaultdict

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def canon_smiles(smi):
    if smi is None:
        return None
    smi = smi.strip()
    if not smi:
        return None
    if ' |' in smi:
        smi = smi.split(' |')[0].strip()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def unit_to_nM_factor(unit_str):
    """Return the multiplicative factor to convert a value in `unit_str` to nM.
    Handles plain units (pM, nM, uM, mM, M) and ChEMBL's '10^N unit' prefix notation
    (e.g. '10^2 uM' means the raw value should be multiplied by 100, then treated as uM).
    Returns None if the unit can't be parsed.
    """
    base_factors_to_nM = {
        'pM': 1e-3,
        'nM': 1.0,
        'uM': 1e3,
        'mM': 1e6,
        'M': 1e9,
    }
    unit_str = unit_str.strip()
    if unit_str in base_factors_to_nM:
        return base_factors_to_nM[unit_str]

    # Handle "10^N unit" prefix notation
    parts = unit_str.split()
    if len(parts) == 2 and parts[0].startswith('10^'):
        try:
            exponent = float(parts[0][3:])
        except ValueError:
            return None
        base_unit = parts[1]
        if base_unit not in base_factors_to_nM:
            return None
        return (10 ** exponent) * base_factors_to_nM[base_unit]

    return None


def load_chembl_ic50(upload_dir, fn, target_label):
    """Load IC50 rows (converted to nM) with assay description and canonical SMILES."""
    out = []
    n_invalid = 0
    n_unit_unparseable = 0
    n_converted = 0
    path = os.path.join(upload_dir, fn)
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('Standard Type') != 'IC50':
                continue
            value = row.get('Standard Value', '').strip()
            if value == '':
                continue
            units = row.get('Standard Units', '').strip()
            try:
                raw_val = float(value)
            except ValueError:
                continue
            if raw_val <= 0:
                continue
            factor = unit_to_nM_factor(units)
            if factor is None:
                n_unit_unparseable += 1
                continue
            ic50 = raw_val * factor
            if units != 'nM':
                n_converted += 1
            smi = row.get('Smiles', '')
            csmi = canon_smiles(smi)
            if csmi is None:
                n_invalid += 1
                continue
            assay_desc = row.get('Assay Description', '').strip()
            out.append({'smiles': csmi, 'assay_desc': assay_desc, 'ic50_nM': ic50, 'target': target_label})
    print(f'  {fn}: {len(out)} IC50 rows kept ({n_converted} unit-converted to nM), '
          f'{n_invalid} dropped (invalid SMILES), {n_unit_unparseable} dropped (unparseable units)')
    return out


def median_pic50_to_ic50(values_nM):
    """Convert IC50(nM) values -> pIC50, take median, convert back to IC50(nM)."""
    pic50s = [9.0 - math.log10(v) for v in values_nM]
    pic50s.sort()
    n = len(pic50s)
    if n % 2 == 1:
        med_pic50 = pic50s[n // 2]
    else:
        med_pic50 = (pic50s[n // 2 - 1] + pic50s[n // 2]) / 2.0
    return 10 ** (9.0 - med_pic50)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--upload-dir', default='./data',
                   help='Base folder containing input files (default: ./data)')
    p.add_argument('--work-dir', default='./data/ChEMBL',
                   help='Folder where output TSVs will be written (default: ./data/ChEMBL)')
    p.add_argument('--jak1-file', default='ChEMBL/chembl_jak1.tsv',
                   help='JAK1 ChEMBL TSV filename, relative to --upload-dir '
                        '(default: ChEMBL/chembl_jak1.tsv, i.e. ./data/ChEMBL/chembl_jak1.tsv)')
    p.add_argument('--jak2-file', default='ChEMBL/chembl_jak2.tsv',
                   help='JAK2 ChEMBL TSV filename, relative to --upload-dir '
                        '(default: ChEMBL/chembl_jak2.tsv, i.e. ./data/ChEMBL/chembl_jak2.tsv)')
    return p.parse_args()


def main():
    args = parse_args()
    upload_dir = args.upload_dir
    work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True)

    print(f'Loading {args.jak1_file} IC50 rows...')
    jak1_rows = load_chembl_ic50(upload_dir, args.jak1_file, 'JAK1')
    print(f'Loading {args.jak2_file} IC50 rows...')
    jak2_rows = load_chembl_ic50(upload_dir, args.jak2_file, 'JAK2')

    jak1_by_smiles = defaultdict(list)  # smiles -> list of (assay_desc, ic50)
    jak2_by_smiles = defaultdict(list)
    for r in jak1_rows:
        jak1_by_smiles[r['smiles']].append((r['assay_desc'], r['ic50_nM']))
    for r in jak2_rows:
        jak2_by_smiles[r['smiles']].append((r['assay_desc'], r['ic50_nM']))

    # Compounds with IC50 in both JAK1 and JAK2 (regardless of other assay types)
    both_targets = set(jak1_by_smiles.keys()) & set(jak2_by_smiles.keys())
    print(f'Compounds with IC50 defined in BOTH JAK1 and JAK2: {len(both_targets)}')

    # Match assay descriptions; compute pIC50-median IC50 per target
    qualifying_rows = []
    reference_rows = []
    for smi in sorted(both_targets):
        jak1_entries = jak1_by_smiles[smi]
        jak2_entries = jak2_by_smiles[smi]
        jak1_descs = {d for d, _ in jak1_entries}
        jak2_descs = {d for d, _ in jak2_entries}
        matched_descs = jak1_descs & jak2_descs
        unmatched_jak1 = jak1_descs - jak2_descs
        unmatched_jak2 = jak2_descs - jak1_descs

        qualifies = bool(matched_descs)

        reference_rows.append({
            'smiles': smi,
            'qualifies': 'YES' if qualifies else 'NO',
            'n_matched_assay_descriptions': len(matched_descs),
            'matched_assay_descriptions': ' ||| '.join(sorted(matched_descs)),
            'jak1_only_assay_descriptions': ' ||| '.join(sorted(unmatched_jak1)),
            'jak2_only_assay_descriptions': ' ||| '.join(sorted(unmatched_jak2)),
        })

        if not qualifies:
            continue

        jak1_vals = [v for d, v in jak1_entries if d in matched_descs]
        jak2_vals = [v for d, v in jak2_entries if d in matched_descs]
        if not jak1_vals or not jak2_vals:
            continue

        jak1_ic50 = median_pic50_to_ic50(jak1_vals)
        jak2_ic50 = median_pic50_to_ic50(jak2_vals)

        if jak1_ic50 <= 0 or jak2_ic50 <= 0:
            continue

        qualifying_rows.append({
            'smiles': smi,
            'jak1_ic50_nM': jak1_ic50,
            'jak2_ic50_nM': jak2_ic50,
            'matched_assay_descriptions': ' ||| '.join(sorted(matched_descs)),
            'n_matched_assay_descriptions': len(matched_descs),
            'n_jak1_values_used': len(jak1_vals),
            'n_jak2_values_used': len(jak2_vals),
            'fold_selective_JAK1_over_JAK2': jak2_ic50 / jak1_ic50,
            'fold_selective_JAK2_over_JAK1': jak1_ic50 / jak2_ic50,
        })

    print(f'Qualifying compounds (>=1 matching assay description): {len(qualifying_rows)}')

    # ---- Write outputs ----

    # Final compounds file
    with open(os.path.join(work_dir, 'ic50_any_shared_assay_compounds.tsv'), 'w', newline='') as f:
        fieldnames = ['smiles', 'JAK1_IC50_nM', 'JAK2_IC50_nM', 'matched_assay_description']
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        for row in qualifying_rows:
            w.writerow({
                'smiles': row['smiles'],
                'JAK1_IC50_nM': f"{row['jak1_ic50_nM']:.6g}",
                'JAK2_IC50_nM': f"{row['jak2_ic50_nM']:.6g}",
                'matched_assay_description': row['matched_assay_descriptions'],
            })

    # Reference file (all candidates, matched + unmatched)
    with open(os.path.join(work_dir, 'ic50_any_shared_assay_REFERENCE.tsv'), 'w', newline='') as f:
        fieldnames = ['smiles', 'qualifies', 'n_matched_assay_descriptions',
                      'matched_assay_descriptions', 'jak1_only_assay_descriptions',
                      'jak2_only_assay_descriptions']
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        w.writerows(reference_rows)

    # Full per-compound selectivity table
    with open(os.path.join(work_dir, 'selectivity_any_ic50_per_compound.tsv'), 'w', newline='') as f:
        fieldnames = ['smiles', 'jak1_ic50_nM', 'jak2_ic50_nM',
                      'matched_assay_descriptions', 'n_matched_assay_descriptions',
                      'n_jak1_values_used', 'n_jak2_values_used',
                      'fold_selective_JAK1_over_JAK2', 'fold_selective_JAK2_over_JAK1']
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        for row in qualifying_rows:
            out_row = dict(row)
            for k in ['jak1_ic50_nM', 'jak2_ic50_nM',
                      'fold_selective_JAK1_over_JAK2', 'fold_selective_JAK2_over_JAK1']:
                out_row[k] = f'{out_row[k]:.6g}'
            w.writerow(out_row)

    # Threshold summary
    thresholds = [2, 5, 10, 20, 30, 50]
    total = len(qualifying_rows)
    print()
    print(f'Total compounds with computable selectivity: {total}')
    print()
    print('Threshold | JAK1-over-JAK2 (n, %) | JAK2-over-JAK1 (n, %)')
    summary_rows = []
    for t in thresholds:
        n_jak1 = sum(1 for r in qualifying_rows if r['fold_selective_JAK1_over_JAK2'] >= t)
        n_jak2 = sum(1 for r in qualifying_rows if r['fold_selective_JAK2_over_JAK1'] >= t)
        pct_jak1 = 100 * n_jak1 / total if total else 0
        pct_jak2 = 100 * n_jak2 / total if total else 0
        print(f'>= {t}x   | {n_jak1} ({pct_jak1:.2f}%) | {n_jak2} ({pct_jak2:.2f}%)')
        summary_rows.append({
            'threshold': f'{t}x',
            'n_JAK1_over_JAK2': n_jak1,
            'pct_JAK1_over_JAK2': f'{pct_jak1:.2f}%',
            'n_JAK2_over_JAK1': n_jak2,
            'pct_JAK2_over_JAK1': f'{pct_jak2:.2f}%',
        })

    with open(os.path.join(work_dir, 'selectivity_any_ic50_summary.tsv'), 'w', newline='') as f:
        fieldnames = ['threshold', 'n_JAK1_over_JAK2', 'pct_JAK1_over_JAK2',
                      'n_JAK2_over_JAK1', 'pct_JAK2_over_JAK1']
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        w.writerows(summary_rows)

    print()
    print(f'Done. Outputs written to {os.path.abspath(work_dir)}')


if __name__ == '__main__':
    main()