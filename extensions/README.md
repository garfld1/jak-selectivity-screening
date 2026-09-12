# Isolated pharmacophore extensions

All scripts leave `src/pharmacophore_discovery.py` and `src/pharmacophore_validation.py` untouched. They import their feature masks/statistics where compatible and write only the new files listed below.

| Script | Reads | Creates |
| --- | --- | --- |
| `scaffold_holdout.py` | feature matrix, discovered candidates, `docking/docking_prep/ligands_pdbqt.csv` ligand-to-SMILES mapping | `scaffold_holdout_results.csv` |
| `specificity_scoring.py` | feature matrix, discovered candidates | `specificity_results.csv` |
| `ablation_check.py` | feature matrix, a user-supplied small survivor list | `ablation_flags.csv` |
| `synthesize_shortlist.py` | discovered candidates, random holdout, independent validation, and the three extension CSVs | `final_shortlist.csv`, `shortlist_report.md` |
| `counter_pharmacophore_discovery.py` | feature matrix; independent validation PLIP/Ki inputs when validation rescoring is requested | JAK1 and JAK2-antipharmacophore candidate/holdout/validation CSVs |

## Added thresholds

- Scaffold Butina clustering: Morgan/ECFP4 Tanimoto cutoff **0.40**; clusters are sampled to **18%** holdout.
- Specificity: score at least **1.5**.
- Ablation: every one-feature deletion retains at least **50%** of full-pattern JAK2 support and has enrichment no greater than **110%** of the full pattern.
- Antipharmacophore depletion: enrichment at most **1 / 1.5 = 0.6667**.

Each stage computes enrichment against its own local class baseline. Do not compare cross-stage enrichment magnitudes directly, especially across IC50 discovery and Ki validation.

## Supplied mappings

The training ligand-to-SMILES mapping is read from `docking/docking_prep/ligands_pdbqt.csv`, as explicitly supplied after the initial specification. The validation mapping is in `docking/docking_prep/validation_set_pdbqt.csv`; the counter-search script combines it with the existing validation PLIP inputs to materialize its new frozen-pattern validation matrix.
