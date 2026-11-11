# Natural Compound Screening for Selective JAK2 Inhibitors

This repository screens compound libraries for molecules predicted to bind JAK2 more tightly than JAK1. It builds a selectivity model from paired ChEMBL IC50 data, ranks a library with that model, and docks the shortlist into the JAK1 and JAK2 JH1 kinase domains with AutoDock Vina and rDock. Protein–ligand interaction fingerprints from PLIP then compare how each pose contacts the two pockets.

The training endpoint is

```text
delta_pIC50 = JAK2_pIC50 − JAK1_pIC50
```

with `pIC50 = −log10(IC50 in M)`. A positive value means the compound is more potent on JAK2. The QSAR screen ranks new compounds by predicted `delta_pIC50`, highest first.

## Pipeline

1. **ChEMBL pairs.** Keep compounds that have an IC50 on both JAK1 and JAK2 in a shared assay description, convert units to nM, and summarize each target by the median of its pIC50 values.
2. **QSAR.** Featurize each SMILES as a Morgan fingerprint (radius 2, 2048 bits) and regress `delta_pIC50`. Cross-validation uses either a random split or Murcko-scaffold groups. A random forest trained on the full set is the screening model.
3. **Library screen.** Score a SMILES library, such as COCONUT or an FDA set, and keep a top fraction or a fixed top N.
4. **Receptors.** Prepare JAK1 (PDB [4EI4](https://www.rcsb.org/structure/4EI4)) and JAK2 (PDB [4JI9](https://www.rcsb.org/structure/4JI9)) from the crystal structures, or from LocalColabFold JH1 models superposed onto those crystals. The docking box is centered on the co-crystal inhibitor. JH1 spans used for the models are JAK1 875–1153 (UniProt P23458) and JAK2 849–1124 (UniProt O60674).
5. **Docking and interactions.** Dock each shortlisted compound into both receptors with Vina and rDock, run PLIP on the retained poses, and write score tables plus per-residue and per-interaction-type fingerprints.

Docking commands, output files, and resume behavior are documented in [`src/screening/jh1_dual_engine/README.md`](src/screening/jh1_dual_engine/README.md).

## Setup

Python dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` covers the cheminformatics and analysis stack (RDKit, scikit-learn, XGBoost, Mordred, Meeko, Gemmi, PLIP). Two extra pieces are needed for structure and docking:

- **Biopython**, used by the ColabFold and receptor-prep scripts.
- **Open Babel**, AutoDock Vina, and rDock. The docking scripts look for Vina at `~/bin/vina` and for rDock under `ensemble_pipeline_new/tools/src/rDock`. Override those paths with the flags in the docking README.
- **LocalColabFold**, only if you regenerate the JH1 models rather than using the structures already in the repo.

Run commands from the repository root. The COCONUT docking wrapper expects a conda environment named `jak_docking` at `~/miniforge3/envs/jak_docking`; set `PY` or `CONDA_ENV` to point elsewhere.

## Run the main stages

Build the paired ChEMBL table from `data/ChEMBL/chembl_jak1.tsv` and `data/ChEMBL/chembl_jak2.tsv`:

```bash
python src/training/chembl_preprocessing.py
```

Outputs land in `data/ChEMBL/`. The modeling table used downstream is `data/qsar_df.csv`, which adds `JAK1_pIC50`, `JAK2_pIC50`, `delta_pIC50`, and a Murcko `scaffold` column.

Compare models with 5-fold cross-validation:

```bash
python src/training/qsar_train.py \
    --input data/qsar_df.csv \
    --model rf \
    --split scaffold \
    --fp-type binary \
    --output-dir results/qsar
```

`--model` is `linear`, `rf`, `svr`, `xgb`, or `dummy`. `--split` is `random` or `scaffold`. `--fp-type` is `binary` or `count`. Metrics, predictions, and plots are written to `--output-dir`.

Fit the screening model on every row of `qsar_df` and save it:

```bash
python src/screening/qsar_train_save_model.py \
    --input data/qsar_df.csv \
    --output results/qsar/rf_final_model.joblib
```

Rank a library. The input needs a `smiles` column. `--top` accepts a percent (`1pct`, `5pct`, `10pct`) or a count (`4000`):

```bash
python src/screening/qsar_screening_model.py \
    --input data/fda.csv \
    --top 5pct \
    --output-dir results/qsar_screen
```

This reads `results/qsar/rf_final_model.joblib` and writes `all_predictions.csv` plus `top_<selection>.csv`, ranked by `predicted_delta_pIC50`.

Prepare receptors, then dock. Crystal and ColabFold sources share one box frame:

```bash
python src/screening/jh1_dual_engine/prepare_receptors.py --source crystal
python src/screening/jh1_dual_engine/prepare_receptors.py --source colabfold
```

```bash
python src/screening/jh1_dual_engine/dual_engine_docking.py \
    --input data/qsar_df.csv \
    --receptors docking/receptors/crystal/receptors.json \
    --outdir results/jh1_dual_engine_crystal \
    --carry-cols delta_pIC50 \
    --exhaustiveness 16 \
    --num-poses 10 \
    --workers 4
```

The COCONUT shortlist against the ColabFold receptors is wrapped here:

```bash
src/screening/jh1_dual_engine/run_coconut_colabfold.sh            # full run
src/screening/jh1_dual_engine/run_coconut_colabfold.sh --limit 5 # smoke test
src/screening/jh1_dual_engine/run_coconut_colabfold.sh --resume  # continue
```

To rebuild the JH1 models before receptor prep:

```bash
python src/structure/jak_colabfold_templates.py --protein JAK1 --construct jh1
python src/structure/colabfold_jh1_docking_prep.py --skip-fold
```

`--dry-run` on the template script prints the ColabFold jobs without launching them. `--skip-fold` reuses models already in `results/colabfold_jak/`.

## Notebooks

The notebooks are the exploratory record. Run them from the repository root.

| Notebook | Contents |
| --- | --- |
| `notebooks/01_chembl_data_exploration.ipynb` | Paired IC50 table, fold-selectivity labels, fingerprint PCA and t-SNE |
| `notebooks/02_QSAR_training.ipynb` | `delta_pIC50` labels, scaffold splits, model comparison, enrichment checks |
| `notebooks/03_test_screen.ipynb` | FDA and COCONUT screening outputs |
| `notebooks/04_coconut_compound_exploration.ipynb` | COCONUT download and similarity to the training set |
| `notebooks/05_docking_run.ipynb` | Docking run notes |
| `notebooks/06_interaction_analysis.ipynb` | Interaction-analysis workspace |

## Layout

```text
data/                     ChEMBL exports, qsar_df.csv, FDA and selectivity subsets
src/training/             ChEMBL preprocessing and QSAR cross-validation
src/screening/            Saved-model training, library screening, dual-engine docking
src/structure/            LocalColabFold JH1 jobs and superposition onto 4EI4 / 4JI9
Crystal Structures/       4EI4 and 4JI9, plus aligned ColabFold JH1 models
docking/receptors/        Prepared Vina and rDock receptors (crystal and colabfold)
notebooks/                Analysis notebooks
results/                  QSAR metrics, screens, and docking tables
```

Docking runs write, for each target and engine, a top-pose table and frequency tables over the top 5 and top 10 poses, at both interaction-type and residue resolution, along with `docking_scores.csv` and `docking_coverage_summary.csv`.
