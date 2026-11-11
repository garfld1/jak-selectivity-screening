# JH1 dual-engine docking pipeline

Self-contained Vina + rDock pipeline for the JAK1 and JAK2 JH1 kinase domains.
Three steps, each runnable on its own:

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `prepare_receptors.py` | Receptor and docking-site prep for both engines, from crystal or ColabFold structures |
| 2 | `dual_engine_docking.py` | Docks a compound set with Vina and rDock, runs PLIP on every retained pose |
| 3 | `plip_tables.py` | Writes the single-pose and multi-pose PLIP tables (also callable standalone to rebuild tables) |

`docking_core.py` holds the engine-level primitives shared by steps 2 and 3.

## 1. Receptor and site preparation

```bash
python src/screening/jh1_dual_engine/prepare_receptors.py --source crystal
python src/screening/jh1_dual_engine/prepare_receptors.py --source colabfold
```

`--source crystal` uses the deposited kinase domains, JAK1 4EI4 and JAK2 4JI9.
`--source colabfold` uses the LocalColabFold JH1 models from
`Crystal Structures/colabfold_jh1/`; a model that has already been superposed
onto its crystal counterpart is reused as-is, otherwise it is superposed here
by CA atoms. Either way the structures end up in the crystal frame.

The docking site is the co-crystal inhibitor in the matching crystal: the
largest bound organic residue after excluding waters, ions, cryoprotectants and
modified amino acids. Its centroid is the box center for both engines. Vina
gets a cubic box of `--box-size` (default 20 Å); rDock gets an
`RbtSphereSiteMapper` cavity at the same center with `RADIUS` = half that edge,
so both engines search the same volume.

Output lands in `docking/receptors/<source>/`:

```
prepared_receptors/pdb/<target>.pdb       cleaned receptor, also the PLIP input
prepared_receptors/pdbqt/<target>.pdbqt   Vina receptor (Meeko)
rdock/<target>_protein.mol2               protonated receptor for rDock
rdock/<target>.prm                        rDock receptor parameter file
rdock/<target>.as                         cavity from `rbcavity --was`
rdock/prep_manifest.json                  rbcavity cavity volumes and warnings
receptors.json                            paths + box, consumed by step 2
```

Useful flags: `--box-size`, `--targets`, `--skip-rdock`, `--ptr-mode`
(phosphotyrosine becomes TYR for rDock by default, since rDock has no PTR
type), `--rdock-root`, `--outdir`.

## 2. Dual-engine docking

```bash
python src/screening/jh1_dual_engine/dual_engine_docking.py \
    --input data/qsar_df.csv \
    --receptors docking/receptors/crystal/receptors.json \
    --outdir results/jh1_dual_engine_crystal \
    --carry-cols delta_pIC50 \
    --exhaustiveness 16 \
    --num-poses 10 \
    --workers 4 --cpu-per-job 1
```

One job per (ligand, target, engine). Vina exhaustiveness is customizable via
`--exhaustiveness` and defaults to the value stored in `receptors.json`, which
`prepare_receptors.py` writes as 16. rDock runs `--rdock-runs` times (default
50) and then a heavy-atom diversity filter at `--rdock-cluster-rmsd` keeps at
most `--num-poses` poses. Lower score is better for both engines, so the top
pose is the best-scoring one in both cases. PLIP runs independently on every
retained pose.

`--num-poses` (default 10) caps how many poses the frequency tables can
aggregate, and `--frequency-windows` (default `5 10`) picks which windows get
a table.

`--save-poses` writes the retained poses to `<outdir>/poses/<target>/` as
gzipped PDBQT (Vina) or SD (rDock).

Runs are resumable. A JSONL checkpoint keyed by (ligand_id, target, engine)
records a config hash, and `--resume` reuses only lines whose hash matches the
current settings. Tables are rebuilt in full from the checkpoint at the end of
every run, so a resumed run still produces complete outputs. Starting a run in
a directory that already holds a checkpoint fails unless `--resume` is given,
so old results never leak into a new run's tables.
`--preflight-only` validates binaries, receptors and dependencies without
docking anything.

## 3. Result tables

Step 2 writes these automatically; `plip_tables.py` can also rebuild them from
an existing run directory without re-docking:

```bash
python src/screening/jh1_dual_engine/plip_tables.py --outdir results/jh1_dual_engine_crystal
```

Each (target, engine) pair gets a table for every combination of two
independent choices.

Pose scope:

- **top1** — the single best-scoring pose, cells are 0/1.
- **frequency_topN** — the N best-scoring poses, each cell the fraction of
  that window showing the interaction, with `n_poses` as the denominator.
  One table per entry in `--frequency-windows`, by default the top 5 and the
  top 10.

Feature resolution:

- **type** — exact PLIP interaction types, one column per
  `RESIDUE:interaction`, e.g. `ARG980:hydrogen_bond`.
- **residue** — binary PLIP contact map, one column per `RESIDUE`, set when
  that residue makes any interaction at all.

```
<engine>_<target>_type_top1.csv
<engine>_<target>_type_frequency_top5.csv
<engine>_<target>_type_frequency_top10.csv
<engine>_<target>_residue_top1.csv
<engine>_<target>_residue_frequency_top5.csv
<engine>_<target>_residue_frequency_top10.csv
```

A window only narrows what a ligand already has. A ligand with 7 retained
poses contributes 5 poses to the top-5 table and 7 to the top-10 table, and
its `n_poses` column says which. Columns are drawn from the windowed poses,
so a top-5 table never carries an all-zero column for an interaction that
only pose 7 made.

Plus, across all targets and engines: `docking_scores.csv` (top score and every
retained pose score), `docking_coverage_summary.csv` (ligand counts, column
counts, pose-count statistics), `failures.csv` and `run_manifest.json`.

Columns listed in `--carry-cols` are copied into every table, which is how an
endpoint such as `delta_pIC50` rides along into the interaction matrices.

## Requirements

`biopython`, `numpy`, `pandas`, `rdkit`, `meeko`, `plip`, `tqdm`, `openbabel`
(for the rDock mol2 conversion), an AutoDock Vina binary (default `~/bin/vina`)
and an rDock installation (default
`ensemble_pipeline_new/tools/src/rDock`).
