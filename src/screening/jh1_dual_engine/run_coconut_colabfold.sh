#!/usr/bin/env bash
#
# Dock the QSAR-prescreened COCONUT set against the ColabFold JAK1/JAK2 JH1
# receptors with both Vina and rDock, then write the PLIP tables.
#
#   src/screening/jh1_dual_engine/run_coconut_colabfold.sh              full 5,043-compound run
#   src/screening/jh1_dual_engine/run_coconut_colabfold.sh --limit 5    smoke test, separate outdir
#   src/screening/jh1_dual_engine/run_coconut_colabfold.sh --resume     continue an interrupted run
#
# Extra arguments are forwarded to dual_engine_docking.py, including
# --skip-mmgbsa. Receptors now keep crystallographic site waters and poses
# are reordered by MM/GBSA. Set FORCE_PREP=1 once so an existing dry
# receptor directory is rebuilt before a production run.
#
# Environment overrides:
#   PY CONDA_ENV INPUT OUTDIR RECEPTOR_DIR WORKERS CPU_PER_JOB
#   EXHAUSTIVENESS NUM_POSES FREQUENCY_WINDOWS FORCE_PREP
#
# The full run takes many hours, so detach it:
#   nohup src/screening/jh1_dual_engine/run_coconut_colabfold.sh > coconut_run.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# The base miniforge interpreter has no biopython; jak_docking has the full stack.
PY="${PY:-$HOME/miniforge3/envs/${CONDA_ENV:-jak_docking}/bin/python}"
INPUT="${INPUT:-results/docking/coconut/top_1pct_coconut.csv}"
RECEPTOR_DIR="${RECEPTOR_DIR:-docking/receptors}"
RECEPTORS="$RECEPTOR_DIR/colabfold/receptors.json"
WORKERS="${WORKERS:-8}"
CPU_PER_JOB="${CPU_PER_JOB:-1}"
EXHAUSTIVENESS="${EXHAUSTIVENESS:-16}"
# Ten poses retained, so the top-5 and top-10 frequency tables are both full.
NUM_POSES="${NUM_POSES:-10}"
FREQUENCY_WINDOWS="${FREQUENCY_WINDOWS:-5 10}"

OUTDIR_WAS_SET="${OUTDIR:+1}"
OUTDIR="${OUTDIR:-results/jh1_colabfold_coconut}"

# Keep a smoke test from sharing a checkpoint with the full run.
for arg in "$@"; do
    if [[ "$arg" == "--limit" && -z "$OUTDIR_WAS_SET" ]]; then
        OUTDIR="${OUTDIR}_smoke"
        break
    fi
done

if [[ ! -x "$PY" ]]; then
    echo "Python not found: $PY" >&2
    echo "Set PY to an interpreter with biopython, rdkit, meeko, plip and openbabel." >&2
    exit 1
fi
if [[ ! -f "$INPUT" ]]; then
    echo "Input not found: $INPUT" >&2
    exit 1
fi

# Step 1. The box is fixed by the crystal ligand, so this only needs doing once.
if [[ -f "$RECEPTORS" && "${FORCE_PREP:-0}" != "1" ]]; then
    echo "==> Using existing receptors: $RECEPTORS"
else
    echo "==> Preparing ColabFold JH1 receptors for Vina and rDock"
    "$PY" src/screening/jh1_dual_engine/prepare_receptors.py \
        --source colabfold \
        --outdir "$RECEPTOR_DIR" \
        --exhaustiveness "$EXHAUSTIVENESS"
fi

# Steps 2 and 3. Docking writes the PLIP tables itself once every job is done.
echo "==> Docking $(basename "$INPUT") into $OUTDIR"
set -x
"$PY" src/screening/jh1_dual_engine/dual_engine_docking.py \
    --input "$INPUT" \
    --receptors "$RECEPTORS" \
    --outdir "$OUTDIR" \
    --carry-cols predicted_delta_pIC50 QSAR_rank \
    --exhaustiveness "$EXHAUSTIVENESS" \
    --num-poses "$NUM_POSES" \
    --frequency-windows $FREQUENCY_WINDOWS \
    --workers "$WORKERS" \
    --cpu-per-job "$CPU_PER_JOB" \
    "$@"
set +x

echo
if [[ -d "$OUTDIR" ]]; then
    echo "==> Done. Tables in $OUTDIR:"
    ls -1 "$OUTDIR" | sed 's/^/    /'
fi
