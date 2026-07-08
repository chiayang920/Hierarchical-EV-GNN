#!/bin/bash
#SBATCH --job-name=evgnn_100cp_agg
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%j.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%j.err

set -euo pipefail

module load miniforge3/24.3.0-0
conda activate evgnn_m3_cpu

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

REPO_DIR="/projects/fr57/cche0357/EV-GNN"
OUT_DIR="/projects/fr57/cche0357/EV-GNN_outputs"
PY_SCRIPT="${REPO_DIR}/analysis/07_100cp_formal_aggregation.py"

EVAL_DIR="/scratch2/fr57/cche0357/EV-GNN_runs/phase2d_100cp_formal_eval30/eval"
AGG_DIR="/scratch2/fr57/cche0357/EV-GNN_runs/phase2d_100cp_formal_aggregation"
SCENARIO_NAME="phase2d_100cp"

mkdir -p "${OUT_DIR}" "${AGG_DIR}"

cd "${REPO_DIR}"

echo "=== Phase 2D.4 100 CP formal aggregation ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Repository: ${REPO_DIR}"
echo "Python script: ${PY_SCRIPT}"
echo "Eval dir: ${EVAL_DIR}"
echo "Aggregation output dir: ${AGG_DIR}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "Python: $(which python)"
python --version

echo "=== Dependency preflight ==="
python - <<'PY'
import sys
import numpy
import pandas
print("NUMPY", numpy.__version__)
print("PANDAS", pandas.__version__)
try:
    import scipy
    print("SCIPY", scipy.__version__)
except Exception as exc:
    print("SCIPY_IMPORT_WARNING", repr(exc))
try:
    import matplotlib
    print("MATPLOTLIB", matplotlib.__version__)
except Exception as exc:
    print("MATPLOTLIB_IMPORT_WARNING", repr(exc))
PY

echo "=== Input checks ==="
if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Missing Python aggregation script: ${PY_SCRIPT}"
  exit 2
fi

if [[ ! -d "${EVAL_DIR}" ]]; then
  echo "Missing eval directory: ${EVAL_DIR}"
  exit 3
fi

find "${EVAL_DIR}" -maxdepth 1 -type f -name "*_100cp_seed*_eval30.csv" | sort
CSV_COUNT=$(find "${EVAL_DIR}" -maxdepth 1 -type f -name "*_100cp_seed*_eval30.csv" | wc -l)
echo "Eval CSV count: ${CSV_COUNT}"

if [[ "${CSV_COUNT}" -ne 10 ]]; then
  echo "Expected exactly 10 eval30 CSV files, found ${CSV_COUNT}"
  exit 4
fi

echo "=== Run aggregation ==="
rm -rf "${AGG_DIR}"
mkdir -p "${AGG_DIR}"

python "${PY_SCRIPT}" \
  --eval_dir "${EVAL_DIR}" \
  --output_dir "${AGG_DIR}" \
  --scenario_name "${SCENARIO_NAME}" \
  --expected_seeds 5 \
  --expected_episodes 30

echo "=== Aggregation outputs ==="
find "${AGG_DIR}" -maxdepth 3 -type f | sort

echo "=== Package aggregation artefacts ==="
PACKAGE="${OUT_DIR}/phase2d_100cp_formal_aggregation_${SLURM_JOB_ID}.tar.gz"

tar -czf "${PACKAGE}" \
  -C "${AGG_DIR}" . \
  -C "${REPO_DIR}/analysis" 07_100cp_formal_aggregation.py

ls -lh "${PACKAGE}"

echo "=== Phase 2D.4 aggregation completed ==="
