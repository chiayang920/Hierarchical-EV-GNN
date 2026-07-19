#!/bin/bash
#SBATCH --job-name=evgnn_vectorised_composition
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-2%1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err

set -euo pipefail

THREAD_COUNT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${THREAD_COUNT}"
export MKL_NUM_THREADS="${THREAD_COUNT}"
export NUMEXPR_NUM_THREADS="${THREAD_COUNT}"

RUN_ROOT="/scratch2/fr57/cche0357/EV-GNN_runs/vectorised_hierarchical_composition"
OUTPUT_ROOT="/projects/fr57/cche0357/EV-GNN_outputs"
REPO_DIR="/projects/fr57/cche0357/EV-GNN"
REPLICATION="${SLURM_ARRAY_TASK_ID}"
RUN_DIR="${RUN_ROOT}/replication_${REPLICATION}_job_${SLURM_JOB_ID}"
PACKAGE_PATH="${OUTPUT_ROOT}/vectorised_hierarchical_composition_rep${REPLICATION}_job${SLURM_JOB_ID}.tar.gz"

mkdir -p "${RUN_DIR}" "${OUTPUT_ROOT}"

echo "date: $(date -Is)"
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"

cd "${REPO_DIR}"
echo "working directory: $(pwd)"

for required_source in \
  TD3/TD3_HierarchicalActionGNN.py \
  profiling/profile_vectorised_hierarchical_composition.py
do
  if [ ! -s "${required_source}" ]; then
    echo "ERROR: required source file missing or empty: ${required_source}" >&2
    exit 1
  fi
done

if [ -n "${EV_GNN_SOURCE_PROVENANCE:-}" ]; then
  echo "source provenance: ${EV_GNN_SOURCE_PROVENANCE}"
  echo "source manifest path: ${EV_GNN_SOURCE_MANIFEST:-}"
elif [ -n "${EV_GNN_SOURCE_MANIFEST:-}" ] && [ -s "${EV_GNN_SOURCE_MANIFEST}" ]; then
  export EV_GNN_SOURCE_PROVENANCE="manifest:${EV_GNN_SOURCE_MANIFEST}"
  echo "source provenance: ${EV_GNN_SOURCE_PROVENANCE}"
  echo "source manifest path: ${EV_GNN_SOURCE_MANIFEST}"
  echo "source manifest contents:"
  cat "${EV_GNN_SOURCE_MANIFEST}"
else
  echo "ERROR: set EV_GNN_SOURCE_PROVENANCE or EV_GNN_SOURCE_MANIFEST before submission." >&2
  exit 1
fi

module load miniforge3/24.3.0-0
conda activate /scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu

python - <<'PY'
import platform
import sys

import torch
import torch_geometric

print(f"Python executable: {sys.executable}")
print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
print(f"Torch num threads: {torch.get_num_threads()}")
print(f"Torch interop threads: {torch.get_num_interop_threads()}")
PY

python profiling/profile_vectorised_hierarchical_composition.py \
  --output_dir "${RUN_DIR}" \
  --run_name "m3_vectorised_hierarchical_composition" \
  --replication "${REPLICATION}" \
  --device cpu \
  --scales 25CP 100CP 500CP \
  --seeds 0 1 2 3 4 \
  --workloads online_representative_state replay_batch \
  --batch_sizes 1 8 16 32 64 \
  --warmup_repetitions 5 \
  --measured_repetitions 30

shopt -s nullglob
csv_files=("${RUN_DIR}"/*.csv)
json_files=("${RUN_DIR}"/*.json)

if [ "${#csv_files[@]}" -eq 0 ]; then
  echo "ERROR: no CSV output produced in ${RUN_DIR}" >&2
  exit 1
fi

if [ "${#json_files[@]}" -eq 0 ]; then
  echo "ERROR: no JSON output produced in ${RUN_DIR}" >&2
  exit 1
fi

output_basenames=()
for output_file in "${csv_files[@]}" "${json_files[@]}"; do
  if [ ! -s "${output_file}" ]; then
    echo "ERROR: missing or empty output ${output_file}" >&2
    exit 1
  fi
  output_basenames+=("$(basename "${output_file}")")
done

tar -czf "${PACKAGE_PATH}" -C "${RUN_DIR}" "${output_basenames[@]}"

if [ ! -s "${PACKAGE_PATH}" ]; then
  echo "ERROR: package was not created: ${PACKAGE_PATH}" >&2
  exit 1
fi

echo "Packaged outputs: ${PACKAGE_PATH}"
echo "EVGNN_VECTORISED_COMPOSITION_M3_COMPLETE_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
