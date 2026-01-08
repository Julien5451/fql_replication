#!/bin/bash
# ==============================================================================
# Quick single experiment submission script
# ==============================================================================
# Usage examples:
#   ./run_single.sh antmaze-large-navigate-singletask-v0
#   ./run_single.sh antmaze-large-navigate-singletask-v0 10 0.99 min
#   ./run_single.sh visual-cube-single-play-singletask-task1-v0 300
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/run_fql.slurm"

# Create logs directory
mkdir -p logs

ENV_NAME=${1:-"cube-double-play-singletask-v0"}
ALPHA=${2:-300}
DISCOUNT=${3:-0.99}
Q_AGG=${4:-"mean"}
SEED=${5:-0}

echo "Submitting single experiment:"
echo "  Environment: ${ENV_NAME}"
echo "  Alpha: ${ALPHA}"
echo "  Discount: ${DISCOUNT}"
echo "  Q_AGG: ${Q_AGG}"
echo "  Seed: ${SEED}"

# Determine offline steps based on environment
if [[ "${ENV_NAME}" == *"visual"* ]] || [[ "${ENV_NAME}" == *"-v1"* ]] || [[ "${ENV_NAME}" == *"-v2"* ]]; then
    OFFLINE_STEPS=500000
else
    OFFLINE_STEPS=1000000
fi

sbatch --export=ALL,ENV_NAME="${ENV_NAME}",ALPHA="${ALPHA}",DISCOUNT="${DISCOUNT}",Q_AGG="${Q_AGG}",SEED="${SEED}",OFFLINE_STEPS="${OFFLINE_STEPS}" \
       --job-name="fql_${ENV_NAME}_s${SEED}" \
       "${SLURM_SCRIPT}"
