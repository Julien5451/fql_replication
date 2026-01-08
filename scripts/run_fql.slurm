#!/bin/bash
#SBATCH --job-name=fql
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu              # GPU partition (adjust for your cluster)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # Number of CPU cores
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --mem=32G                    # Memory
#SBATCH --time=24:00:00              # Max walltime (adjust as needed)
#SBATCH --mail-type=BEGIN,END,FAIL   # Email notifications
#SBATCH --mail-user=your_email@tamu.edu  # Replace with your email

# ==============================================================================
# FQL Experiment SLURM Script for HPRC Cluster
# ==============================================================================
# Usage:
#   sbatch run_fql.slurm                                    # Run with defaults
#   sbatch --export=ENV_NAME="antmaze-large-navigate-singletask-v0" run_fql.slurm
#   sbatch --export=ENV_NAME="...",ALPHA=10,SEED=42 run_fql.slurm
# ==============================================================================

# Print job info
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "======================================"

# Load modules
module purge
module load GCC/11.3.0
module load CUDA/12.0.0
module load Anaconda3/2023.03

# Activate conda environment
source activate fql

# Set up environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export WANDB_MODE=${WANDB_MODE:-"online"}  # Set to "offline" if no internet

# Create logs directory
mkdir -p logs

# Navigate to project directory
cd ${SLURM_SUBMIT_DIR:-/workspace}

# Default parameters (can be overridden via --export)
ENV_NAME=${ENV_NAME:-"cube-double-play-singletask-v0"}
SEED=${SEED:-0}
RUN_GROUP=${RUN_GROUP:-"hprc_exp"}
AGENT=${AGENT:-"agents/fql.py"}
ALPHA=${ALPHA:-300}
DISCOUNT=${DISCOUNT:-0.99}
Q_AGG=${Q_AGG:-"mean"}
OFFLINE_STEPS=${OFFLINE_STEPS:-1000000}
ONLINE_STEPS=${ONLINE_STEPS:-0}

# Build command
CMD="python main.py \
    --env_name=${ENV_NAME} \
    --seed=${SEED} \
    --run_group=${RUN_GROUP} \
    --agent=${AGENT} \
    --agent.alpha=${ALPHA} \
    --agent.discount=${DISCOUNT} \
    --agent.q_agg=${Q_AGG} \
    --offline_steps=${OFFLINE_STEPS} \
    --online_steps=${ONLINE_STEPS}"

# Add optional parameters for visual environments
if [[ "${ENV_NAME}" == *"visual"* ]]; then
    CMD="${CMD} --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3"
fi

echo "Running command: ${CMD}"
echo "======================================"

# Run the experiment
${CMD}

echo "======================================"
echo "Job finished: $(date)"
echo "======================================"
