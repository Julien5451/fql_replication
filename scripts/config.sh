#!/bin/bash
# ==============================================================================
# HPRC Cluster Configuration
# ==============================================================================
# Modify these settings according to your HPRC cluster setup
# ==============================================================================

# ------------------------------------------------------------------------------
# Cluster-specific settings (MODIFY THESE)
# ------------------------------------------------------------------------------

# Email for job notifications (CHANGE THIS!)
export HPRC_EMAIL="your_netid@tamu.edu"

# Partition/Queue settings
# Common TAMU HPRC partitions: gpu, gpu-a100, gpu-v100
export HPRC_PARTITION="gpu"

# GPU type (optional, e.g., a100, v100, rtx2080)
export HPRC_GPU_TYPE=""  # Leave empty to use any available GPU

# Account/Allocation (if required by your cluster)
export HPRC_ACCOUNT=""

# ------------------------------------------------------------------------------
# Resource settings
# ------------------------------------------------------------------------------

# Number of CPUs per task
export HPRC_CPUS=8

# Memory per job
export HPRC_MEM="32G"

# Default walltime (HH:MM:SS)
export HPRC_TIME="24:00:00"

# For visual/pixel experiments (may need more resources)
export HPRC_VISUAL_MEM="64G"
export HPRC_VISUAL_TIME="48:00:00"

# ------------------------------------------------------------------------------
# Module settings (MODIFY BASED ON YOUR CLUSTER)
# ------------------------------------------------------------------------------

# Modules to load (adjust versions based on what's available)
export HPRC_GCC_MODULE="GCC/11.3.0"
export HPRC_CUDA_MODULE="CUDA/12.0.0"
export HPRC_ANACONDA_MODULE="Anaconda3/2023.03"

# Alternative module configurations for different clusters:
# TAMU Grace cluster:
#   GCC/11.2.0, CUDA/11.7.0, Anaconda3/2022.05
# TAMU Terra cluster:
#   GCC/10.2.0, CUDA/11.3.0, Anaconda3/2021.05

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------

# Project directory (usually auto-detected)
export FQL_PROJECT_DIR="${FQL_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Results/checkpoints directory
export FQL_SAVE_DIR="${FQL_PROJECT_DIR}/exp"

# Logs directory
export FQL_LOG_DIR="${FQL_PROJECT_DIR}/logs"

# Conda environment name
export FQL_CONDA_ENV="fql"

# ------------------------------------------------------------------------------
# W&B (Weights & Biases) settings
# ------------------------------------------------------------------------------

# Set to "offline" if cluster has no internet access
export WANDB_MODE="online"

# W&B project name
export WANDB_PROJECT="fql"

# ------------------------------------------------------------------------------
# JAX settings
# ------------------------------------------------------------------------------

# Prevent JAX from preallocating all GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Fraction of GPU memory for JAX (0.8 = 80%)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
