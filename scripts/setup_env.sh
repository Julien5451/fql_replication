#!/bin/bash
# Setup script for FQL experiments on HPRC cluster
# This script creates a conda environment with all required dependencies

set -e

ENV_NAME="fql"

echo "=== Setting up FQL environment on HPRC cluster ==="

# Load required modules (adjust based on your cluster's module system)
module purge
module load GCC/11.3.0
module load CUDA/12.0.0  # Adjust CUDA version as needed
module load Anaconda3/2023.03

# Create conda environment
echo "Creating conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=3.10 -y

# Activate environment
source activate ${ENV_NAME}

# Install JAX with CUDA support
echo "Installing JAX with CUDA support..."
pip install --upgrade pip
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
echo "Installing project dependencies..."
pip install ogbench==1.1.0
pip install flax>=0.8.4
pip install distrax>=0.1.5
pip install ml_collections
pip install matplotlib
pip install moviepy
pip install wandb
pip install gymnasium==0.29.1
pip install gym==0.23.1
pip install numpy==1.26.4
pip install "shimmy[gym-v21,gym-v26]"
pip install "Cython<3"

# Optional: Install D4RL (may require MuJoCo setup)
echo "Installing D4RL (requires MuJoCo 2.1.0 to be set up)..."
pip install d4rl || echo "Warning: D4RL installation failed. Make sure MuJoCo 2.1.0 is set up."

echo "=== Environment setup complete! ==="
echo "To activate: conda activate ${ENV_NAME}"
