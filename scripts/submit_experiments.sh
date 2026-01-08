#!/bin/bash
# ==============================================================================
# Batch experiment submission script for HPRC cluster
# ==============================================================================
# This script submits multiple FQL experiments with different configurations
# Usage: ./submit_experiments.sh [experiment_set]
#   experiment_set: ogbench_state | ogbench_visual | d4rl | offline2online | all
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/run_fql.slurm"

# Create logs directory
mkdir -p logs

# Seeds for multiple runs
SEEDS=(0 1 2)

# Function to submit a single job
submit_job() {
    local env_name=$1
    local alpha=$2
    local discount=${3:-0.99}
    local q_agg=${4:-"mean"}
    local offline_steps=${5:-1000000}
    local online_steps=${6:-0}
    local run_group=${7:-"hprc_exp"}
    
    for seed in "${SEEDS[@]}"; do
        echo "Submitting: env=${env_name}, alpha=${alpha}, seed=${seed}"
        sbatch --export=ALL,ENV_NAME="${env_name}",ALPHA="${alpha}",DISCOUNT="${discount}",Q_AGG="${q_agg}",OFFLINE_STEPS="${offline_steps}",ONLINE_STEPS="${online_steps}",SEED="${seed}",RUN_GROUP="${run_group}" \
               --job-name="fql_${env_name}_s${seed}" \
               "${SLURM_SCRIPT}"
        sleep 0.5  # Small delay between submissions
    done
}

# ==============================================================================
# OGBench State-based Experiments (Offline RL)
# ==============================================================================
submit_ogbench_state() {
    echo "=== Submitting OGBench State-based Experiments ==="
    
    # AntMaze
    submit_job "antmaze-large-navigate-singletask-v0" 10 0.99 "min" 1000000 0 "ogbench_state"
    submit_job "antmaze-giant-navigate-singletask-v0" 10 0.995 "min" 1000000 0 "ogbench_state"
    
    # HumanoidMaze
    submit_job "humanoidmaze-medium-navigate-singletask-v0" 30 0.995 "mean" 1000000 0 "ogbench_state"
    submit_job "humanoidmaze-large-navigate-singletask-v0" 30 0.995 "mean" 1000000 0 "ogbench_state"
    
    # AntSoccer
    submit_job "antsoccer-arena-navigate-singletask-v0" 10 0.995 "mean" 1000000 0 "ogbench_state"
    
    # Cube manipulation
    submit_job "cube-single-play-singletask-v0" 300 0.99 "mean" 1000000 0 "ogbench_state"
    submit_job "cube-double-play-singletask-v0" 300 0.99 "mean" 1000000 0 "ogbench_state"
    
    # Scene
    submit_job "scene-play-singletask-v0" 300 0.99 "mean" 1000000 0 "ogbench_state"
    
    # Puzzle
    submit_job "puzzle-3x3-play-singletask-v0" 1000 0.99 "mean" 1000000 0 "ogbench_state"
    submit_job "puzzle-4x4-play-singletask-v0" 1000 0.99 "mean" 1000000 0 "ogbench_state"
}

# ==============================================================================
# OGBench Visual Experiments (Offline RL)
# ==============================================================================
submit_ogbench_visual() {
    echo "=== Submitting OGBench Visual Experiments ==="
    
    submit_job "visual-cube-single-play-singletask-task1-v0" 300 0.99 "mean" 500000 0 "ogbench_visual"
    submit_job "visual-cube-double-play-singletask-task1-v0" 100 0.99 "mean" 500000 0 "ogbench_visual"
    submit_job "visual-scene-play-singletask-task1-v0" 100 0.99 "mean" 500000 0 "ogbench_visual"
    submit_job "visual-puzzle-3x3-play-singletask-task1-v0" 300 0.99 "mean" 500000 0 "ogbench_visual"
    submit_job "visual-puzzle-4x4-play-singletask-task1-v0" 300 0.99 "mean" 500000 0 "ogbench_visual"
}

# ==============================================================================
# D4RL Experiments (Offline RL)
# ==============================================================================
submit_d4rl() {
    echo "=== Submitting D4RL Experiments ==="
    
    # AntMaze
    submit_job "antmaze-umaze-v2" 10 0.99 "mean" 500000 0 "d4rl"
    submit_job "antmaze-umaze-diverse-v2" 10 0.99 "mean" 500000 0 "d4rl"
    submit_job "antmaze-medium-play-v2" 10 0.99 "mean" 500000 0 "d4rl"
    submit_job "antmaze-medium-diverse-v2" 10 0.99 "mean" 500000 0 "d4rl"
    submit_job "antmaze-large-play-v2" 3 0.99 "mean" 500000 0 "d4rl"
    submit_job "antmaze-large-diverse-v2" 3 0.99 "mean" 500000 0 "d4rl"
    
    # Adroit
    submit_job "pen-human-v1" 10000 0.99 "min" 500000 0 "d4rl"
    submit_job "pen-cloned-v1" 10000 0.99 "min" 500000 0 "d4rl"
    submit_job "door-human-v1" 30000 0.99 "min" 500000 0 "d4rl"
    submit_job "door-cloned-v1" 30000 0.99 "min" 500000 0 "d4rl"
}

# ==============================================================================
# Offline-to-Online Experiments
# ==============================================================================
submit_offline2online() {
    echo "=== Submitting Offline-to-Online Experiments ==="
    
    submit_job "humanoidmaze-medium-navigate-singletask-v0" 100 0.995 "mean" 1000000 1000000 "offline2online"
    submit_job "antsoccer-arena-navigate-singletask-v0" 30 0.995 "mean" 1000000 1000000 "offline2online"
    submit_job "cube-double-play-singletask-v0" 300 0.99 "mean" 1000000 1000000 "offline2online"
    submit_job "scene-play-singletask-v0" 300 0.99 "mean" 1000000 1000000 "offline2online"
    submit_job "puzzle-4x4-play-singletask-v0" 1000 0.99 "mean" 1000000 1000000 "offline2online"
}

# ==============================================================================
# Main
# ==============================================================================
case "${1:-all}" in
    ogbench_state)
        submit_ogbench_state
        ;;
    ogbench_visual)
        submit_ogbench_visual
        ;;
    d4rl)
        submit_d4rl
        ;;
    offline2online)
        submit_offline2online
        ;;
    all)
        submit_ogbench_state
        submit_ogbench_visual
        submit_d4rl
        submit_offline2online
        ;;
    *)
        echo "Usage: $0 [ogbench_state|ogbench_visual|d4rl|offline2online|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Job submission complete! ==="
echo "Check job status with: squeue -u \$USER"
