"""Main entry point for Flow Model Dynamics Validation.

Validates whether a conditional flow matching model can approximate
the nominal transition distribution p_0(s' | s, a) from an offline
RL dataset (e.g., D4RL hopper-medium-v2).

This is a diagnostic experiment only -- no policy learning or robust RL.

Usage:
    python3 main.py [--env_name hopper-medium-v2] [--num_epochs 200] ...
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from dataset import (
    load_dataset,
    compute_normalization_stats,
    normalize_dataset,
    unnormalize_next_obs,
    select_evaluation_batch,
    collect_neighborhood_samples,
)
from flow_model import create_flow_model
from train import train_flow_model
from sample import euler_sample_fast
from evaluate import (
    compute_pointwise_accuracy,
    compare_statistics,
    compare_global_statistics,
    print_comparison_report,
)
from visualize import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Flow Model Dynamics Validation")
    parser.add_argument("--env_name", type=str, default="hopper-medium-v2",
                        help="D4RL environment name")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory for dataset files")
    parser.add_argument("--plot_dir", type=str, default="plots",
                        help="Directory for output plots")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Data
    parser.add_argument("--max_data_size", type=int, default=100000,
                        help="Maximum number of transitions to use (subsample if larger)")

    # Training
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256, 256],
                        help="Hidden dimensions for velocity field MLP")
    parser.add_argument("--time_embed_dim", type=int, default=64,
                        help="Time embedding dimension")
    parser.add_argument("--layer_norm", action="store_true", default=True,
                        help="Use layer normalization")
    parser.add_argument("--log_interval", type=int, default=20,
                        help="Print loss every N epochs")

    # Evaluation
    parser.add_argument("--num_eval_pairs", type=int, default=100,
                        help="Number of (s, a) pairs for evaluation")
    parser.add_argument("--k_neighbors", type=int, default=50,
                        help="Number of nearest neighbors for empirical distribution")
    parser.add_argument("--num_flow_samples", type=int, default=50,
                        help="Number of flow model samples per (s, a)")
    parser.add_argument("--num_ode_steps", type=int, default=50,
                        help="Number of Euler ODE integration steps")

    return parser.parse_args()


def write_summary(pointwise_results, neighbor_results, global_results,
                  losses, args, save_dir="plots"):
    """Write a markdown summary of the validation results.

    Args:
        pointwise_results: Pointwise accuracy results.
        neighbor_results: KNN-based conditional comparison results.
        global_results: Global marginal comparison results.
        losses: Training losses.
        args: Command-line arguments.
        save_dir: Directory to save the summary.
    """
    obs_dim = len(pointwise_results["pointwise_per_dim_mae"])

    # Pointwise metrics
    pw_mae = pointwise_results["pointwise_avg_mae"]
    pw_rmse = pointwise_results["pointwise_avg_rmse"]

    # Global marginal metrics
    global_std_ratio = np.mean(global_results["global_std_ratio"])
    global_mean_err = np.mean(global_results["global_mean_abs_error"])

    # Neighborhood metrics
    nb_mean_err = np.mean(neighbor_results["mean_abs_error"])
    nb_std_ratio = np.mean(neighbor_results["std_ratio"])

    # Quality assessment based on pointwise accuracy (most meaningful)
    pw_quality = "GOOD" if pw_mae < 0.15 else ("MODERATE" if pw_mae < 0.4 else "POOR")

    # Global marginal quality
    global_mean_quality = "GOOD" if global_mean_err < 0.1 else (
        "MODERATE" if global_mean_err < 0.3 else "POOR")
    global_std_quality = "GOOD" if 0.7 < global_std_ratio < 1.3 else (
        "MODERATE" if 0.5 < global_std_ratio < 2.0 else "POOR")

    if global_std_ratio < 0.7:
        global_std_assessment = "UNDER-estimated (variance collapse)"
    elif global_std_ratio > 1.3:
        global_std_assessment = "OVER-estimated (variance explosion)"
    else:
        global_std_assessment = "well-matched"

    summary = f"""# Flow Model Dynamics Validation Summary

## Configuration
- **Environment**: {args.env_name}
- **Max data size**: {args.max_data_size}
- **Training epochs**: {args.num_epochs}
- **Batch size**: {args.batch_size}
- **Learning rate**: {args.learning_rate}
- **Hidden dims**: {args.hidden_dims}
- **Time embed dim**: {args.time_embed_dim}
- **Layer norm**: {args.layer_norm}
- **Eval pairs**: {args.num_eval_pairs}
- **Neighbors (k)**: {args.k_neighbors}
- **Flow samples**: {args.num_flow_samples}
- **ODE steps**: {args.num_ode_steps}

## Training
- **Final loss**: {losses[-1]:.6f}
- **Min loss**: {min(losses):.6f}

## 1. Pointwise Prediction Accuracy (Most Important)

This measures whether the flow model can predict the true s' for a given (s, a).

- **Average MAE**: {pw_mae:.4f}
- **Average RMSE**: {pw_rmse:.4f}
- **Assessment**: {pw_quality}

### Per-Dimension Pointwise Error
| Dimension | MAE | RMSE |
|-----------|-----|------|
"""
    for i in range(obs_dim):
        summary += (f"| dim_{i} | {pointwise_results['pointwise_per_dim_mae'][i]:.4f} | "
                    f"{pointwise_results['pointwise_per_dim_rmse'][i]:.4f} |\n")

    summary += f"""
## 2. Global Marginal Distribution Comparison

This measures whether the overall distribution of flow-generated s' matches
the dataset distribution of s' (when conditioned on random (s, a) pairs).

- **Global Mean Abs Error (avg)**: {global_mean_err:.4f}
- **Global Std Ratio (avg)**: {global_std_ratio:.4f} (ideal: 1.0)
- **Mean Assessment**: {global_mean_quality}
- **Std Assessment**: {global_std_quality} -- variance is {global_std_assessment}

### Per-Dimension Global Comparison
| Dimension | DS Mean | Flow Mean | DS Std | Flow Std | Std Ratio |
|-----------|---------|-----------|--------|----------|-----------|
"""
    for i in range(obs_dim):
        summary += (f"| dim_{i} | {global_results['dataset_global_mean'][i]:.4f} | "
                    f"{global_results['flow_global_mean'][i]:.4f} | "
                    f"{global_results['dataset_global_std'][i]:.4f} | "
                    f"{global_results['flow_global_std'][i]:.4f} | "
                    f"{global_results['global_std_ratio'][i]:.4f} |\n")

    summary += f"""
## 3. Neighborhood (KNN) Conditional Comparison

This compares the flow model's conditional samples against KNN-based empirical
samples. Note: in near-deterministic environments, the flow model correctly
produces low variance (concentrated around the true s'), while KNN-based
empirical samples have higher variance because neighbors have different (s, a).

- **Neighbor Mean Abs Error (avg)**: {nb_mean_err:.4f}
- **Neighbor Std Ratio (avg)**: {nb_std_ratio:.4f}

## Conclusion

The conditional flow matching model {"is a reasonable" if pw_quality != "POOR" else "may NOT be a suitable"} \
proposal distribution for p_0(s' | s, a) in offline RL on {args.env_name}.

- **Pointwise prediction**: {pw_quality} (MAE = {pw_mae:.4f}, RMSE = {pw_rmse:.4f})
- **Global mean matching**: {global_mean_quality} (avg error = {global_mean_err:.4f})
- **Global variance matching**: {global_std_quality} (std ratio = {global_std_ratio:.4f})

Note: The flow model learns a near-deterministic mapping for each (s, a) -> s',
which is appropriate for environments with deterministic (or near-deterministic) dynamics.
The low conditional variance is expected and correct behavior.

## Plots
- `training_loss.png`: Training loss curve
- `pointwise_prediction.png`: Flow mean vs true s' scatter
- `pointwise_error_bars.png`: Per-dimension MAE/RMSE
- `mean_comparison.png`: KNN conditional mean comparison
- `std_comparison.png`: KNN conditional std comparison
- `per_pair_mean_scatter.png`: Per-pair mean scatter
- `global_comparison.png`: Global marginal mean and std
- `marginal_histograms_global.png`: Global marginal distributions
- `marginal_histograms_neighborhood.png`: Neighborhood marginal distributions
"""

    filepath = os.path.join(save_dir, "summary.md")
    with open(filepath, "w") as f:
        f.write(summary)
    print(f"\nSummary written to {filepath}")
    return summary


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 70)
    print("Flow Model Dynamics Validation")
    print(f"Environment: {args.env_name}")
    print("=" * 70)

    # ---- 1. Dataset Preparation ----
    print("\n[Step 1] Loading dataset...")
    raw_dataset = load_dataset(args.env_name, data_dir=args.data_dir)

    # Subsample if dataset is too large
    n = len(raw_dataset["observations"])
    if args.max_data_size > 0 and n > args.max_data_size:
        print(f"  Subsampling from {n} to {args.max_data_size} transitions...")
        rng_sub = np.random.RandomState(args.seed)
        idx = rng_sub.choice(n, size=args.max_data_size, replace=False)
        for key in raw_dataset:
            raw_dataset[key] = raw_dataset[key][idx]

    stats = compute_normalization_stats(raw_dataset)
    dataset = normalize_dataset(raw_dataset, stats)
    print(f"  Dataset size: {len(dataset['observations'])}")
    print(f"  Normalized dataset: obs range [{dataset['observations'].min():.2f}, "
          f"{dataset['observations'].max():.2f}]")

    # ---- 2. Train Flow Model ----
    print("\n[Step 2] Training conditional flow matching model...")
    model, params, losses = train_flow_model(
        dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=tuple(args.hidden_dims),
        time_embed_dim=args.time_embed_dim,
        layer_norm=args.layer_norm,
        seed=args.seed,
        log_interval=args.log_interval,
    )

    # ---- 3. Select Evaluation Batch & Collect Empirical Samples ----
    print("\n[Step 3] Preparing evaluation data...")
    eval_batch = select_evaluation_batch(dataset, num_pairs=args.num_eval_pairs, seed=args.seed)
    empirical_samples = collect_neighborhood_samples(
        dataset, eval_batch, k_neighbors=args.k_neighbors, seed=args.seed
    )
    print(f"  Eval pairs: {args.num_eval_pairs}")
    print(f"  Empirical samples per pair: {args.k_neighbors}")
    print(f"  Empirical samples shape: {empirical_samples.shape}")

    # ---- 4. Sample from Flow Model (conditional, for eval pairs) ----
    print("\n[Step 4] Sampling from flow model...")
    rng = jax.random.PRNGKey(args.seed + 1)
    flow_samples_jax = euler_sample_fast(
        model, params,
        jnp.array(eval_batch["observations"]),
        jnp.array(eval_batch["actions"]),
        rng,
        num_samples=args.num_flow_samples,
        num_steps=args.num_ode_steps,
    )
    flow_samples = np.array(flow_samples_jax)
    print(f"  Flow samples shape: {flow_samples.shape}")

    # ---- 4b. Sample from Flow Model (global, for marginal comparison) ----
    print("\n[Step 4b] Generating global flow samples for marginal comparison...")
    num_global = min(2000, len(dataset["observations"]))
    rng_global = jax.random.PRNGKey(args.seed + 2)
    global_idx = np.random.choice(len(dataset["observations"]), size=num_global, replace=False)
    global_flow_jax = euler_sample_fast(
        model, params,
        jnp.array(dataset["observations"][global_idx]),
        jnp.array(dataset["actions"][global_idx]),
        rng_global,
        num_samples=1,
        num_steps=args.num_ode_steps,
    )
    flow_global_samples = np.array(global_flow_jax).squeeze(1)  # (num_global, obs_dim)
    print(f"  Global flow samples shape: {flow_global_samples.shape}")

    # ---- 5. Statistical Comparison ----
    print("\n[Step 5] Computing statistical comparison...")

    # 5a. Pointwise prediction accuracy
    pointwise_results = compute_pointwise_accuracy(
        eval_batch["next_observations"], flow_samples
    )

    # 5b. Neighborhood (conditional) comparison
    neighbor_results = compare_statistics(empirical_samples, flow_samples)

    # 5c. Global marginal comparison
    global_results = compare_global_statistics(
        dataset["next_observations"], flow_global_samples
    )

    print_comparison_report(pointwise_results, neighbor_results, global_results)

    # ---- 6. Visualization ----
    print("\n[Step 6] Generating plots...")
    os.makedirs(args.plot_dir, exist_ok=True)
    generate_all_plots(
        pointwise_results, neighbor_results, global_results,
        empirical_samples, flow_samples, losses,
        dataset["next_observations"], flow_global_samples,
        save_dir=args.plot_dir,
    )

    # ---- 7. Summary ----
    print("\n[Step 7] Writing summary...")
    summary = write_summary(pointwise_results, neighbor_results, global_results,
                            losses, args, save_dir=args.plot_dir)
    print(summary)

    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
