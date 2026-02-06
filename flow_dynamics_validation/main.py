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
from evaluate import compare_statistics, print_comparison_report
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


def write_summary(results, losses, args, save_dir="plots"):
    """Write a markdown summary of the validation results.

    Args:
        results: Statistics comparison results.
        losses: Training losses.
        args: Command-line arguments.
        save_dir: Directory to save the summary.
    """
    obs_dim = len(results["avg_empirical_mean"])
    mean_err = np.mean(results["mean_abs_error"])
    std_ratio_avg = np.mean(results["std_ratio"])
    std_ratio_min = np.min(results["std_ratio"])
    std_ratio_max = np.max(results["std_ratio"])

    # Determine quality assessment
    mean_quality = "GOOD" if mean_err < 0.3 else ("MODERATE" if mean_err < 0.8 else "POOR")
    std_quality = "GOOD" if 0.7 < std_ratio_avg < 1.3 else (
        "MODERATE" if 0.5 < std_ratio_avg < 2.0 else "POOR"
    )

    if std_ratio_avg < 0.7:
        std_assessment = "UNDER-estimated (variance collapse)"
    elif std_ratio_avg > 1.3:
        std_assessment = "OVER-estimated (variance explosion)"
    else:
        std_assessment = "well-matched"

    summary = f"""# Flow Model Dynamics Validation Summary

## Configuration
- **Environment**: {args.env_name}
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

## Mean Comparison
- **Overall Mean Absolute Error**: {mean_err:.4f}
- **Assessment**: {mean_quality}
- The flow model {"captures" if mean_quality != "POOR" else "does NOT capture"} the center (mean) of the transition distribution.

### Per-Dimension Mean Error
| Dimension | Empirical Mean | Flow Mean | Abs Error |
|-----------|---------------|-----------|-----------|
"""
    for i in range(obs_dim):
        summary += (f"| dim_{i} | {results['avg_empirical_mean'][i]:.4f} | "
                    f"{results['avg_flow_mean'][i]:.4f} | "
                    f"{results['mean_abs_error'][i]:.4f} |\n")

    summary += f"""
## Variance Comparison
- **Average Std Ratio (flow/empirical)**: {std_ratio_avg:.4f} (ideal: 1.0)
- **Min Std Ratio**: {std_ratio_min:.4f}
- **Max Std Ratio**: {std_ratio_max:.4f}
- **Assessment**: {std_quality} -- variance is {std_assessment}

### Per-Dimension Std Ratio
| Dimension | Empirical Std | Flow Std | Ratio |
|-----------|--------------|----------|-------|
"""
    for i in range(obs_dim):
        summary += (f"| dim_{i} | {results['avg_empirical_std'][i]:.4f} | "
                    f"{results['avg_flow_std'][i]:.4f} | "
                    f"{results['std_ratio'][i]:.4f} |\n")

    summary += f"""
## Conclusion
The conditional flow matching model {"appears to be a reasonable" if mean_quality != "POOR" and std_quality != "POOR" else "may NOT be a suitable"} \
proposal distribution for p_0(s' | s, a) in offline RL on {args.env_name}.

- **Mean matching**: {mean_quality} (avg abs error = {mean_err:.4f})
- **Variance matching**: {std_quality} (avg std ratio = {std_ratio_avg:.4f})

## Plots
- `training_loss.png`: Training loss curve
- `mean_comparison.png`: Per-dimension mean comparison
- `std_comparison.png`: Per-dimension std comparison
- `marginal_histograms.png`: Marginal distribution histograms
- `per_pair_mean_scatter.png`: Per-pair mean scatter plot
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
    stats = compute_normalization_stats(raw_dataset)
    dataset = normalize_dataset(raw_dataset, stats)
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

    # ---- 4. Sample from Flow Model ----
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

    # ---- 5. Statistical Comparison ----
    print("\n[Step 5] Computing statistical comparison...")
    results = compare_statistics(empirical_samples, flow_samples)
    print_comparison_report(results)

    # ---- 6. Visualization ----
    print("\n[Step 6] Generating plots...")
    os.makedirs(args.plot_dir, exist_ok=True)
    generate_all_plots(results, empirical_samples, flow_samples, losses,
                       save_dir=args.plot_dir)

    # ---- 7. Summary ----
    print("\n[Step 7] Writing summary...")
    summary = write_summary(results, losses, args, save_dir=args.plot_dir)
    print(summary)

    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
