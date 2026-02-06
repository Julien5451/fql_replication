"""Main entry point for Flow Model Dynamics Validation.

Validates whether a conditional flow matching model can approximate
the nominal transition distribution p_0(s' | s, a) from an offline
RL dataset (e.g., D4RL hopper-medium-v2).

This is a diagnostic experiment only -- no policy learning or robust RL.

Usage:
    # Base run (original behaviour, no variance loss):
    python3 main.py

    # With variance-matching loss:
    python3 main.py --w_std 0.3 --knn_k_train 50

    # With stochastic SDE sampling:
    python3 main.py --noise_scale 0.05 --temperature 1.0

    # Full run with both:
    python3 main.py --w_std 0.3 --noise_scale 0.05 --knn_k_train 50
"""

import argparse
import os
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
    precompute_knn_std,
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


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Flow Model Dynamics Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
                        help="Maximum transitions to use (0 = all)")

    # Training -- base
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--hidden_dims", type=int, nargs="+",
                        default=[256, 256, 256],
                        help="Hidden dims for velocity field MLP")
    parser.add_argument("--time_embed_dim", type=int, default=64,
                        help="Time embedding dimension")
    parser.add_argument("--layer_norm", action="store_true", default=True,
                        help="Use layer normalisation")
    parser.add_argument("--log_interval", type=int, default=20,
                        help="Print loss every N epochs")

    # Training -- variance-matching loss (B)
    parser.add_argument("--w_std", type=float, default=0.0,
                        help="Weight on variance-matching loss (0 = off)")
    parser.add_argument("--knn_k_train", type=int, default=50,
                        help="k for KNN empirical std used in variance loss")
    parser.add_argument("--num_model_samples_train_std", type=int, default=64,
                        help="M: model samples per (s,a) in variance loss")
    parser.add_argument("--std_batch_size", type=int, default=16,
                        help="Sub-batch for the variance-matching term")
    parser.add_argument("--num_ode_steps_std", type=int, default=20,
                        help="ODE steps inside the variance-matching ODE")

    # Training -- diagnostics (C)
    parser.add_argument("--std_eval_interval", type=int, default=0,
                        help="Evaluate std_ratio every N epochs (0 = off)")
    parser.add_argument("--num_std_eval_pairs", type=int, default=100,
                        help="Eval pairs for std_ratio diagnostic")

    # Evaluation / sampling
    parser.add_argument("--num_eval_pairs", type=int, default=100,
                        help="Number of (s,a) pairs for final evaluation")
    parser.add_argument("--k_neighbors", type=int, default=50,
                        help="Neighbours for empirical distribution")
    parser.add_argument("--num_flow_samples", type=int, default=50,
                        help="Flow model samples per (s,a)")
    parser.add_argument("--num_ode_steps", type=int, default=50,
                        help="Euler ODE integration steps")

    # Sampling stochasticity (A)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Scale of base noise x_0 ~ N(0, T^2 I)")
    parser.add_argument("--noise_scale", type=float, default=0.0,
                        help="Per-step SDE noise magnitude (0 = deterministic ODE)")

    return parser.parse_args()


# ------------------------------------------------------------------
# Summary writer
# ------------------------------------------------------------------

def write_summary(pointwise_results, neighbor_results, global_results,
                  history, args, save_dir="plots"):
    """Write a Markdown summary of the validation results."""
    obs_dim = len(pointwise_results["pointwise_per_dim_mae"])
    pw_mae = pointwise_results["pointwise_avg_mae"]
    pw_rmse = pointwise_results["pointwise_avg_rmse"]

    global_std_ratio = np.mean(global_results["global_std_ratio"])
    global_mean_err = np.mean(global_results["global_mean_abs_error"])

    nb_mean_err = np.mean(neighbor_results["mean_abs_error"])
    nb_std_ratio = np.mean(neighbor_results["std_ratio"])
    nb_std_median = float(np.median(neighbor_results["std_ratio"]))

    pw_quality = "GOOD" if pw_mae < 0.15 else ("MODERATE" if pw_mae < 0.4 else "POOR")
    global_mean_quality = "GOOD" if global_mean_err < 0.1 else (
        "MODERATE" if global_mean_err < 0.3 else "POOR")
    global_std_quality = "GOOD" if 0.7 < global_std_ratio < 1.3 else (
        "MODERATE" if 0.5 < global_std_ratio < 2.0 else "POOR")
    nb_std_quality = "GOOD" if 0.5 < nb_std_ratio < 2.0 else (
        "MODERATE" if 0.2 < nb_std_ratio < 5.0 else "POOR")

    if global_std_ratio < 0.7:
        global_std_assessment = "UNDER-estimated (variance collapse)"
    elif global_std_ratio > 1.3:
        global_std_assessment = "OVER-estimated (variance explosion)"
    else:
        global_std_assessment = "well-matched"

    losses = history["losses"]

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
- **w_std (variance loss weight)**: {args.w_std}
- **knn_k_train**: {args.knn_k_train}
- **M (model samples for std loss)**: {args.num_model_samples_train_std}
- **std_batch_size**: {args.std_batch_size}
- **temperature**: {args.temperature}
- **noise_scale**: {args.noise_scale}
- **Eval pairs**: {args.num_eval_pairs}
- **Neighbours (k)**: {args.k_neighbors}
- **Flow samples**: {args.num_flow_samples}
- **ODE steps**: {args.num_ode_steps}

## Training
- **Final total loss**: {losses[-1]:.6f}
- **Final base loss**: {history['base_losses'][-1]:.6f}
- **Final std loss**: {history['std_losses'][-1]:.6f}
- **Min total loss**: {min(losses):.6f}

## 1. Pointwise Prediction Accuracy

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
## 3. Neighbourhood (KNN) Conditional Comparison

- **Neighbour Mean Abs Error (avg)**: {nb_mean_err:.4f}
- **Neighbour Std Ratio (avg)**: {nb_std_ratio:.4f}
- **Neighbour Std Ratio (median)**: {nb_std_median:.4f}
- **Assessment**: {nb_std_quality}

### Per-Dimension KNN Std Ratio
| Dimension | Emp Std | Flow Std | Ratio |
|-----------|---------|----------|-------|
"""
    for i in range(obs_dim):
        summary += (f"| dim_{i} | {neighbor_results['avg_empirical_std'][i]:.4f} | "
                    f"{neighbor_results['avg_flow_std'][i]:.4f} | "
                    f"{neighbor_results['std_ratio'][i]:.4f} |\n")

    summary += f"""
## Conclusion

The conditional flow matching model {"is a reasonable" if pw_quality != "POOR" else "may NOT be a suitable"} \
proposal distribution for p_0(s' | s, a) in offline RL on {args.env_name}.

- **Pointwise prediction**: {pw_quality} (MAE = {pw_mae:.4f}, RMSE = {pw_rmse:.4f})
- **Global mean matching**: {global_mean_quality} (avg error = {global_mean_err:.4f})
- **Global variance matching**: {global_std_quality} (std ratio = {global_std_ratio:.4f})
- **Conditional (KNN) std ratio**: avg={nb_std_ratio:.4f}, median={nb_std_median:.4f}

## Plots
- `training_loss.png` / `training_losses.png`: Loss curves
- `std_ratio_trends.png`: Conditional std_ratio during training (if enabled)
- `pointwise_prediction.png` / `pointwise_error_bars.png`
- `mean_comparison.png` / `std_comparison.png`
- `per_pair_mean_scatter.png`
- `global_comparison.png`
- `marginal_histograms_global.png` / `marginal_histograms_neighborhood.png`
"""

    filepath = os.path.join(save_dir, "summary.md")
    with open(filepath, "w") as f:
        f.write(summary)
    print(f"\nSummary written to {filepath}")
    return summary


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 70)
    print("Flow Model Dynamics Validation")
    print(f"Environment: {args.env_name}")
    print(f"w_std={args.w_std}  temperature={args.temperature}  "
          f"noise_scale={args.noise_scale}")
    print("=" * 70)

    # ---- 1. Dataset Preparation ----
    print("\n[Step 1] Loading dataset...")
    raw_dataset = load_dataset(args.env_name, data_dir=args.data_dir)

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
    print(f"  Normalised obs range [{dataset['observations'].min():.2f}, "
          f"{dataset['observations'].max():.2f}]")

    # ---- 1b. Precompute KNN std (if needed) ----
    knn_std_all = None
    if args.w_std > 0.0 or args.std_eval_interval > 0:
        print("\n[Step 1b] Precomputing KNN empirical std...")
        knn_std_all = precompute_knn_std(
            dataset, k=args.knn_k_train, chunk_size=1000,
        )
        print(f"  knn_std shape: {knn_std_all.shape}, "
              f"mean={knn_std_all.mean():.4f}, max={knn_std_all.max():.4f}")

    # ---- 1c. Prepare std_ratio eval set (if diagnostics enabled) ----
    std_eval_obs = std_eval_act = std_eval_knn_std = None
    if args.std_eval_interval > 0 and knn_std_all is not None:
        eval_idx = np.random.choice(
            len(dataset["observations"]),
            size=min(args.num_std_eval_pairs, len(dataset["observations"])),
            replace=False,
        )
        std_eval_obs = dataset["observations"][eval_idx]
        std_eval_act = dataset["actions"][eval_idx]
        std_eval_knn_std = knn_std_all[eval_idx]

    # ---- 2. Train Flow Model ----
    print("\n[Step 2] Training conditional flow matching model...")
    model, params, history = train_flow_model(
        dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=tuple(args.hidden_dims),
        time_embed_dim=args.time_embed_dim,
        layer_norm=args.layer_norm,
        seed=args.seed,
        log_interval=args.log_interval,
        # variance-matching
        w_std=args.w_std,
        knn_std_all=knn_std_all,
        std_batch_size=args.std_batch_size,
        num_model_samples_train_std=args.num_model_samples_train_std,
        num_ode_steps_std=args.num_ode_steps_std,
        # diagnostics
        std_eval_interval=args.std_eval_interval,
        std_eval_obs=std_eval_obs,
        std_eval_act=std_eval_act,
        std_eval_knn_std=std_eval_knn_std,
        plot_dir=args.plot_dir,
    )

    # ---- 3. Select Evaluation Batch & Collect Empirical Samples ----
    print("\n[Step 3] Preparing evaluation data...")
    eval_batch = select_evaluation_batch(
        dataset, num_pairs=args.num_eval_pairs, seed=args.seed,
    )
    empirical_samples = collect_neighborhood_samples(
        dataset, eval_batch, k_neighbors=args.k_neighbors, seed=args.seed,
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
        temperature=args.temperature,
        noise_scale=args.noise_scale,
    )
    flow_samples = np.array(flow_samples_jax)
    print(f"  Flow samples shape: {flow_samples.shape}")

    # ---- 4b. Global flow samples (for marginal comparison) ----
    print("\n[Step 4b] Generating global flow samples for marginal comparison...")
    num_global = min(2000, len(dataset["observations"]))
    rng_global = jax.random.PRNGKey(args.seed + 2)
    global_idx = np.random.choice(
        len(dataset["observations"]), size=num_global, replace=False,
    )
    global_flow_jax = euler_sample_fast(
        model, params,
        jnp.array(dataset["observations"][global_idx]),
        jnp.array(dataset["actions"][global_idx]),
        rng_global,
        num_samples=1,
        num_steps=args.num_ode_steps,
        temperature=args.temperature,
        noise_scale=args.noise_scale,
    )
    flow_global_samples = np.array(global_flow_jax).squeeze(1)
    print(f"  Global flow samples shape: {flow_global_samples.shape}")

    # ---- 5. Statistical Comparison ----
    print("\n[Step 5] Computing statistical comparison...")
    pointwise_results = compute_pointwise_accuracy(
        eval_batch["next_observations"], flow_samples,
    )
    neighbor_results = compare_statistics(empirical_samples, flow_samples)
    global_results = compare_global_statistics(
        dataset["next_observations"], flow_global_samples,
    )
    print_comparison_report(pointwise_results, neighbor_results, global_results)

    # ---- 6. Visualization ----
    print("\n[Step 6] Generating plots...")
    os.makedirs(args.plot_dir, exist_ok=True)
    generate_all_plots(
        pointwise_results, neighbor_results, global_results,
        empirical_samples, flow_samples, history,
        dataset["next_observations"], flow_global_samples,
        save_dir=args.plot_dir,
    )

    # ---- 7. Summary ----
    print("\n[Step 7] Writing summary...")
    summary = write_summary(
        pointwise_results, neighbor_results, global_results,
        history, args, save_dir=args.plot_dir,
    )
    print(summary)

    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
