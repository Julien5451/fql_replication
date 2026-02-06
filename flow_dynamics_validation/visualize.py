"""Visualization for flow model dynamics validation.

Generates comparison plots between empirical dataset statistics
and flow model sampled statistics.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_mean_comparison(results, save_dir="plots", filename="mean_comparison.png"):
    """Plot per-dimension mean comparison: dataset vs flow model.

    Args:
        results: Results dictionary from compare_statistics.
        save_dir: Directory to save the plot.
        filename: Filename for the plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    obs_dim = len(results["avg_empirical_mean"])
    dims = np.arange(obs_dim)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, obs_dim * 0.8), 6))
    bars1 = ax.bar(dims - width / 2, results["avg_empirical_mean"], width,
                   label="Dataset (empirical)", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(dims + width / 2, results["avg_flow_mean"], width,
                   label="Flow model", color="#FF5722", alpha=0.8)

    ax.set_xlabel("State Dimension", fontsize=12)
    ax.set_ylabel("Mean Value", fontsize=12)
    ax.set_title("Mean Comparison: Dataset vs Flow Model", fontsize=14)
    ax.set_xticks(dims)
    ax.set_xticklabels([f"d{i}" for i in range(obs_dim)])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved mean comparison plot to {filepath}")


def plot_std_comparison(results, save_dir="plots", filename="std_comparison.png"):
    """Plot per-dimension std comparison: dataset vs flow model.

    Args:
        results: Results dictionary from compare_statistics.
        save_dir: Directory to save the plot.
        filename: Filename for the plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    obs_dim = len(results["avg_empirical_std"])
    dims = np.arange(obs_dim)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, obs_dim * 0.8), 6))
    bars1 = ax.bar(dims - width / 2, results["avg_empirical_std"], width,
                   label="Dataset (empirical)", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(dims + width / 2, results["avg_flow_std"], width,
                   label="Flow model", color="#FF5722", alpha=0.8)

    ax.set_xlabel("State Dimension", fontsize=12)
    ax.set_ylabel("Standard Deviation", fontsize=12)
    ax.set_title("Std Comparison: Dataset vs Flow Model", fontsize=14)
    ax.set_xticks(dims)
    ax.set_xticklabels([f"d{i}" for i in range(obs_dim)])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved std comparison plot to {filepath}")


def plot_marginal_histograms(empirical_samples, flow_samples, dims_to_plot=(0, 1),
                              save_dir="plots", filename="marginal_histograms.png"):
    """Plot marginal distribution comparison via histograms/KDE.

    Args:
        empirical_samples: shape (num_pairs, k_neighbors, obs_dim)
        flow_samples: shape (num_pairs, num_samples, obs_dim)
        dims_to_plot: Which state dimensions to plot.
        save_dir: Directory to save the plot.
        filename: Filename for the plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Flatten across pairs to get overall marginal distributions
    emp_flat = empirical_samples.reshape(-1, empirical_samples.shape[-1])
    flow_flat = flow_samples.reshape(-1, flow_samples.shape[-1])

    num_dims = len(dims_to_plot)
    fig, axes = plt.subplots(1, num_dims, figsize=(7 * num_dims, 5))
    if num_dims == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims_to_plot):
        emp_vals = emp_flat[:, dim]
        flow_vals = flow_flat[:, dim]

        # Determine common range
        all_vals = np.concatenate([emp_vals, flow_vals])
        vmin, vmax = np.percentile(all_vals, [1, 99])

        bins = np.linspace(vmin, vmax, 60)

        ax.hist(emp_vals, bins=bins, density=True, alpha=0.6,
                color="#2196F3", label="Dataset", edgecolor="white", linewidth=0.5)
        ax.hist(flow_vals, bins=bins, density=True, alpha=0.6,
                color="#FF5722", label="Flow model", edgecolor="white", linewidth=0.5)

        ax.set_xlabel(f"State Dimension {dim}", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Marginal Distribution (dim {dim})", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved marginal histogram plot to {filepath}")


def plot_per_pair_scatter(results, save_dir="plots", filename="per_pair_mean_scatter.png"):
    """Scatter plot of per-pair empirical mean vs flow mean across all dimensions.

    Args:
        results: Results dictionary from compare_statistics.
        save_dir: Directory to save the plot.
        filename: Filename for the plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    emp_means = results["empirical_mean"].flatten()
    flow_means = results["flow_mean"].flatten()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(emp_means, flow_means, alpha=0.3, s=10, color="#673AB7")

    # Perfect prediction line
    lim_min = min(emp_means.min(), flow_means.min())
    lim_max = max(emp_means.max(), flow_means.max())
    margin = (lim_max - lim_min) * 0.05
    ax.plot([lim_min - margin, lim_max + margin],
            [lim_min - margin, lim_max + margin],
            "k--", alpha=0.5, label="y = x (perfect)")

    ax.set_xlabel("Empirical Mean", fontsize=12)
    ax.set_ylabel("Flow Model Mean", fontsize=12)
    ax.set_title("Per-Pair Mean: Empirical vs Flow", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-pair scatter plot to {filepath}")


def plot_training_loss(losses, save_dir="plots", filename="training_loss.png"):
    """Plot training loss curve.

    Args:
        losses: List of per-epoch average losses.
        save_dir: Directory to save the plot.
        filename: Filename for the plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(losses) + 1), losses, color="#009688", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Flow Matching Loss", fontsize=12)
    ax.set_title("Training Loss Curve", fontsize=14)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training loss plot to {filepath}")


def plot_pointwise_prediction(pointwise_results, save_dir="plots",
                               filename="pointwise_prediction.png"):
    """Plot pointwise prediction: flow mean vs true s' per dimension.

    Args:
        pointwise_results: Results from compute_pointwise_accuracy.
        save_dir: Directory to save the plot.
        filename: Filename for the plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    true_vals = pointwise_results["pointwise_true"]
    pred_vals = pointwise_results["pointwise_flow_mean"]
    obs_dim = true_vals.shape[1]

    num_dims = min(4, obs_dim)
    fig, axes = plt.subplots(1, num_dims, figsize=(5 * num_dims, 5))
    if num_dims == 1:
        axes = [axes]

    for ax, dim in zip(axes, range(num_dims)):
        ax.scatter(true_vals[:, dim], pred_vals[:, dim], alpha=0.5, s=20, color="#4CAF50")
        lim_min = min(true_vals[:, dim].min(), pred_vals[:, dim].min())
        lim_max = max(true_vals[:, dim].max(), pred_vals[:, dim].max())
        margin = (lim_max - lim_min) * 0.05
        ax.plot([lim_min - margin, lim_max + margin],
                [lim_min - margin, lim_max + margin],
                "k--", alpha=0.5, label="y = x")
        ax.set_xlabel(f"True s'[{dim}]", fontsize=11)
        ax.set_ylabel(f"Predicted s'[{dim}]", fontsize=11)
        ax.set_title(f"Dim {dim}", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

    fig.suptitle("Pointwise Prediction: Flow Mean vs True s'", fontsize=14, y=1.02)
    plt.tight_layout()
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved pointwise prediction plot to {filepath}")


def plot_pointwise_error_bars(pointwise_results, save_dir="plots",
                               filename="pointwise_error_bars.png"):
    """Bar plot of per-dimension MAE and RMSE.

    Args:
        pointwise_results: Results from compute_pointwise_accuracy.
        save_dir: Directory to save the plot.
        filename: Filename for the plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    mae = pointwise_results["pointwise_per_dim_mae"]
    rmse = pointwise_results["pointwise_per_dim_rmse"]
    obs_dim = len(mae)
    dims = np.arange(obs_dim)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, obs_dim * 0.8), 6))
    ax.bar(dims - width / 2, mae, width, label="MAE", color="#4CAF50", alpha=0.8)
    ax.bar(dims + width / 2, rmse, width, label="RMSE", color="#FF9800", alpha=0.8)

    ax.set_xlabel("State Dimension", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title("Pointwise Prediction Error per Dimension", fontsize=14)
    ax.set_xticks(dims)
    ax.set_xticklabels([f"d{i}" for i in range(obs_dim)])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved pointwise error bar plot to {filepath}")


def plot_global_comparison(global_results, save_dir="plots",
                            filename="global_comparison.png"):
    """Plot global marginal mean and std comparison.

    Args:
        global_results: Results from compare_global_statistics.
        save_dir: Directory to save the plot.
        filename: Filename for the plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    obs_dim = len(global_results["dataset_global_mean"])
    dims = np.arange(obs_dim)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Mean comparison
    ax1.bar(dims - width / 2, global_results["dataset_global_mean"], width,
            label="Dataset", color="#2196F3", alpha=0.8)
    ax1.bar(dims + width / 2, global_results["flow_global_mean"], width,
            label="Flow model", color="#FF5722", alpha=0.8)
    ax1.set_xlabel("State Dimension", fontsize=12)
    ax1.set_ylabel("Mean", fontsize=12)
    ax1.set_title("Global Marginal Mean", fontsize=13)
    ax1.set_xticks(dims)
    ax1.set_xticklabels([f"d{i}" for i in range(obs_dim)])
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # Std comparison
    ax2.bar(dims - width / 2, global_results["dataset_global_std"], width,
            label="Dataset", color="#2196F3", alpha=0.8)
    ax2.bar(dims + width / 2, global_results["flow_global_std"], width,
            label="Flow model", color="#FF5722", alpha=0.8)
    ax2.set_xlabel("State Dimension", fontsize=12)
    ax2.set_ylabel("Std", fontsize=12)
    ax2.set_title("Global Marginal Std", fontsize=13)
    ax2.set_xticks(dims)
    ax2.set_xticklabels([f"d{i}" for i in range(obs_dim)])
    ax2.legend(fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global comparison plot to {filepath}")


def plot_std_ratio_trends(history, save_dir="plots",
                          filename="std_ratio_trends.png"):
    """Plot std_ratio evolution during training (per-dim + median).

    Args:
        history: Training history dict with 'std_ratio_epochs',
            'std_ratio_per_dim', 'std_ratio_median'.
        save_dir: Directory to save the plot.
        filename: Filename for the plot.
    """
    epochs = history.get("std_ratio_epochs", [])
    per_dim = history.get("std_ratio_per_dim", [])
    medians = history.get("std_ratio_median", [])
    if not epochs:
        return  # nothing to plot

    os.makedirs(save_dir, exist_ok=True)
    per_dim = np.array(per_dim)  # (num_evals, obs_dim)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: median std_ratio over training
    ax1.plot(epochs, medians, "o-", color="#E91E63", linewidth=2, markersize=4)
    ax1.axhline(1.0, color="grey", linestyle="--", alpha=0.6, label="ideal = 1.0")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Median std_ratio (flow / empirical)", fontsize=12)
    ax1.set_title("Median Conditional Std Ratio During Training", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Right: per-dim std_ratio over training
    obs_dim = per_dim.shape[1]
    cmap = plt.cm.viridis(np.linspace(0, 1, obs_dim))
    for d in range(obs_dim):
        ax2.plot(epochs, per_dim[:, d], "-", color=cmap[d], alpha=0.7,
                 label=f"d{d}")
    ax2.axhline(1.0, color="grey", linestyle="--", alpha=0.6)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("std_ratio per dim", fontsize=12)
    ax2.set_title("Per-Dimension Conditional Std Ratio", fontsize=13)
    if obs_dim <= 12:
        ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved std_ratio trend plot to {filepath}")


def plot_training_losses(history, save_dir="plots",
                         filename="training_losses.png"):
    """Plot base loss, std loss, and total loss curves.

    Args:
        history: Training history dict with 'losses', 'base_losses',
            'std_losses'.
        save_dir: Directory to save the plot.
        filename: Filename for the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["losses"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["losses"], label="total", color="#009688", linewidth=1.5)
    ax.plot(epochs, history["base_losses"], label="base (CFM)",
            color="#2196F3", linewidth=1, alpha=0.8)
    if any(v > 0 for v in history["std_losses"]):
        ax.plot(epochs, history["std_losses"], label="std (variance)",
                color="#FF5722", linewidth=1, alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Curves", fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training losses plot to {filepath}")


def generate_all_plots(pointwise_results, neighbor_results, global_results,
                       empirical_samples, flow_samples, history,
                       dataset_next_obs, flow_global_samples,
                       save_dir="plots"):
    """Generate all validation plots.

    Args:
        pointwise_results: From compute_pointwise_accuracy.
        neighbor_results: From compare_statistics.
        global_results: From compare_global_statistics.
        empirical_samples: shape (num_pairs, k, obs_dim).
        flow_samples: shape (num_pairs, num_samples, obs_dim).
        history: Training history dict (from train_flow_model).
        dataset_next_obs: All dataset s' for global histograms.
        flow_global_samples: Flow samples for global histograms (flat).
        save_dir: Directory to save plots.
    """
    # Training curves
    losses = history["losses"]
    plot_training_loss(losses, save_dir)
    plot_training_losses(history, save_dir)

    # Std ratio trends (if available)
    plot_std_ratio_trends(history, save_dir)

    # Pointwise
    plot_pointwise_prediction(pointwise_results, save_dir)
    plot_pointwise_error_bars(pointwise_results, save_dir)

    # Neighborhood conditional
    plot_mean_comparison(neighbor_results, save_dir)
    plot_std_comparison(neighbor_results, save_dir)
    plot_per_pair_scatter(neighbor_results, save_dir)

    # Global marginal
    plot_global_comparison(global_results, save_dir)

    obs_dim = empirical_samples.shape[-1]
    dims_to_plot = list(range(min(3, obs_dim)))

    # Marginal histograms using global samples (more meaningful)
    # Reshape: dataset_next_obs (N, dim) -> (1, N, dim); flow_global (M, dim) -> (1, M, dim)
    ds_for_hist = dataset_next_obs[np.newaxis, :, :]
    flow_for_hist = flow_global_samples[np.newaxis, :, :]
    plot_marginal_histograms(ds_for_hist, flow_for_hist,
                              dims_to_plot=dims_to_plot, save_dir=save_dir,
                              filename="marginal_histograms_global.png")

    # Also plot neighborhood-level histograms
    plot_marginal_histograms(empirical_samples, flow_samples,
                              dims_to_plot=dims_to_plot, save_dir=save_dir,
                              filename="marginal_histograms_neighborhood.png")

    print(f"\nAll plots saved to {save_dir}/")
