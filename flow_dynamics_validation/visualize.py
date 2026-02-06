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


def generate_all_plots(results, empirical_samples, flow_samples, losses,
                       save_dir="plots"):
    """Generate all validation plots.

    Args:
        results: Results dictionary from compare_statistics.
        empirical_samples: shape (num_pairs, k, obs_dim).
        flow_samples: shape (num_pairs, num_samples, obs_dim).
        losses: Training loss history.
        save_dir: Directory to save plots.
    """
    plot_training_loss(losses, save_dir)
    plot_mean_comparison(results, save_dir)
    plot_std_comparison(results, save_dir)

    obs_dim = empirical_samples.shape[-1]
    dims_to_plot = list(range(min(3, obs_dim)))  # Plot first 3 dimensions
    plot_marginal_histograms(empirical_samples, flow_samples,
                              dims_to_plot=dims_to_plot, save_dir=save_dir)
    plot_per_pair_scatter(results, save_dir)

    print(f"\nAll plots saved to {save_dir}/")
