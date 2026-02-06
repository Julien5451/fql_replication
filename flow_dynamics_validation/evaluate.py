"""Statistical comparison between empirical and flow model samples.

Computes per-dimension mean and standard deviation for both the empirical
dataset distribution and the flow model's sampled distribution, then
quantifies the agreement.
"""

import numpy as np


def compute_statistics(samples):
    """Compute mean and std for a set of samples.

    Args:
        samples: Array of shape (num_samples, dim) or (num_pairs, num_samples, dim).

    Returns:
        Dictionary with 'mean' and 'std' arrays.
    """
    return {
        "mean": np.mean(samples, axis=-2),
        "std": np.std(samples, axis=-2),
    }


def compare_statistics(empirical_samples, flow_samples):
    """Compare statistics between empirical and flow model samples.

    For each evaluation (s, a) pair, computes the mean and std of both
    empirical and flow-generated next states.

    Args:
        empirical_samples: shape (num_pairs, k_neighbors, obs_dim)
        flow_samples: shape (num_pairs, num_flow_samples, obs_dim)

    Returns:
        Dictionary containing all comparison statistics.
    """
    emp_stats = compute_statistics(empirical_samples)
    flow_stats = compute_statistics(flow_samples)

    # Aggregate across all eval pairs
    results = {
        # Per-pair statistics
        "empirical_mean": emp_stats["mean"],  # (num_pairs, obs_dim)
        "empirical_std": emp_stats["std"],  # (num_pairs, obs_dim)
        "flow_mean": flow_stats["mean"],  # (num_pairs, obs_dim)
        "flow_std": flow_stats["std"],  # (num_pairs, obs_dim)

        # Averaged across eval pairs -> per-dimension summary
        "avg_empirical_mean": np.mean(emp_stats["mean"], axis=0),  # (obs_dim,)
        "avg_empirical_std": np.mean(emp_stats["std"], axis=0),  # (obs_dim,)
        "avg_flow_mean": np.mean(flow_stats["mean"], axis=0),  # (obs_dim,)
        "avg_flow_std": np.mean(flow_stats["std"], axis=0),  # (obs_dim,)
    }

    # Mean absolute error between means (per dimension)
    results["mean_abs_error"] = np.mean(np.abs(emp_stats["mean"] - flow_stats["mean"]), axis=0)
    # Mean absolute error between stds (per dimension)
    results["std_abs_error"] = np.mean(np.abs(emp_stats["std"] - flow_stats["std"]), axis=0)

    # Relative mean error
    denom = np.maximum(np.abs(results["avg_empirical_mean"]), 1e-8)
    results["mean_rel_error"] = results["mean_abs_error"] / denom

    # Std ratio (flow / empirical) - 1.0 means perfect match
    results["std_ratio"] = results["avg_flow_std"] / np.maximum(results["avg_empirical_std"], 1e-8)

    return results


def compute_global_statistics(dataset):
    """Compute global mean and std of next_observations across entire dataset.

    Args:
        dataset: Dataset dict with 'next_observations'.

    Returns:
        Dictionary with 'global_mean' and 'global_std'.
    """
    next_obs = dataset["next_observations"]
    return {
        "global_mean": np.mean(next_obs, axis=0),
        "global_std": np.std(next_obs, axis=0),
    }


def compare_global_statistics(dataset, flow_samples_flat):
    """Compare global statistics: all dataset s' vs all flow samples.

    Args:
        dataset: Dataset dict with 'next_observations' of shape (N, obs_dim).
        flow_samples_flat: Flow samples of shape (M, obs_dim).

    Returns:
        Dictionary with global comparison results.
    """
    ds_next = dataset["next_observations"]
    ds_mean = np.mean(ds_next, axis=0)
    ds_std = np.std(ds_next, axis=0)

    flow_mean = np.mean(flow_samples_flat, axis=0)
    flow_std = np.std(flow_samples_flat, axis=0)

    return {
        "dataset_global_mean": ds_mean,
        "dataset_global_std": ds_std,
        "flow_global_mean": flow_mean,
        "flow_global_std": flow_std,
        "global_mean_abs_error": np.abs(ds_mean - flow_mean),
        "global_std_abs_error": np.abs(ds_std - flow_std),
        "global_std_ratio": flow_std / np.maximum(ds_std, 1e-8),
    }


def print_comparison_report(results, dim_names=None):
    """Print a formatted comparison report.

    Args:
        results: Results dictionary from compare_statistics.
        dim_names: Optional list of dimension names.
    """
    obs_dim = len(results["avg_empirical_mean"])
    if dim_names is None:
        dim_names = [f"dim_{i}" for i in range(obs_dim)]

    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON: Dataset vs Flow Model")
    print("=" * 70)

    # Mean comparison
    print(f"\n{'Dim':<8} {'Emp Mean':>10} {'Flow Mean':>10} {'Abs Err':>10}")
    print("-" * 40)
    for i in range(obs_dim):
        print(f"{dim_names[i]:<8} {results['avg_empirical_mean'][i]:>10.4f} "
              f"{results['avg_flow_mean'][i]:>10.4f} {results['mean_abs_error'][i]:>10.4f}")

    # Std comparison
    print(f"\n{'Dim':<8} {'Emp Std':>10} {'Flow Std':>10} {'Ratio':>10}")
    print("-" * 40)
    for i in range(obs_dim):
        print(f"{dim_names[i]:<8} {results['avg_empirical_std'][i]:>10.4f} "
              f"{results['avg_flow_std'][i]:>10.4f} {results['std_ratio'][i]:>10.4f}")

    # Summary
    print(f"\nOverall Mean Abs Error: {np.mean(results['mean_abs_error']):.4f}")
    print(f"Overall Std Ratio (avg): {np.mean(results['std_ratio']):.4f} (ideal: 1.0)")
    print("=" * 70)


if __name__ == "__main__":
    # Quick test with synthetic data
    rng = np.random.RandomState(42)
    num_pairs = 10
    k = 50
    obs_dim = 11

    emp = rng.randn(num_pairs, k, obs_dim).astype(np.float32)
    flow = emp + 0.1 * rng.randn(num_pairs, k, obs_dim).astype(np.float32)

    results = compare_statistics(emp, flow)
    print_comparison_report(results)
