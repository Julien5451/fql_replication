"""Statistical comparison between empirical and flow model samples.

Provides three levels of comparison:
1. Pointwise prediction: Is the flow model's mean prediction close to the true s'?
2. Neighborhood (conditional): Does the flow model's conditional distribution
   match the KNN-based empirical distribution?
3. Global (marginal): Does the overall distribution of flow-generated s'
   match the marginal distribution of s' in the dataset?
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


def compute_pointwise_accuracy(true_next_obs, flow_samples):
    """Compute pointwise prediction accuracy.

    For each (s, a, s') triple, check if the flow model's mean prediction
    (averaged over samples) is close to the true s'. This is the most
    meaningful metric for near-deterministic transitions.

    Args:
        true_next_obs: Ground-truth s' of shape (num_pairs, obs_dim).
        flow_samples: Flow samples of shape (num_pairs, num_samples, obs_dim).

    Returns:
        Dictionary with pointwise accuracy metrics.
    """
    flow_mean = np.mean(flow_samples, axis=1)  # (num_pairs, obs_dim)
    errors = true_next_obs - flow_mean  # (num_pairs, obs_dim)

    per_dim_mae = np.mean(np.abs(errors), axis=0)  # (obs_dim,)
    per_dim_rmse = np.sqrt(np.mean(errors ** 2, axis=0))  # (obs_dim,)
    per_pair_mse = np.mean(errors ** 2, axis=1)  # (num_pairs,)

    return {
        "pointwise_per_dim_mae": per_dim_mae,
        "pointwise_per_dim_rmse": per_dim_rmse,
        "pointwise_avg_mae": np.mean(per_dim_mae),
        "pointwise_avg_rmse": np.mean(per_dim_rmse),
        "pointwise_per_pair_mse": per_pair_mse,
        "pointwise_flow_mean": flow_mean,
        "pointwise_true": true_next_obs,
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


def compare_global_statistics(dataset_next_obs, flow_samples_all):
    """Compare global (marginal) statistics: dataset s' vs flow-generated s'.

    Given a set of (s, a) pairs sampled from the dataset, the flow model
    generates s' for each. We compare the marginal distribution of these
    generated s' against the dataset's marginal distribution of s'.

    Args:
        dataset_next_obs: All s' from dataset, shape (N, obs_dim).
        flow_samples_all: Flow samples flattened, shape (M, obs_dim).

    Returns:
        Dictionary with global comparison results.
    """
    ds_mean = np.mean(dataset_next_obs, axis=0)
    ds_std = np.std(dataset_next_obs, axis=0)

    flow_mean = np.mean(flow_samples_all, axis=0)
    flow_std = np.std(flow_samples_all, axis=0)

    return {
        "dataset_global_mean": ds_mean,
        "dataset_global_std": ds_std,
        "flow_global_mean": flow_mean,
        "flow_global_std": flow_std,
        "global_mean_abs_error": np.abs(ds_mean - flow_mean),
        "global_std_abs_error": np.abs(ds_std - flow_std),
        "global_std_ratio": flow_std / np.maximum(ds_std, 1e-8),
    }


def print_comparison_report(pointwise_results, neighbor_results, global_results, dim_names=None):
    """Print a formatted comparison report.

    Args:
        pointwise_results: Results from compute_pointwise_accuracy.
        neighbor_results: Results from compare_statistics.
        global_results: Results from compare_global_statistics.
        dim_names: Optional list of dimension names.
    """
    obs_dim = len(pointwise_results["pointwise_per_dim_mae"])
    if dim_names is None:
        dim_names = [f"dim_{i}" for i in range(obs_dim)]

    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON: Dataset vs Flow Model")
    print("=" * 70)

    # --- Pointwise prediction accuracy ---
    print("\n--- 1. POINTWISE PREDICTION (flow mean vs true s') ---")
    print(f"\n{'Dim':<8} {'MAE':>10} {'RMSE':>10}")
    print("-" * 30)
    for i in range(obs_dim):
        print(f"{dim_names[i]:<8} {pointwise_results['pointwise_per_dim_mae'][i]:>10.4f} "
              f"{pointwise_results['pointwise_per_dim_rmse'][i]:>10.4f}")
    print(f"\nAvg MAE: {pointwise_results['pointwise_avg_mae']:.4f}")
    print(f"Avg RMSE: {pointwise_results['pointwise_avg_rmse']:.4f}")

    # --- Neighbor-based conditional comparison ---
    print("\n--- 2. NEIGHBORHOOD (KNN) CONDITIONAL COMPARISON ---")
    print(f"\n{'Dim':<8} {'Emp Mean':>10} {'Flow Mean':>10} {'Abs Err':>10}")
    print("-" * 40)
    for i in range(obs_dim):
        print(f"{dim_names[i]:<8} {neighbor_results['avg_empirical_mean'][i]:>10.4f} "
              f"{neighbor_results['avg_flow_mean'][i]:>10.4f} "
              f"{neighbor_results['mean_abs_error'][i]:>10.4f}")

    print(f"\n{'Dim':<8} {'Emp Std':>10} {'Flow Std':>10} {'Ratio':>10}")
    print("-" * 40)
    for i in range(obs_dim):
        print(f"{dim_names[i]:<8} {neighbor_results['avg_empirical_std'][i]:>10.4f} "
              f"{neighbor_results['avg_flow_std'][i]:>10.4f} "
              f"{neighbor_results['std_ratio'][i]:>10.4f}")

    print(f"\nOverall Neighbor Mean Abs Error: {np.mean(neighbor_results['mean_abs_error']):.4f}")
    print(f"Overall Neighbor Std Ratio (avg): {np.mean(neighbor_results['std_ratio']):.4f} (ideal: 1.0)")

    # --- Global marginal comparison ---
    print("\n--- 3. GLOBAL MARGINAL COMPARISON ---")
    print(f"\n{'Dim':<8} {'DS Mean':>10} {'Flow Mean':>10} {'DS Std':>10} {'Flow Std':>10} {'Std Ratio':>10}")
    print("-" * 60)
    for i in range(obs_dim):
        print(f"{dim_names[i]:<8} {global_results['dataset_global_mean'][i]:>10.4f} "
              f"{global_results['flow_global_mean'][i]:>10.4f} "
              f"{global_results['dataset_global_std'][i]:>10.4f} "
              f"{global_results['flow_global_std'][i]:>10.4f} "
              f"{global_results['global_std_ratio'][i]:>10.4f}")

    print(f"\nGlobal Mean Abs Error (avg): {np.mean(global_results['global_mean_abs_error']):.4f}")
    print(f"Global Std Ratio (avg): {np.mean(global_results['global_std_ratio']):.4f} (ideal: 1.0)")
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
