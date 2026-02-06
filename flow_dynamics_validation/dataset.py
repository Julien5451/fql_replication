"""Dataset loading and preparation for D4RL offline RL datasets.

Loads transition data (s, a, s') from D4RL HDF5 files and prepares
evaluation batches for flow model validation.
"""

import os
import urllib.request

import h5py
import numpy as np


# D4RL dataset URLs (v2 gym mujoco)
D4RL_URLS = {
    "hopper-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5",
    "halfcheetah-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5",
    "walker2d-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5",
    "hopper-medium-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_expert-v2.hdf5",
    "halfcheetah-medium-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_expert-v2.hdf5",
}


def download_dataset(env_name, data_dir="data"):
    """Download D4RL dataset HDF5 file if not already present.

    Args:
        env_name: Name of the D4RL environment (e.g. 'hopper-medium-v2').
        data_dir: Directory to store downloaded files.

    Returns:
        Path to the downloaded HDF5 file.
    """
    os.makedirs(data_dir, exist_ok=True)
    filename = f"{env_name}.hdf5"
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        print(f"Dataset already exists at {filepath}")
        return filepath

    url = D4RL_URLS.get(env_name)
    if url is None:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(D4RL_URLS.keys())}")

    print(f"Downloading {env_name} dataset from {url}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"Downloaded to {filepath}")
    return filepath


def load_dataset(env_name, data_dir="data"):
    """Load D4RL dataset from HDF5 file.

    Args:
        env_name: Name of the D4RL environment.
        data_dir: Directory containing dataset files.

    Returns:
        Dictionary with keys 'observations', 'actions', 'next_observations',
        'rewards', 'terminals'. All arrays are float32.
    """
    filepath = download_dataset(env_name, data_dir)

    with h5py.File(filepath, "r") as f:
        dataset = {
            "observations": f["observations"][:].astype(np.float32),
            "actions": f["actions"][:].astype(np.float32),
            "next_observations": f["next_observations"][:].astype(np.float32),
            "rewards": f["rewards"][:].astype(np.float32),
            "terminals": f["terminals"][:].astype(np.float32),
        }

    n = len(dataset["observations"])
    obs_dim = dataset["observations"].shape[1]
    act_dim = dataset["actions"].shape[1]
    print(f"Loaded {env_name}: {n} transitions, obs_dim={obs_dim}, act_dim={act_dim}")

    return dataset


def compute_normalization_stats(dataset):
    """Compute mean and std for observations, actions, and next_observations.

    Args:
        dataset: Dictionary with 'observations', 'actions', 'next_observations'.

    Returns:
        Dictionary with normalization statistics.
    """
    obs = dataset["observations"]
    act = dataset["actions"]
    next_obs = dataset["next_observations"]

    stats = {
        "obs_mean": obs.mean(axis=0),
        "obs_std": obs.std(axis=0) + 1e-6,
        "act_mean": act.mean(axis=0),
        "act_std": act.std(axis=0) + 1e-6,
        "next_obs_mean": next_obs.mean(axis=0),
        "next_obs_std": next_obs.std(axis=0) + 1e-6,
    }
    return stats


def normalize_dataset(dataset, stats):
    """Normalize dataset using precomputed statistics.

    Args:
        dataset: Raw dataset dictionary.
        stats: Normalization statistics from compute_normalization_stats.

    Returns:
        Normalized dataset dictionary.
    """
    return {
        "observations": (dataset["observations"] - stats["obs_mean"]) / stats["obs_std"],
        "actions": (dataset["actions"] - stats["act_mean"]) / stats["act_std"],
        "next_observations": (dataset["next_observations"] - stats["next_obs_mean"]) / stats["next_obs_std"],
        "rewards": dataset["rewards"],
        "terminals": dataset["terminals"],
    }


def unnormalize_next_obs(next_obs_normalized, stats):
    """Unnormalize next observations back to original scale.

    Args:
        next_obs_normalized: Normalized next observations.
        stats: Normalization statistics.

    Returns:
        Unnormalized next observations.
    """
    return next_obs_normalized * stats["next_obs_std"] + stats["next_obs_mean"]


def select_evaluation_batch(dataset, num_pairs=100, seed=42):
    """Select a fixed batch of (s, a) pairs for evaluation.

    For each selected (s, a) pair, also collects all matching s' from dataset
    within a neighborhood (since exact matches are rare in continuous spaces).

    Instead, we select specific indices and use the corresponding s' as
    the 'empirical' ground truth.

    Args:
        dataset: Normalized dataset dictionary.
        num_pairs: Number of (s, a) pairs to select.
        seed: Random seed for reproducibility.

    Returns:
        eval_batch: Dictionary with 'observations', 'actions', 'next_observations',
                     and 'indices' for the selected pairs.
    """
    rng = np.random.RandomState(seed)
    n = len(dataset["observations"])
    indices = rng.choice(n, size=num_pairs, replace=False)

    eval_batch = {
        "observations": dataset["observations"][indices],
        "actions": dataset["actions"][indices],
        "next_observations": dataset["next_observations"][indices],
        "indices": indices,
    }
    return eval_batch


def collect_neighborhood_samples(dataset, eval_batch, k_neighbors=50, seed=42):
    """For each eval (s, a), find the k nearest (s, a) pairs in the dataset
    and collect their s' as empirical samples.

    This gives us a distribution of s' for each query (s, a) to compare against.

    Args:
        dataset: Full normalized dataset.
        eval_batch: Evaluation batch from select_evaluation_batch.
        k_neighbors: Number of nearest neighbors to find.
        seed: Random seed.

    Returns:
        empirical_next_obs: Array of shape (num_pairs, k_neighbors, obs_dim)
            containing empirical s' samples for each eval (s, a).
    """
    obs = dataset["observations"]
    act = dataset["actions"]
    next_obs = dataset["next_observations"]

    # Concatenate (s, a) for distance computation
    sa = np.concatenate([obs, act], axis=-1)
    eval_sa = np.concatenate([eval_batch["observations"], eval_batch["actions"]], axis=-1)

    num_pairs = len(eval_sa)
    obs_dim = next_obs.shape[1]
    empirical_next_obs = np.zeros((num_pairs, k_neighbors, obs_dim), dtype=np.float32)

    for i in range(num_pairs):
        # Compute distances from eval (s,a) to all dataset (s,a)
        dists = np.linalg.norm(sa - eval_sa[i:i+1], axis=-1)
        # Get k nearest neighbor indices
        nn_indices = np.argpartition(dists, k_neighbors)[:k_neighbors]
        empirical_next_obs[i] = next_obs[nn_indices]

    return empirical_next_obs


def precompute_knn_std(dataset, k=50, chunk_size=1000):
    """Precompute per-datapoint KNN empirical std of next_observations.

    For every transition i in the dataset, finds its k nearest neighbours
    in (s, a)-space and stores std(next_obs[neighbours]) as the target
    conditional spread.  This table is used by the variance-matching loss
    during training.

    Args:
        dataset: Normalised dataset dictionary.
        k: Number of nearest neighbours.
        chunk_size: Number of query points to process at once (controls memory).

    Returns:
        knn_std: Array of shape (N, obs_dim) with per-point KNN std.
    """
    obs = dataset["observations"]
    act = dataset["actions"]
    next_obs = dataset["next_observations"]

    sa = np.concatenate([obs, act], axis=-1)  # (N, D)
    N = len(sa)
    obs_dim = next_obs.shape[1]

    sq_norms = np.sum(sa ** 2, axis=1)  # (N,)
    knn_std = np.zeros((N, obs_dim), dtype=np.float32)

    print(f"  Precomputing KNN std (N={N}, k={k}) ...")
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = sa[start:end]  # (C, D)

        # Squared distances: (C, N) via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        chunk_sq_norms = np.sum(chunk ** 2, axis=1, keepdims=True)  # (C, 1)
        dists_sq = chunk_sq_norms + sq_norms[np.newaxis, :] - 2 * (chunk @ sa.T)

        # k+1 because self is included (dist=0)
        nn_indices = np.argpartition(dists_sq, k + 1, axis=1)[:, : k + 1]  # (C, k+1)
        nn_next_obs = next_obs[nn_indices]  # (C, k+1, obs_dim)
        knn_std[start:end] = np.std(nn_next_obs, axis=1)  # (C, obs_dim)

        if (end // chunk_size) % 10 == 0 or end == N:
            print(f"    ... {end}/{N}")

    return knn_std


if __name__ == "__main__":
    # Quick test
    dataset = load_dataset("hopper-medium-v2", data_dir="data")
    stats = compute_normalization_stats(dataset)
    norm_dataset = normalize_dataset(dataset, stats)
    eval_batch = select_evaluation_batch(norm_dataset, num_pairs=10)
    print(f"Eval batch shapes: obs={eval_batch['observations'].shape}, "
          f"act={eval_batch['actions'].shape}, "
          f"next_obs={eval_batch['next_observations'].shape}")
    empirical = collect_neighborhood_samples(norm_dataset, eval_batch, k_neighbors=50)
    print(f"Empirical samples shape: {empirical.shape}")
