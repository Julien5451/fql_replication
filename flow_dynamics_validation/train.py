"""Training loop for the conditional flow matching dynamics model.

Trains v_theta(x_t, t, s, a) using the CFM loss on offline RL transitions.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from flow_model import create_flow_model, flow_matching_loss


def create_train_state(rng, obs_dim, act_dim, learning_rate=3e-4,
                       hidden_dims=(256, 256, 256), time_embed_dim=64,
                       layer_norm=True):
    """Create model and optimizer state.

    Args:
        rng: JAX random key.
        obs_dim: Observation dimension.
        act_dim: Action dimension.
        learning_rate: Learning rate for Adam optimizer.
        hidden_dims: Hidden dimensions for the velocity field MLP.
        time_embed_dim: Time embedding dimension.
        layer_norm: Whether to use layer normalization.

    Returns:
        Tuple of (model, params, opt_state, optimizer).
    """
    model = create_flow_model(obs_dim, act_dim, hidden_dims, time_embed_dim, layer_norm)

    # Initialize with dummy inputs
    dummy_x = jnp.ones((1, obs_dim))
    dummy_t = jnp.ones((1, 1))
    dummy_cond = jnp.ones((1, obs_dim + act_dim))
    params = model.init(rng, dummy_x, dummy_t, dummy_cond)["params"]

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model created: {num_params} parameters")

    return model, params, opt_state, optimizer


def make_train_step(apply_fn, optimizer):
    """Create a JIT-compiled training step function.

    Args:
        apply_fn: Model apply function (closed over, not traced).
        optimizer: Optax optimizer (closed over, not traced).

    Returns:
        JIT-compiled train_step function.
    """
    @jax.jit
    def train_step(params, opt_state, batch, rng):
        loss, grads = jax.value_and_grad(flow_matching_loss)(params, apply_fn, batch, rng)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    return train_step


def train_flow_model(dataset, num_epochs=100, batch_size=256, learning_rate=3e-4,
                     hidden_dims=(256, 256, 256), time_embed_dim=64,
                     layer_norm=True, seed=0, log_interval=10):
    """Train the conditional flow matching dynamics model.

    Args:
        dataset: Normalized dataset dict with 'observations', 'actions', 'next_observations'.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        hidden_dims: Hidden dimensions for velocity field.
        time_embed_dim: Time embedding dimension.
        layer_norm: Whether to use layer normalization.
        seed: Random seed.
        log_interval: Print loss every this many epochs.

    Returns:
        Tuple of (model, trained_params, losses).
    """
    obs = dataset["observations"]
    act = dataset["actions"]
    next_obs = dataset["next_observations"]

    n = len(obs)
    obs_dim = obs.shape[1]
    act_dim = act.shape[1]

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    # Create model and optimizer
    model, params, opt_state, optimizer = create_train_state(
        init_rng, obs_dim, act_dim, learning_rate, hidden_dims, time_embed_dim, layer_norm
    )

    # Create JIT-compiled train step
    train_step = make_train_step(model.apply, optimizer)

    # Precompute number of batches per epoch
    num_batches = max(n // batch_size, 1)
    losses = []

    print(f"\nTraining for {num_epochs} epochs, {num_batches} batches/epoch, batch_size={batch_size}")
    print(f"Dataset size: {n}, obs_dim={obs_dim}, act_dim={act_dim}")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_losses = []

        # Shuffle dataset
        rng, shuffle_rng = jax.random.split(rng)
        perm = np.random.permutation(n)

        for batch_idx in range(num_batches):
            # Get batch indices
            start = batch_idx * batch_size
            end = min(start + batch_size, n)
            idx = perm[start:end]

            batch = {
                "observations": jnp.array(obs[idx]),
                "actions": jnp.array(act[idx]),
                "next_observations": jnp.array(next_obs[idx]),
            }

            rng, step_rng = jax.random.split(rng)
            params, opt_state, loss = train_step(params, opt_state, batch, step_rng)
            epoch_losses.append(float(loss))

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch + 1:4d}/{num_epochs} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete in {total_time:.1f}s. Final loss: {losses[-1]:.6f}")

    return model, params, losses


if __name__ == "__main__":
    # Quick test with synthetic data
    n = 1000
    obs_dim, act_dim = 11, 3
    rng = np.random.RandomState(42)
    dataset = {
        "observations": rng.randn(n, obs_dim).astype(np.float32),
        "actions": rng.randn(n, act_dim).astype(np.float32),
        "next_observations": rng.randn(n, obs_dim).astype(np.float32),
    }
    model, params, losses = train_flow_model(dataset, num_epochs=20, batch_size=128, log_interval=5)
    print(f"Final loss: {losses[-1]:.6f}")
