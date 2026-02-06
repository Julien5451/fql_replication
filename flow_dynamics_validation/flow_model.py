"""Conditional Flow Matching model for dynamics prediction.

Implements a conditional flow model p_psi(s' | s, a) using the
Flow Matching (FM) / Conditional Flow Matching (CFM) framework.

The vector field v_theta(x_t, t, s, a) is parameterized by an MLP
and trained with the CFM loss to transport samples from N(0, I) to
the target distribution of s'.

Reference: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
"""

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the time variable.

    Attributes:
        embed_dim: Dimension of the embedding.
    """

    embed_dim: int = 64

    @nn.compact
    def __call__(self, t):
        """Embed time t into a sinusoidal representation.

        Args:
            t: Time values of shape (..., 1) in [0, 1].

        Returns:
            Embedding of shape (..., embed_dim).
        """
        half_dim = self.embed_dim // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half_dim) / half_dim)
        # t shape: (..., 1), freqs shape: (half_dim,)
        args = t * freqs
        embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        return embedding


class VelocityField(nn.Module):
    """MLP-based velocity field for flow matching.

    Predicts the velocity v(x_t, t, s, a) that transports noise to data.

    Attributes:
        hidden_dims: Tuple of hidden layer dimensions.
        output_dim: Dimension of the output (same as s' dimension).
        time_embed_dim: Dimension of the time embedding.
        layer_norm: Whether to use layer normalization.
    """

    hidden_dims: Sequence[int] = (256, 256, 256)
    output_dim: int = 11  # Will be set based on obs_dim
    time_embed_dim: int = 64
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x_t, t, condition):
        """Predict velocity at (x_t, t) conditioned on (s, a).

        Args:
            x_t: Noisy state of shape (batch, obs_dim).
            t: Time of shape (batch, 1) in [0, 1].
            condition: Concatenation of (s, a) of shape (batch, obs_dim + act_dim).

        Returns:
            Predicted velocity of shape (batch, obs_dim).
        """
        # Time embedding
        time_emb = SinusoidalTimeEmbedding(self.time_embed_dim)(t)

        # Concatenate all inputs: [x_t, time_embedding, condition]
        inputs = jnp.concatenate([x_t, time_emb, condition], axis=-1)

        # MLP
        x = inputs
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, kernel_init=default_init())(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.gelu(x)

        # Output layer
        v = nn.Dense(self.output_dim, kernel_init=default_init(0.01))(x)
        return v


def create_flow_model(obs_dim, act_dim, hidden_dims=(256, 256, 256), time_embed_dim=64, layer_norm=True):
    """Create a VelocityField model for dynamics prediction.

    Args:
        obs_dim: Dimension of observation / next state.
        act_dim: Dimension of action.
        hidden_dims: Hidden layer dimensions for the MLP.
        time_embed_dim: Dimension of the sinusoidal time embedding.
        layer_norm: Whether to use layer normalization.

    Returns:
        VelocityField model instance.
    """
    model = VelocityField(
        hidden_dims=hidden_dims,
        output_dim=obs_dim,
        time_embed_dim=time_embed_dim,
        layer_norm=layer_norm,
    )
    return model


def flow_matching_loss(params, apply_fn, batch, rng):
    """Compute the Conditional Flow Matching (CFM) loss.

    Given a batch of (s, a, s') transitions:
    1. Sample noise x_0 ~ N(0, I)
    2. Sample time t ~ U(0, 1)
    3. Compute interpolant x_t = (1-t)*x_0 + t*x_1 where x_1 = s'
    4. Compute target velocity u_t = x_1 - x_0
    5. Predict velocity v_theta(x_t, t, [s, a])
    6. Loss = ||v_theta - u_t||^2

    Args:
        params: Model parameters.
        apply_fn: Model apply function.
        batch: Dictionary with 'observations', 'actions', 'next_observations'.
        rng: JAX random key.

    Returns:
        Scalar loss value.
    """
    obs = batch["observations"]  # (B, obs_dim) - s
    act = batch["actions"]  # (B, act_dim) - a
    x_1 = batch["next_observations"]  # (B, obs_dim) - s'

    batch_size = obs.shape[0]
    obs_dim = x_1.shape[1]

    rng_noise, rng_time = jax.random.split(rng)

    # Sample noise from base distribution
    x_0 = jax.random.normal(rng_noise, (batch_size, obs_dim))

    # Sample time uniformly in [0, 1]
    t = jax.random.uniform(rng_time, (batch_size, 1))

    # Compute interpolant: x_t = (1 - t) * x_0 + t * x_1
    x_t = (1.0 - t) * x_0 + t * x_1

    # Target velocity: u_t = x_1 - x_0
    u_t = x_1 - x_0

    # Condition on (s, a)
    condition = jnp.concatenate([obs, act], axis=-1)

    # Predict velocity
    v_pred = apply_fn({"params": params}, x_t, t, condition)

    # MSE loss
    loss = jnp.mean((v_pred - u_t) ** 2)

    return loss


def variance_matching_loss(params, apply_fn, obs, act, knn_std,
                           rng, num_samples=64, num_ode_steps=20):
    """Compute the variance / moment-matching loss.

    For each (s, a) in the mini-batch:
      1. Draw M noise vectors  x_0^{(m)} ~ N(0, I)
      2. Integrate each through the learned ODE to get s'^{(m)}
      3. Compute  std_model = std_over_m( s'^{(m)} )   per dimension
      4. Compare with the precomputed KNN empirical std:
             L = mean_over_batch( | log(std_model + eps) - log(std_emp + eps) |_1 )

    Because `jax.lax.scan` is differentiable, gradients flow through
    the ODE integration back into the velocity-field parameters.

    Args:
        params: Model parameters (traced by jax.grad).
        apply_fn: Model apply function (closed over, not traced).
        obs: Observations of shape (B, obs_dim).
        act: Actions of shape (B, act_dim).
        knn_std: Precomputed empirical KNN std, shape (B, obs_dim).
        rng: JAX PRNG key.
        num_samples: M – number of ODE samples per (s, a).
        num_ode_steps: Integration steps for the inner ODE.

    Returns:
        Scalar loss.
    """
    B = obs.shape[0]
    obs_dim = obs.shape[1]
    dt = 1.0 / num_ode_steps

    condition = jnp.concatenate([obs, act], axis=-1)  # (B, cond_dim)

    # Expand condition for M samples:  (B, cond) -> (B*M, cond)
    cond_expanded = jnp.repeat(condition, num_samples, axis=0)   # (B*M, cond_dim)

    # Initial noise: (B*M, obs_dim)
    noise = jax.random.normal(rng, (B * num_samples, obs_dim))

    # ---- ODE integration (differentiable via scan) ----
    def step_fn(x, step_idx):
        t = jnp.full((x.shape[0], 1), step_idx * dt)
        v = apply_fn({"params": params}, x, t, cond_expanded)
        return x + v * dt, None

    x_final, _ = jax.lax.scan(step_fn, noise, jnp.arange(num_ode_steps))
    # x_final: (B*M, obs_dim)

    # Reshape -> (B, M, obs_dim)
    samples = x_final.reshape(B, num_samples, obs_dim)

    # Per (s,a) per-dim std of model samples
    std_model = jnp.std(samples, axis=1)  # (B, obs_dim)

    # Log-space L1 loss
    eps = 1e-6
    loss = jnp.mean(jnp.abs(
        jnp.log(std_model + eps) - jnp.log(knn_std + eps)
    ))

    return loss


def combined_loss(params, apply_fn, batch, knn_std_batch, rng,
                  w_std=0.0, std_batch_size=16,
                  num_model_samples=64, num_ode_steps_std=20):
    """Base CFM loss + optional variance-matching loss.

    Args:
        params: Model parameters.
        apply_fn: Model apply function.
        batch: Dict with 'observations', 'actions', 'next_observations'.
        knn_std_batch: Precomputed KNN std for this batch, (B, obs_dim).
        rng: JAX PRNG key.
        w_std: Weight on the variance-matching term. 0 disables it.
        std_batch_size: How many items from the batch to use for the
            (expensive) variance-matching term.
        num_model_samples: M – ODE samples per query for variance loss.
        num_ode_steps_std: Integration steps for the variance ODE.

    Returns:
        (total_loss, info_dict)
    """
    rng_base, rng_std = jax.random.split(rng)

    base_loss = flow_matching_loss(params, apply_fn, batch, rng_base)

    if w_std <= 0.0:
        return base_loss, {
            "base_loss": base_loss,
            "std_loss": jnp.float32(0.0),
            "total_loss": base_loss,
        }

    # Sub-sample the batch for the expensive variance term
    obs_sub = batch["observations"][:std_batch_size]
    act_sub = batch["actions"][:std_batch_size]
    knn_sub = knn_std_batch[:std_batch_size]

    std_loss = variance_matching_loss(
        params, apply_fn, obs_sub, act_sub, knn_sub,
        rng_std, num_model_samples, num_ode_steps_std,
    )

    total_loss = base_loss + w_std * std_loss
    return total_loss, {
        "base_loss": base_loss,
        "std_loss": std_loss,
        "total_loss": total_loss,
    }


if __name__ == "__main__":
    # Quick test
    rng = jax.random.PRNGKey(0)
    obs_dim, act_dim = 11, 3
    model = create_flow_model(obs_dim, act_dim)

    # Initialize
    rng, init_rng = jax.random.split(rng)
    dummy_x = jnp.ones((2, obs_dim))
    dummy_t = jnp.ones((2, 1))
    dummy_cond = jnp.ones((2, obs_dim + act_dim))
    params = model.init(init_rng, dummy_x, dummy_t, dummy_cond)["params"]

    # Forward pass
    v = model.apply({"params": params}, dummy_x, dummy_t, dummy_cond)
    print(f"Velocity field output shape: {v.shape}")
    print(f"Number of parameters: {sum(p.size for p in jax.tree_util.tree_leaves(params))}")
