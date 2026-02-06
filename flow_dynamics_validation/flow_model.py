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
