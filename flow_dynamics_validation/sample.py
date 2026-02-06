"""Conditional sampling from the trained flow model.

Given (s, a), generates s' samples by integrating the learned velocity field
from t=0 (noise) to t=1 (data) using the Euler method or higher-order ODE solvers.
"""

import jax
import jax.numpy as jnp
import numpy as np


def euler_sample(model, params, observations, actions, rng,
                 num_samples=1, num_steps=50):
    """Sample next states using Euler integration of the flow ODE.

    For each (s, a), integrates:
        dx/dt = v_theta(x_t, t, [s, a])
    from t=0 to t=1, starting from x_0 ~ N(0, I).

    Args:
        model: Flow model (VelocityField).
        params: Trained model parameters.
        observations: Observations (s) of shape (B, obs_dim).
        actions: Actions (a) of shape (B, act_dim).
        rng: JAX random key.
        num_samples: Number of samples to generate per (s, a) pair.
        num_steps: Number of Euler integration steps.

    Returns:
        Sampled next states of shape (B, num_samples, obs_dim).
    """
    batch_size = observations.shape[0]
    obs_dim = observations.shape[1]
    dt = 1.0 / num_steps

    # Condition: concatenate (s, a)
    condition = jnp.concatenate([observations, actions], axis=-1)  # (B, obs_dim + act_dim)

    all_samples = []
    for k in range(num_samples):
        rng, noise_rng = jax.random.split(rng)
        # Start from noise
        x = jax.random.normal(noise_rng, (batch_size, obs_dim))

        # Euler integration from t=0 to t=1
        for step in range(num_steps):
            t = jnp.full((batch_size, 1), step * dt)
            v = model.apply({"params": params}, x, t, condition)
            x = x + v * dt

        all_samples.append(x)

    # Stack samples: (B, num_samples, obs_dim)
    samples = jnp.stack(all_samples, axis=1)
    return samples


@jax.jit
def _euler_sample_single(params, apply_fn, condition, x_0, num_steps):
    """JIT-compiled Euler integration for a single noise sample.

    Args:
        params: Model parameters.
        apply_fn: Model apply function.
        condition: Concatenated (s, a) of shape (B, cond_dim).
        x_0: Initial noise of shape (B, obs_dim).
        num_steps: Number of integration steps.

    Returns:
        Final samples of shape (B, obs_dim).
    """
    dt = 1.0 / num_steps

    def step_fn(x, step_idx):
        t = jnp.full((x.shape[0], 1), step_idx * dt)
        v = apply_fn({"params": params}, x, t, condition)
        x_new = x + v * dt
        return x_new, None

    x_final, _ = jax.lax.scan(step_fn, x_0, jnp.arange(num_steps))
    return x_final


def euler_sample_fast(model, params, observations, actions, rng,
                      num_samples=50, num_steps=50):
    """Faster sampling using jax.lax.scan for Euler integration.

    Args:
        model: Flow model.
        params: Trained model parameters.
        observations: Observations (s) of shape (B, obs_dim).
        actions: Actions (a) of shape (B, act_dim).
        rng: JAX random key.
        num_samples: Number of samples per (s, a).
        num_steps: Number of Euler steps.

    Returns:
        Sampled next states of shape (B, num_samples, obs_dim).
    """
    batch_size = observations.shape[0]
    obs_dim = observations.shape[1]

    condition = jnp.concatenate([observations, actions], axis=-1)

    all_samples = []
    for k in range(num_samples):
        rng, noise_rng = jax.random.split(rng)
        x_0 = jax.random.normal(noise_rng, (batch_size, obs_dim))
        x_final = _euler_sample_single(params, model.apply, condition, x_0, num_steps)
        all_samples.append(x_final)

    samples = jnp.stack(all_samples, axis=1)
    return samples


def rk4_sample(model, params, observations, actions, rng,
               num_samples=1, num_steps=50):
    """Sample using 4th-order Runge-Kutta integration (more accurate).

    Args:
        model: Flow model.
        params: Trained model parameters.
        observations: Observations (s) of shape (B, obs_dim).
        actions: Actions (a) of shape (B, act_dim).
        rng: JAX random key.
        num_samples: Number of samples per (s, a).
        num_steps: Number of RK4 steps.

    Returns:
        Sampled next states of shape (B, num_samples, obs_dim).
    """
    batch_size = observations.shape[0]
    obs_dim = observations.shape[1]
    dt = 1.0 / num_steps

    condition = jnp.concatenate([observations, actions], axis=-1)

    def v_fn(x, t_scalar):
        t = jnp.full((batch_size, 1), t_scalar)
        return model.apply({"params": params}, x, t, condition)

    all_samples = []
    for k in range(num_samples):
        rng, noise_rng = jax.random.split(rng)
        x = jax.random.normal(noise_rng, (batch_size, obs_dim))

        for step in range(num_steps):
            t = step * dt
            k1 = v_fn(x, t)
            k2 = v_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = v_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = v_fn(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        all_samples.append(x)

    samples = jnp.stack(all_samples, axis=1)
    return samples


if __name__ == "__main__":
    from flow_model import create_flow_model

    rng = jax.random.PRNGKey(0)
    obs_dim, act_dim = 11, 3
    model = create_flow_model(obs_dim, act_dim)

    # Initialize
    rng, init_rng = jax.random.split(rng)
    dummy_x = jnp.ones((2, obs_dim))
    dummy_t = jnp.ones((2, 1))
    dummy_cond = jnp.ones((2, obs_dim + act_dim))
    params = model.init(init_rng, dummy_x, dummy_t, dummy_cond)["params"]

    # Test sampling
    obs = jnp.ones((4, obs_dim))
    act = jnp.ones((4, act_dim))
    rng, sample_rng = jax.random.split(rng)
    samples = euler_sample_fast(model, params, obs, act, sample_rng, num_samples=10, num_steps=20)
    print(f"Samples shape: {samples.shape}")  # Should be (4, 10, 11)
