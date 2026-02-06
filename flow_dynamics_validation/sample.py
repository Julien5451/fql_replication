"""Conditional sampling from the trained flow model.

Given (s, a), generates s' samples by integrating the learned velocity field
from t=0 (noise) to t=1 (data).

Supports:
  - Deterministic ODE (Euler) sampling  [noise_scale=0]
  - Stochastic SDE sampling             [noise_scale>0]
  - Temperature scaling of base noise   [temperature]
  - RK4 integration
"""

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Internal sampler factories (JIT-compiled, created once per config)
# ---------------------------------------------------------------------------

def _make_euler_sampler(apply_fn, num_steps):
    """Create a JIT-compiled *deterministic* Euler sampler.

    Args:
        apply_fn: Model apply function.
        num_steps: Number of integration steps.

    Returns:
        JIT-compiled function (params, condition, x_0) -> x_final.
    """
    dt = 1.0 / num_steps

    @jax.jit
    def euler_integrate(params, condition, x_0):
        def step_fn(x, step_idx):
            t = jnp.full((x.shape[0], 1), step_idx * dt)
            v = apply_fn({"params": params}, x, t, condition)
            x_new = x + v * dt
            return x_new, None

        x_final, _ = jax.lax.scan(step_fn, x_0, jnp.arange(num_steps))
        return x_final

    return euler_integrate


def _make_sde_sampler(apply_fn, num_steps, noise_scale):
    """Create a JIT-compiled *stochastic* SDE-Euler sampler.

    At each step the update is:
        x_{t+dt} = x_t + v(x_t, t, cond) * dt + noise_scale * sqrt(dt) * z_t

    where z_t ~ N(0, I) is fresh noise at every step, ensuring that
    independent calls with different RNG keys produce diverse samples.

    Args:
        apply_fn: Model apply function.
        num_steps: Number of integration steps.
        noise_scale: Standard-deviation of injected noise per sqrt(dt).

    Returns:
        JIT-compiled function (params, condition, x_0, rng) -> x_final.
    """
    dt = 1.0 / num_steps
    sqrt_dt = jnp.sqrt(dt)

    @jax.jit
    def sde_integrate(params, condition, x_0, rng):
        def step_fn(carry, step_idx):
            x, rng = carry
            rng, noise_rng = jax.random.split(rng)
            t = jnp.full((x.shape[0], 1), step_idx * dt)
            v = apply_fn({"params": params}, x, t, condition)
            z = jax.random.normal(noise_rng, x.shape)
            x_new = x + v * dt + noise_scale * sqrt_dt * z
            return (x_new, rng), None

        (x_final, _), _ = jax.lax.scan(step_fn, (x_0, rng), jnp.arange(num_steps))
        return x_final

    return sde_integrate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def euler_sample_fast(model, params, observations, actions, rng,
                      num_samples=50, num_steps=50,
                      temperature=1.0, noise_scale=0.0):
    """Sample next states via Euler (ODE or SDE) integration.

    Each of the *num_samples* draws for a given (s, a) starts from an
    **independent** noise vector and (when noise_scale > 0) receives
    independent per-step noise, so the samples are fully independent
    conditioned on (s, a).

    Args:
        model: Flow model (VelocityField).
        params: Trained model parameters.
        observations: (B, obs_dim).
        actions: (B, act_dim).
        rng: JAX PRNG key.
        num_samples: Samples per (s, a).
        num_steps: Integration steps.
        temperature: Scaling of base noise  x_0 ~ N(0, temperature^2 I).
        noise_scale: Per-step SDE noise magnitude. 0 = deterministic ODE.

    Returns:
        Sampled s' of shape (B, num_samples, obs_dim).
    """
    batch_size = observations.shape[0]
    obs_dim = observations.shape[1]
    condition = jnp.concatenate([observations, actions], axis=-1)

    stochastic = (noise_scale > 0.0)
    if stochastic:
        sampler_fn = _make_sde_sampler(model.apply, num_steps, noise_scale)
    else:
        sampler_fn = _make_euler_sampler(model.apply, num_steps)

    all_samples = []
    for _ in range(num_samples):
        rng, noise_rng, step_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(noise_rng, (batch_size, obs_dim)) * temperature
        if stochastic:
            x_final = sampler_fn(params, condition, x_0, step_rng)
        else:
            x_final = sampler_fn(params, condition, x_0)
        all_samples.append(x_final)

    return jnp.stack(all_samples, axis=1)


def euler_sample(model, params, observations, actions, rng,
                 num_samples=1, num_steps=50,
                 temperature=1.0, noise_scale=0.0):
    """Non-JIT (Python-loop) Euler sampler – mainly for debugging.

    Same interface as euler_sample_fast but uses Python loops for
    the ODE steps instead of jax.lax.scan.
    """
    batch_size = observations.shape[0]
    obs_dim = observations.shape[1]
    dt = 1.0 / num_steps
    sqrt_dt = jnp.sqrt(dt)
    condition = jnp.concatenate([observations, actions], axis=-1)

    all_samples = []
    for _ in range(num_samples):
        rng, noise_rng = jax.random.split(rng)
        x = jax.random.normal(noise_rng, (batch_size, obs_dim)) * temperature

        for step in range(num_steps):
            t = jnp.full((batch_size, 1), step * dt)
            v = model.apply({"params": params}, x, t, condition)
            x = x + v * dt
            if noise_scale > 0:
                rng, z_rng = jax.random.split(rng)
                z = jax.random.normal(z_rng, x.shape)
                x = x + noise_scale * sqrt_dt * z

        all_samples.append(x)

    return jnp.stack(all_samples, axis=1)


def rk4_sample(model, params, observations, actions, rng,
               num_samples=1, num_steps=50,
               temperature=1.0):
    """Sample using 4th-order Runge-Kutta integration (deterministic).

    Args:
        model: Flow model.
        params: Trained model parameters.
        observations: (B, obs_dim).
        actions: (B, act_dim).
        rng: JAX random key.
        num_samples: Samples per (s, a).
        num_steps: RK4 steps.
        temperature: Base noise scaling.

    Returns:
        Sampled s' of shape (B, num_samples, obs_dim).
    """
    batch_size = observations.shape[0]
    obs_dim = observations.shape[1]
    dt = 1.0 / num_steps
    condition = jnp.concatenate([observations, actions], axis=-1)

    def v_fn(x, t_scalar):
        t = jnp.full((batch_size, 1), t_scalar)
        return model.apply({"params": params}, x, t, condition)

    all_samples = []
    for _ in range(num_samples):
        rng, noise_rng = jax.random.split(rng)
        x = jax.random.normal(noise_rng, (batch_size, obs_dim)) * temperature

        for step in range(num_steps):
            t = step * dt
            k1 = v_fn(x, t)
            k2 = v_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = v_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = v_fn(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        all_samples.append(x)

    return jnp.stack(all_samples, axis=1)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from flow_model import create_flow_model

    rng = jax.random.PRNGKey(0)
    obs_dim, act_dim = 11, 3
    model = create_flow_model(obs_dim, act_dim)

    rng, init_rng = jax.random.split(rng)
    dummy_x = jnp.ones((2, obs_dim))
    dummy_t = jnp.ones((2, 1))
    dummy_cond = jnp.ones((2, obs_dim + act_dim))
    params = model.init(init_rng, dummy_x, dummy_t, dummy_cond)["params"]

    obs = jnp.ones((4, obs_dim))
    act = jnp.ones((4, act_dim))

    # Deterministic ODE
    rng, sr = jax.random.split(rng)
    s1 = euler_sample_fast(model, params, obs, act, sr,
                           num_samples=10, num_steps=20)
    print(f"ODE samples shape: {s1.shape}")

    # SDE with noise
    rng, sr = jax.random.split(rng)
    s2 = euler_sample_fast(model, params, obs, act, sr,
                           num_samples=10, num_steps=20,
                           temperature=1.0, noise_scale=0.1)
    print(f"SDE samples shape: {s2.shape}")
    print(f"ODE sample std (dim 0): {float(s1[:, :, 0].std()):.4f}")
    print(f"SDE sample std (dim 0): {float(s2[:, :, 0].std()):.4f}")
