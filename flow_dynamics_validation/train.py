"""Training loop for the conditional flow matching dynamics model.

Supports:
  - Base CFM loss only                       (w_std = 0, default)
  - Base CFM + variance-matching loss        (w_std > 0)
  - Per-epoch std_ratio diagnostics          (std_eval_interval > 0)

All new parameters default to values that reproduce the original behaviour.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from flow_model import (
    create_flow_model,
    flow_matching_loss,
    combined_loss,
)
from sample import _make_euler_sampler


# ------------------------------------------------------------------
# Train-step factories
# ------------------------------------------------------------------

def make_train_step(apply_fn, optimizer):
    """JIT-compiled step for base-only training (w_std = 0).

    Backward-compatible with the original code.
    """
    @jax.jit
    def train_step(params, opt_state, batch, rng):
        loss, grads = jax.value_and_grad(flow_matching_loss)(
            params, apply_fn, batch, rng,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        info = {"base_loss": loss, "std_loss": jnp.float32(0.0), "total_loss": loss}
        return new_params, new_opt_state, loss, info
    return train_step


def make_combined_train_step(apply_fn, optimizer,
                             w_std, std_batch_size,
                             num_model_samples, num_ode_steps_std):
    """JIT-compiled step with base + variance-matching loss (w_std > 0).

    Non-static scalars are closed over so that the JIT signature only
    contains array leaves.
    """
    def _loss_fn(params, batch, knn_std_batch, rng):
        return combined_loss(
            params, apply_fn, batch, knn_std_batch, rng,
            w_std=w_std,
            std_batch_size=std_batch_size,
            num_model_samples=num_model_samples,
            num_ode_steps_std=num_ode_steps_std,
        )

    @jax.jit
    def train_step(params, opt_state, batch, knn_std_batch, rng):
        (loss, info), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            params, batch, knn_std_batch, rng,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, info
    return train_step


# ------------------------------------------------------------------
# Diagnostics helpers
# ------------------------------------------------------------------

def _compute_std_ratio_eval(model, params, eval_obs, eval_act, eval_knn_std,
                            rng, num_samples=30, num_steps=30):
    """Compute std_ratio = std_model / std_emp on a fixed eval set.

    Returns:
        per_dim_ratio: (obs_dim,)
        median_ratio: scalar
    """
    from sample import euler_sample_fast  # avoid circular at module level

    flow_samples = euler_sample_fast(
        model, params,
        jnp.array(eval_obs), jnp.array(eval_act), rng,
        num_samples=num_samples, num_steps=num_steps,
    )
    # flow_samples: (num_eval, num_samples, obs_dim)
    std_model = np.array(jnp.std(flow_samples, axis=1))   # (num_eval, obs_dim)
    avg_std_model = std_model.mean(axis=0)                 # (obs_dim,)
    avg_std_emp = eval_knn_std.mean(axis=0)                # (obs_dim,)

    eps = 1e-8
    per_dim_ratio = avg_std_model / np.maximum(avg_std_emp, eps)
    median_ratio = float(np.median(per_dim_ratio))
    return per_dim_ratio, median_ratio


# ------------------------------------------------------------------
# Model + optimiser creation
# ------------------------------------------------------------------

def create_train_state(rng, obs_dim, act_dim, learning_rate=3e-4,
                       hidden_dims=(256, 256, 256), time_embed_dim=64,
                       layer_norm=True):
    """Create model and optimiser state."""
    model = create_flow_model(obs_dim, act_dim, hidden_dims, time_embed_dim, layer_norm)

    dummy_x = jnp.ones((1, obs_dim))
    dummy_t = jnp.ones((1, 1))
    dummy_cond = jnp.ones((1, obs_dim + act_dim))
    params = model.init(rng, dummy_x, dummy_t, dummy_cond)["params"]

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model created: {num_params} parameters")

    return model, params, opt_state, optimizer


# ------------------------------------------------------------------
# Main training entry-point
# ------------------------------------------------------------------

def train_flow_model(
    dataset,
    num_epochs=100,
    batch_size=256,
    learning_rate=3e-4,
    hidden_dims=(256, 256, 256),
    time_embed_dim=64,
    layer_norm=True,
    seed=0,
    log_interval=10,
    # ---- variance-matching loss ----
    w_std=0.0,
    knn_std_all=None,
    std_batch_size=16,
    num_model_samples_train_std=64,
    num_ode_steps_std=20,
    # ---- diagnostics ----
    std_eval_interval=0,
    std_eval_obs=None,
    std_eval_act=None,
    std_eval_knn_std=None,
    plot_dir="plots",
):
    """Train the conditional flow matching dynamics model.

    New parameters (all have backward-compatible defaults):

    Args:
        w_std: Weight on the variance-matching loss term.  0 = disabled.
        knn_std_all: Precomputed KNN std for every dataset point,
            shape (N, obs_dim).  Required when w_std > 0.
        std_batch_size: Number of batch items used for the (expensive)
            variance-matching term.
        num_model_samples_train_std: M – number of ODE samples per query
            inside the variance-matching loss.
        num_ode_steps_std: Integration steps for the inner ODE in the
            variance loss.
        std_eval_interval: Evaluate std_ratio every N epochs (0 = off).
        std_eval_obs, std_eval_act, std_eval_knn_std: Fixed eval set for
            the std_ratio diagnostic.  Ignored when std_eval_interval = 0.
        plot_dir: Where to save diagnostic plots.

    Returns:
        (model, params, train_history)
        train_history is a dict:
            losses         – list[float], per-epoch avg total loss
            base_losses    – list[float]
            std_losses     – list[float]
            std_ratio_epochs  – list[int]   (epochs at which std_ratio was logged)
            std_ratio_per_dim – list[ndarray]  per-dim std_ratio at each eval
            std_ratio_median  – list[float]
    """
    obs = dataset["observations"]
    act = dataset["actions"]
    next_obs = dataset["next_observations"]

    n = len(obs)
    obs_dim = obs.shape[1]
    act_dim = act.shape[1]

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    model, params, opt_state, optimizer = create_train_state(
        init_rng, obs_dim, act_dim, learning_rate, hidden_dims,
        time_embed_dim, layer_norm,
    )

    # ---- choose training step ----
    use_std = (w_std > 0.0) and (knn_std_all is not None)
    if use_std:
        train_step = make_combined_train_step(
            model.apply, optimizer, w_std, std_batch_size,
            num_model_samples_train_std, num_ode_steps_std,
        )
        print(f"  Variance-matching ON: w_std={w_std}, M={num_model_samples_train_std}, "
              f"std_batch={std_batch_size}, ode_steps_std={num_ode_steps_std}")
    else:
        train_step = make_train_step(model.apply, optimizer)
        if w_std > 0 and knn_std_all is None:
            print("  WARNING: w_std > 0 but knn_std_all not provided; "
                  "falling back to base-only loss.")

    # Placeholder KNN std (never accessed when use_std=False)
    _dummy_knn = np.zeros((batch_size, obs_dim), dtype=np.float32)

    num_batches = max(n // batch_size, 1)

    # ---- history containers ----
    history = {
        "losses": [],
        "base_losses": [],
        "std_losses": [],
        "std_ratio_epochs": [],
        "std_ratio_per_dim": [],
        "std_ratio_median": [],
    }

    print(f"\nTraining for {num_epochs} epochs, {num_batches} batches/epoch, "
          f"batch_size={batch_size}")
    print(f"Dataset size: {n}, obs_dim={obs_dim}, act_dim={act_dim}")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(num_epochs):
        ep_total, ep_base, ep_std = [], [], []
        perm = np.random.permutation(n)

        for batch_idx in range(num_batches):
            start_i = batch_idx * batch_size
            end_i = min(start_i + batch_size, n)
            idx = perm[start_i:end_i]

            batch = {
                "observations": jnp.array(obs[idx]),
                "actions": jnp.array(act[idx]),
                "next_observations": jnp.array(next_obs[idx]),
            }

            rng, step_rng = jax.random.split(rng)

            if use_std:
                knn_batch = jnp.array(knn_std_all[idx])
                params, opt_state, loss, info = train_step(
                    params, opt_state, batch, knn_batch, step_rng,
                )
            else:
                params, opt_state, loss, info = train_step(
                    params, opt_state, batch, step_rng,
                )

            ep_total.append(float(info["total_loss"]))
            ep_base.append(float(info["base_loss"]))
            ep_std.append(float(info["std_loss"]))

        avg_total = np.mean(ep_total)
        avg_base = np.mean(ep_base)
        avg_std = np.mean(ep_std)
        history["losses"].append(avg_total)
        history["base_losses"].append(avg_base)
        history["std_losses"].append(avg_std)

        # ---- logging ----
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            elapsed = time.time() - start_time
            msg = (f"Epoch {epoch + 1:4d}/{num_epochs} | "
                   f"total={avg_total:.6f}  base={avg_base:.6f}  "
                   f"std={avg_std:.6f} | {elapsed:.1f}s")
            print(msg)

        # ---- std_ratio diagnostic ----
        do_eval = (
            std_eval_interval > 0
            and std_eval_obs is not None
            and ((epoch + 1) % std_eval_interval == 0 or epoch == 0)
        )
        if do_eval:
            rng, eval_rng = jax.random.split(rng)
            per_dim, median_r = _compute_std_ratio_eval(
                model, params, std_eval_obs, std_eval_act,
                std_eval_knn_std, eval_rng,
                num_samples=30, num_steps=30,
            )
            history["std_ratio_epochs"].append(epoch + 1)
            history["std_ratio_per_dim"].append(per_dim)
            history["std_ratio_median"].append(median_r)
            print(f"  >> std_ratio median={median_r:.4f}  "
                  f"per_dim={np.array2string(per_dim, precision=3, separator=', ')}")

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete in {total_time:.1f}s. "
          f"Final total_loss: {history['losses'][-1]:.6f}")

    return model, params, history


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    n = 1000
    obs_dim, act_dim = 11, 3
    rng_np = np.random.RandomState(42)
    dataset = {
        "observations": rng_np.randn(n, obs_dim).astype(np.float32),
        "actions": rng_np.randn(n, act_dim).astype(np.float32),
        "next_observations": rng_np.randn(n, obs_dim).astype(np.float32),
    }
    model, params, hist = train_flow_model(
        dataset, num_epochs=10, batch_size=128, log_interval=5,
    )
    print(f"Final loss: {hist['losses'][-1]:.6f}")
