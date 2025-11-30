"""
train.py
Training loop for multi-step Transformer forecasting
compatible with Option C model and data.py windowing.
"""

import os
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.training import train_state, checkpoints

from model import TimeSeriesTransformer


# ============================================================
# Create Training State
# ============================================================
def create_train_state(rng, learning_rate, seq_len, out_len, num_features):
    model = TimeSeriesTransformer(
        seq_len=seq_len,
        d_model=128,
        num_heads=4,
        num_layers=4,
        mlp_dim=256,
        out_len=out_len,
        num_features=num_features,
    )

    dummy_input = jnp.ones((1, seq_len, num_features))

    params = model.init(
        {"params": rng, "dropout": rng},
        dummy_input,
        train=True
    )["params"]

    tx = optax.adamw(learning_rate)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )




# ============================================================
# Loss Function
# ============================================================

def loss_fn(params, batch, state):
    preds = state.apply_fn(
        {"params": params},
        batch["X"],
        train=True,
        rngs={"dropout": jax.random.PRNGKey(0)}
    )
    return jnp.mean((preds - batch["y"]) ** 2)



# ============================================================
# Training Step
# ============================================================

@jax.jit
def train_step(state, batch):
    grads = jax.grad(loss_fn)(state.params, batch, state)
    return state.apply_gradients(grads=grads)


# ============================================================
# Validation Step
# ============================================================

@jax.jit
def eval_step(state, batch):
    preds = state.apply_fn(
        {"params": state.params},
        batch["X"],
        train=False,
        rngs={"dropout": jax.random.PRNGKey(0)}
    )
    return jnp.mean((preds - batch["y"]) ** 2)



# ============================================================
# Batch Iterator
# ============================================================

def get_batches(X, y, batch_size):
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)

    for i in range(0, n, batch_size):
        batch_idx = idx[i:i + batch_size]
        yield {
            "X": jnp.array(X[batch_idx]),
            "y": jnp.array(y[batch_idx]),
        }


# ============================================================
# Training Loop
# ============================================================

def train_model(
    X_train, y_train,
    X_val, y_val,
    seq_len,
    out_len,
    num_features=5,
    batch_size=64,
    epochs=10,
    learning_rate=1e-4,
    ckpt_dir="./checkpoints"
):

    os.makedirs(ckpt_dir, exist_ok=True)

    rng = jax.random.PRNGKey(0)

    state = create_train_state(
        rng,
        learning_rate,
        seq_len,
        out_len,
        num_features
    )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # -------- TRAINING --------
        train_losses = []
        for batch in get_batches(X_train, y_train, batch_size):
            state = train_step(state, batch)
            train_losses.append(
                float(loss_fn(state.params, batch, state))
            )

        train_loss = np.mean(train_losses)

        # -------- VALIDATION --------
        val_batch = {
            "X": jnp.array(X_val),
            "y": jnp.array(y_val)
        }
        val_loss = float(eval_step(state, val_batch))

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        # -------- CHECKPOINTING --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoints.save_checkpoint(
                ckpt_dir, target=state.params, step=epoch, overwrite=True
            )
            print("   ✔ Saved new best checkpoint.")

    print("Training complete.")
    return state
