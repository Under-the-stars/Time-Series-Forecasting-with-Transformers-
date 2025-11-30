"""
eval.py
Evaluation + prediction for 5-step Transformer model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints

from model import TimeSeriesTransformer


# ------------------------------------------------------------
# Load model parameters (checkpoint)
# ------------------------------------------------------------
def load_model(ckpt_dir, seq_len, out_len, num_features=5):
    print("🔹 Loading best checkpoint...")

    model = TimeSeriesTransformer(
        seq_len=seq_len,
        d_model=256,
        num_heads=8,
        num_layers=6,
        mlp_dim=512,
        out_len=out_len,
        dropout=0.15,
        num_features=num_features,
    )

    dummy = jnp.ones((1, seq_len, num_features))
    init_params = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        dummy,
        train=False
    )["params"]

    params = checkpoints.restore_checkpoint(ckpt_dir, target=init_params)
    return model, params


# ------------------------------------------------------------
# Run predictions
# ------------------------------------------------------------
def predict(model, params, X_test):
    print("🔹 Predicting...")
    return model.apply(
        {"params": params},
        jnp.array(X_test),
        train=False,
        rngs={"dropout": jax.random.PRNGKey(0)}
    )


# ------------------------------------------------------------
# Compute metrics horizon-by-horizon
# ------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    print("🔹 Computing metrics...")

    results = {}
    for h in range(5):
        true_h = y_true[:, h]
        pred_h = y_pred[:, h]

        mae = np.mean(np.abs(true_h - pred_h))
        rmse = np.sqrt(np.mean((true_h - pred_h) ** 2))

        results[f"MAE_h{h+1}"] = float(mae)
        results[f"RMSE_h{h+1}"] = float(rmse)

    return results


# ------------------------------------------------------------
# Full evaluation entry point
# ------------------------------------------------------------
def evaluate_model(X_test, y_test, ckpt_dir, seq_len, out_len, num_features=5):
    model, params = load_model(ckpt_dir, seq_len, out_len, num_features)

    preds = predict(model, params, X_test)
    preds = np.array(preds)

    results = compute_metrics(y_test, preds)
    return results, preds
