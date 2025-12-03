"""
eval.py
Final — matching architecture (128-d) for checkpoint.
"""

import sys
sys.path.insert(0, "/content/Time-Series-Forecasting-with-Transformers-/src")

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from src.model import TimeSeriesTransformer


def load_model(ckpt_dir, seq_len, out_len, num_features=5):
    print("🔹 Loading best checkpoint...")

    # MUST MATCH model.py EXACTLY:
    model = TimeSeriesTransformer(
        seq_len=seq_len,
        d_model=128,      # ← FIXED
        num_heads=4,      # ← FIXED
        num_layers=4,     # ← FIXED
        mlp_dim=256,      # ← FIXED
        out_len=out_len,
        dropout=0.10,
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


def predict(model, params, X_test):
    print("🔹 Predicting...")
    return model.apply(
        {"params": params},
        jnp.array(X_test),
        train=False, 
        rngs={"dropout": jax.random.PRNGKey(0)}
    )


def compute_metrics(y_true, y_pred):
    print("🔹 Computing metrics...")

    results = {}
    for h in range(y_true.shape[1]):
        mae = float(np.mean(np.abs(y_true[:, h] - y_pred[:, h])))
        rmse = float(np.sqrt(np.mean((y_true[:, h] - y_pred[:, h]) ** 2)))

        results[f"MAE_h{h+1}"] = mae
        results[f"RMSE_h{h+1}"] = rmse

    return results


def evaluate_model(X_test, y_test, ckpt_dir, seq_len, out_len, num_features=5):
    model, params = load_model(ckpt_dir, seq_len, out_len, num_features)
    preds = np.array(predict(model, params, X_test))
    results = compute_metrics(y_test, preds)
    return results, preds
