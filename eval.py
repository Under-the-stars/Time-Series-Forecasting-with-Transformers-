"""
eval.py — Evaluation utilities for Transformer forecasting
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from model import TimeSeriesTransformer


# ============================================================
# Load Trained Model + Checkpoint
# ============================================================

def load_model(
    ckpt_dir,
    seq_len,
    out_len,
    num_features,
):
    """
    Loads the model and restores parameters from checkpoint.
    """

    print("🔹 Loading trained Transformer model...")

    model = TimeSeriesTransformer(
        seq_len=seq_len,
        out_len=out_len,
        num_features=num_features,
    )

    dummy = jnp.ones((1, seq_len, num_features))

    init_params = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        dummy
