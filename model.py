"""
model.py
Multi-step TimeSeries Transformer (Option C)
Fully compatible with Flax 0.8+ dropout API
"""

import jax.numpy as jnp
import flax.linen as nn


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    @nn.compact
    def __call__(self, x):
        pos_emb = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model),
        )
        return x + pos_emb[:x.shape[1], :]


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, train=True):

        # ---- Multi-head Self Attention ----
        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout
        )
        x2 = attn(x, deterministic=not train)
        x = nn.LayerNorm()(x + x2)

        # ---- Feedforward ----
        x2 = nn.Dense(self.mlp_dim)(x)
        x2 = nn.gelu(x2)
        x2 = nn.Dropout(self.dropout)(x2, deterministic=not train)
        x2 = nn.Dense(self.d_model)(x2)

        x = nn.LayerNorm()(x + x2)
        return x


# ============================================================
# TimeSeries Transformer
# ============================================================

class TimeSeriesTransformer(nn.Module):
    seq_len: int = 60
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    mlp_dim: int = 256
    dropout: float = 0.1
    out_len: int = 5
    num_features: int = 5

    @nn.compact
    def __call__(self, x, train=True):
        """
        x shape: (batch, seq_len, num_features)
        """

        # Project inputs to d_model
        x = nn.Dense(self.d_model)(x)

        # Add positional encoding
        x = PositionalEncoding(self.d_model)(x)

        # Transformer layers
        for _ in range(self.num_layers):
            x = TransformerBlock(
                self.d_model,
                self.num_heads,
                self.mlp_dim,
                self.dropout
            )(x, train=train)

        # Final timestep representation
        last = x[:, -1, :]  # (batch, d_model)

        # Predict next 5 Close prices
        out = nn.Dense(self.out_len)(last)

        return out
