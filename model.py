"""
model.py
Multi-step TimeSeries Transformer (final version)
"""

import jax.numpy as jnp
import flax.linen as nn


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


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.15

    @nn.compact
    def __call__(self, x, train=True):
        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout
        )
        x2 = attn(x, deterministic=not train)
        x = nn.LayerNorm()(x + x2)

        x2 = nn.Dense(self.mlp_dim)(x)
        x2 = nn.gelu(x2)
        x2 = nn.Dropout(self.dropout)(x2, deterministic=not train)
        x2 = nn.Dense(self.d_model)(x2)

        x = nn.LayerNorm()(x + x2)
        return x


class TimeSeriesTransformer(nn.Module):
    seq_len: int = 60
    d_model: int = 128          # <-- CHANGED BACK
    num_heads: int = 4          # <-- CHANGED BACK
    num_layers: int = 4         # <-- CHANGED BACK
    mlp_dim: int = 256          # <-- CHANGED BACK
    dropout: float = 0.10
    out_len: int = 5
    num_features: int = 5

    @nn.compact
    def __call__(self, x, train=True):

        x = nn.Dense(self.d_model)(x)
        x = PositionalEncoding(self.d_model)(x)

        for _ in range(self.num_layers):
            x = TransformerBlock(
                self.d_model,
                self.num_heads,
                self.mlp_dim,
                self.dropout
            )(x, train=train)

        last = x[:, -1, :]
        out = nn.Dense(self.out_len)(last)
        return out
