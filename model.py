"""
model.py — Encoder–Decoder Transformer for Multi-Step Time Series Forecasting
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Any


# ============================================================
# Positional Encoding (learnable)
# ============================================================

class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    @nn.compact
    def __call__(self, x):
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        seq_len = x.shape[1]

        pos_emb = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model)
        )

        return x + pos_emb[:seq_len, :]


# ============================================================
# Transformer Encoder Block
# ============================================================

class EncoderBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = True):

        # Self Attention
        x_norm = nn.LayerNorm()(x)
        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout,
        )(x_norm, deterministic=not train)
        x = x + attn

        # MLP
        x_norm = nn.LayerNorm()(x)
        mlp = nn.Dense(self.mlp_dim)(x_norm)
        mlp = nn.gelu(mlp)
        mlp = nn.Dropout(self.dropout)(mlp, deterministic=not train)
        mlp = nn.Dense(self.d_model)(mlp)

        return x + mlp


# ============================================================
# Transformer Decoder Block (cross-attention)
# ============================================================

class DecoderBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, enc_out, train: bool = True):
        """
        x: decoder queries (batch, out_len, d_model)
        enc_out: encoder output (batch, seq_len, d_model)
        """

        # Self-Attention (decoder)
        x_norm = nn.LayerNorm()(x)
        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout,
        )(x_norm, deterministic=not train)
        x = x + attn

        # Cross Attention
        x_norm = nn.LayerNorm()(x)
        cross = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout
        )(x_norm, enc_out, deterministic=not train)
        x = x + cross

        # MLP
        x_norm = nn.LayerNorm()(x)
        mlp = nn.Dense(self.mlp_dim)(x_norm)
        mlp = nn.gelu(mlp)
        mlp = nn.Dropout(self.dropout)(mlp, deterministic=not train)
        mlp = nn.Dense(self.d_model)(mlp)

        return x + mlp


# ============================================================
# Full Transformer Model
# ============================================================

class TimeSeriesTransformer(nn.Module):
    seq_len: int
    out_len: int
    num_features: int
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 3
    mlp_dim: int = 256
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        x: (batch, seq_len, num_features)
        returns: (batch, out_len)
        """

        # ====================================================
        # 1. Input Projection
        # ====================================================
        x = nn.Dense(self.d_model)(x)

        # Add positional encoding
        x = PositionalEncoding(self.d_model)(x)

        # ====================================================
        # 2. Encoder
        # ====================================================
        for _ in range(self.num_layers):
            x = EncoderBlock(
                self.d_model,
                self.num_heads,
                self.mlp_dim,
                self.dropout,
            )(x, train=train)
        
        encoder_output = x  # (batch, seq_len, d_model)

        # ====================================================
        # 3. Decoder: Learnable Horizon Queries
        # ====================================================
        horizon_queries = self.param(
            "horizon_queries",
            nn.initializers.normal(stddev=0.02),
            (self.out_len, self.d_model),
        )

        # Repeat queries for batch
        q = jnp.tile(horizon_queries[None, :, :], (x.shape[0], 1, 1))

        # Decoder blocks
        for _ in range(self.num_layers):
            q = DecoderBlock(
                self.d_model,
                self.num_heads,
                self.mlp_dim,
                self.dropout
            )(q, encoder_output, train=train)

        # ====================================================
        # 4. Output Head: Predict Close price for each horizon
        # ====================================================
        out = nn.Dense(1)(q)          # (batch, out_len, 1)
        out = out.squeeze(-1)         # (batch, out_len)

        return out
