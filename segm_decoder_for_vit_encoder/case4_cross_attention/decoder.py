"""
Case 4: Cross-Attention Decoder (精度最重視)

特徴:
- 最深層をquery、他の層をkey/valueとするcross-attention
- Transformer Decoderライクな構造
- 各層の情報を適応的に重み付け
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention module."""

    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        Args:
            query: (B, N_q, C)
            key_value: (B, N_kv, C)

        Returns:
            (B, N_q, C)
        """
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]

        # Project to Q, K, V
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: (B, num_heads, N_q, N_kv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.out_proj(x)

        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention and cross-attention."""

    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Cross-attention
        self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, memory):
        """
        Args:
            x: Query features (B, N, C)
            memory: Key/Value features from encoder (B, M, C)

        Returns:
            (B, N, C)
        """
        # Self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # Cross-attention
        x = x + self.cross_attn(self.norm2(x), memory)

        # FFN
        x = x + self.mlp(self.norm3(x))

        return x


class ViTDecoderCrossAttention(nn.Module):
    """
    ViT Decoder using Cross-Attention mechanism.

    最深層の特徴をqueryとして、他の全層をkey/valueとするcross-attention。
    最も表現力が高いが、計算コストも高い。
    """

    def __init__(
        self,
        encoder_channels,  # e.g., [384, 384, 384] for ViT
        decoder_channels=256,
        num_decoder_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        upsampling_mode='bilinear',
        final_upsampling=16,
    ):
        """
        Args:
            encoder_channels: List of encoder output channels for each stage
            decoder_channels: Embedding dimension for decoder
            num_decoder_layers: Number of transformer decoder layers
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            dropout: Dropout rate
            upsampling_mode: Upsampling mode ('bilinear' or 'nearest')
            final_upsampling: Final upsampling factor to match input resolution
        """
        super().__init__()

        self.upsampling_mode = upsampling_mode
        self.final_upsampling = final_upsampling
        self.decoder_channels = decoder_channels

        # Project encoder features to common embedding dimension
        self.encoder_projs = nn.ModuleList([
            nn.Linear(ch, decoder_channels)
            for ch in encoder_channels
        ])

        # Positional encoding for query
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, decoder_channels))

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(decoder_channels, num_heads, mlp_ratio, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Final projection back to spatial format
        self.final_norm = nn.LayerNorm(decoder_channels)

    def forward(self, *features):
        """
        Args:
            features: Tuple of encoder outputs, each of shape (B, C, H, W)

        Returns:
            Decoded feature map of shape (B, decoder_channels, H_up, W_up)
        """
        B, _, H, W = features[0].shape
        N = H * W  # Number of tokens

        # Flatten and project all encoder features
        projected_features = []
        for feat, proj in zip(features, self.encoder_projs):
            # (B, C, H, W) -> (B, C, N) -> (B, N, C) -> (B, N, decoder_channels)
            feat_flat = feat.flatten(2).transpose(1, 2)
            feat_proj = proj(feat_flat)
            projected_features.append(feat_proj)

        # Concatenate all projected features as memory for cross-attention
        # (B, N * num_layers, decoder_channels)
        memory = torch.cat(projected_features, dim=1)

        # Use the last (deepest) layer as initial query
        query = projected_features[-1]

        # Add positional encoding
        query = query + self.pos_encoding

        # Apply transformer decoder layers
        for layer in self.decoder_layers:
            query = layer(query, memory)

        # Final normalization
        x = self.final_norm(query)

        # Reshape back to spatial format: (B, N, C) -> (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, self.decoder_channels, H, W)

        # Final upsampling
        if self.final_upsampling > 1:
            x = F.interpolate(
                x,
                scale_factor=self.final_upsampling,
                mode=self.upsampling_mode,
                align_corners=False if self.upsampling_mode == 'bilinear' else None
            )

        return x


# Usage example
if __name__ == "__main__":
    # ViT encoder channels (all same resolution)
    encoder_channels = [384, 384, 384]

    decoder = ViTDecoderCrossAttention(
        encoder_channels=encoder_channels,
        decoder_channels=256,
        num_decoder_layers=4,
        num_heads=8,
        final_upsampling=16
    )

    # Simulate ViT encoder outputs
    B, H, W = 2, 32, 80
    dummy_features = [
        torch.randn(B, 384, H, W),
        torch.randn(B, 384, H, W),
        torch.randn(B, 384, H, W),
    ]

    output = decoder(*dummy_features)
    print(f"Input feature shapes: {[f.shape for f in dummy_features]}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (B={B}, C=256, H={H*16}, W={W*16})")

    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Total parameters: {total_params:,}")
