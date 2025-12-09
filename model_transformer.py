"""
model_transformer.py

Advanced Transformer model for Predictive Maintenance using
reconstruction error-based anomaly detection.

Features:
- Feature extraction with 1D CNNs
- Trend + seasonality decomposition
- Block-sparse self-attention with stable padding
- Transformer encoder layers with LayerNorm & residuals
- Final linear prediction layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# =========================================================
# 1) Feature Extractor (1D CNN-based)
# =========================================================

class FeatureExtractor(nn.Module):
    """
    Extracts local temporal patterns using lightweight 1D CNN layers.
    Converts raw time-series (C channels) into a higher dimensional latent space.
    """

    def __init__(self, input_dim, model_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, model_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # → (B, C, T)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.transpose(1, 2)  # back to (B, T, model_dim)
        x = self.norm(x)
        return x
# =========================================================
# 2) Trend / Seasonality Decomposition
# =========================================================

class TrendLayer(nn.Module):
    """Linear trend extraction."""

    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(x)


class SeasonalityLayer(nn.Module):
    """Frequency-like component extraction."""

    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.sin(self.fc(x))
# =========================================================
# 3) Block-Sparse Self-Attention (Stable Version)
# =========================================================

class BlockSparseSelfAttention(nn.Module):
    """
    Block-level self-attention.

    To avoid irregular block sizes breaking MultiHeadAttention,
    the final block is zero-padded to match block_size.
    """

    def __init__(self, dim, num_heads=4, block_size=20):
        super().__init__()
        self.block_size = block_size

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, D = x.size()
        outputs = []

        # Number of full blocks
        num_blocks = (T + self.block_size - 1) // self.block_size

        for b in range(num_blocks):
            start = b * self.block_size
            end = min(start + self.block_size, T)
            block = x[:, start:end, :]

            # Pad last block if needed
            if block.size(1) < self.block_size:
                pad_len = self.block_size - block.size(1)
                block = torch.cat(
                    [block, torch.zeros(B, pad_len, D, device=x.device)],
                    dim=1
                )

            attn_out, _ = self.attn(block, block, block)
            outputs.append(attn_out)

        # Concatenate and trim back to original T
        out = torch.cat(outputs, dim=1)
        out = out[:, :T, :]
        return out
    
# =========================================================
# 4) Transformer Encoder Block
# =========================================================

class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block:
    - Block-sparse attention
    - Feed-forward network
    - Residual connections
    - LayerNorm
    """

    def __init__(self, dim, num_heads=4, dropout=0.3):
        super().__init__()

        self.attn = BlockSparseSelfAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention + residual
        attn_out = self.attn(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed-forward + residual
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x
    
# =========================================================
# 5) Final Model: AdvancedAutoInformerModel
# =========================================================

class AdvancedAutoInformerModel(nn.Module):
    """
    Main Transformer model for predictive maintenance.

    Input   : (batch, seq_len, channels)
    Output  : next-timestep prediction (batch, channels)
    """

    def __init__(self, 
                 input_dim=4, 
                 model_dim=32, 
                 num_heads=4, 
                 num_layers=2, 
                 dropout=0.3):
        super().__init__()

        self.feature = FeatureExtractor(input_dim, model_dim)

        # Decomposition layers
        self.trend = TrendLayer(model_dim)
        self.season = SeasonalityLayer(model_dim)

        # Encoder stack
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(
                EncoderBlock(model_dim, num_heads=num_heads, dropout=dropout)
            )
        self.encoder_stack = nn.Sequential(*encoder_layers)

        # Final prediction head
        self.fc_out = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        """
        x shape: (B, T, C)
        Returns predicted next timestep y_pred: (B, C)
        """

        # 1) Feature extraction
        x = self.feature(x)

        # 2) Trend + seasonality residual add
        x = x + self.trend(x) + self.season(x)

        # 3) Transformer encoding
        x = self.encoder_stack(x)

        # 4) Final → use last timestep’s representation
        x_last = x[:, -1, :]    
        out = self.fc_out(x_last) 

        return out
