from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


class MoleculeEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 28,
        hidden_dim: int = 128,
        heads: int = 4,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = GATv2Conv(
            input_dim,
            hidden_dim,
            heads=heads,
            edge_dim=6,
            dropout=dropout,
        )

        self.conv2 = GATv2Conv(
            hidden_dim * heads,
            hidden_dim,
            heads=heads,
            edge_dim=6,
            dropout=dropout,
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * heads, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
    ) -> Tensor:

        x = self.conv1(
            x,
            edge_index,
            edge_attr,
        )

        x = torch.nn.functional.gelu(x)

        x = self.conv2(
            x,
            edge_index,
            edge_attr,
        )

        x = torch.nn.functional.gelu(x)

        x = global_mean_pool(
            x,
            batch,
        )

        return self.projection(x)


class ProteinEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int = 22,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        feedforward_dim: int = 512,
        output_dim: int = 256,
        max_length: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            max_length,
            embedding_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:

        batch_size, sequence_length = tokens.shape

        positions = torch.arange(
            sequence_length,
            device=tokens.device,
        )

        positions = positions.unsqueeze(0).expand(
            batch_size,
            -1,
        )

        x = self.token_embedding(tokens)
        x = x + self.position_embedding(positions)

        padding_mask = ~attention_mask

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask,
        )

        mask = attention_mask.unsqueeze(-1).float()

        pooled = (
            x * mask
        ).sum(dim=1) / mask.sum(
            dim=1
        ).clamp(min=1.0)

        return self.projection(pooled)