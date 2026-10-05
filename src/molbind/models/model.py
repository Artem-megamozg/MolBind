from __future__ import annotations

import torch
from torch import Tensor, nn

from molbind.models.encoders import (
    MoleculeEncoder,
    ProteinEncoder,
)


class MolBindModel(nn.Module):

    def __init__(
        self,
        molecule_output_dim: int = 256,
        protein_output_dim: int = 256,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.2,
        max_protein_length: int = 1024,
    ):
        super().__init__()

        self.molecule_encoder = MoleculeEncoder(
            output_dim=molecule_output_dim,
            dropout=dropout,
        )

        self.protein_encoder = ProteinEncoder(
            output_dim=protein_output_dim,
            max_length=max_protein_length,
            dropout=dropout,
        )

        fusion_dim = (
            molecule_output_dim
            + protein_output_dim
            + molecule_output_dim
            + molecule_output_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_dim,
                fusion_hidden_dim,
            ),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                fusion_hidden_dim,
                fusion_hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                fusion_hidden_dim // 2,
                1,
            ),
        )

    def forward(
        self,
        molecule_batch,
        protein_tokens: Tensor,
        protein_mask: Tensor,
    ) -> Tensor:

        molecule_embedding = self.molecule_encoder(
            molecule_batch.x,
            molecule_batch.edge_index,
            molecule_batch.edge_attr,
            molecule_batch.batch,
        )

        protein_embedding = self.protein_encoder(
            protein_tokens,
            protein_mask,
        )

        interaction = (
            molecule_embedding
            * protein_embedding
        )

        difference = torch.abs(
            molecule_embedding
            - protein_embedding
        )

        fused = torch.cat(
            [
                molecule_embedding,
                protein_embedding,
                interaction,
                difference,
            ],
            dim=1,
        )

        return self.fusion(fused).squeeze(-1)