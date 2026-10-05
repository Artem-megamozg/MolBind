from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from molbind.features.molecule import smiles_to_graph
from molbind.features.protein import encode_protein


class MolBindDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        max_protein_length: int = 1024,
    ):
        self.data = dataframe.reset_index(drop=True)
        self.max_protein_length = max_protein_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        row = self.data.iloc[index]

        molecule = smiles_to_graph(
            row["smiles"]
        )

        protein_tokens, protein_mask = encode_protein(
            row["protein_sequence"],
            max_length=self.max_protein_length,
        )

        target = torch.tensor(
            float(row["pkd"]),
            dtype=torch.float32,
        )

        return {
            "molecule": molecule,
            "protein_tokens": protein_tokens,
            "protein_mask": protein_mask,
            "target": target,
        }


def collate_batch(
    batch: list[dict],
    max_protein_length: int = 1024,
) -> dict:

    molecules = [
        item["molecule"]
        for item in batch
    ]

    molecule_batch = Batch.from_data_list(
        molecules
    )

    max_length = min(
        max(
            item["protein_tokens"].shape[0]
            for item in batch
        ),
        max_protein_length,
    )

    protein_tokens = torch.zeros(
        (len(batch), max_length),
        dtype=torch.long,
    )

    protein_mask = torch.zeros(
        (len(batch), max_length),
        dtype=torch.bool,
    )

    for index, item in enumerate(batch):
        length = min(
            item["protein_tokens"].shape[0],
            max_length,
        )

        protein_tokens[
            index,
            :length,
        ] = item["protein_tokens"][:length]

        protein_mask[
            index,
            :length,
        ] = item["protein_mask"][:length]

    targets = torch.stack(
        [
            item["target"]
            for item in batch
        ]
    )

    return {
        "molecule": molecule_batch,
        "protein_tokens": protein_tokens,
        "protein_mask": protein_mask,
        "target": targets,
    }


def create_dataloader(
    dataframe: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    max_protein_length: int = 1024,
    num_workers: int = 0,
) -> DataLoader:

    dataset = MolBindDataset(
        dataframe,
        max_protein_length=max_protein_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_batch(
            batch,
            max_protein_length=max_protein_length,
        ),
        pin_memory=torch.cuda.is_available(),
    )