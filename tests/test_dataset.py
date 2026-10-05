from pathlib import Path

import pandas as pd

from molbind.data.dataset import (
    create_dataloader,
)


ROOT = Path(__file__).resolve().parents[1]


def test_dataloader():
    data = pd.read_parquet(
        ROOT
        / "data"
        / "processed"
        / "train.parquet"
    )

    data = data.head(8)

    loader = create_dataloader(
        data,
        batch_size=4,
        shuffle=False,
        max_protein_length=1024,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert batch["protein_tokens"].shape[0] == 4

    assert batch["protein_mask"].shape[0] == 4

    assert batch["target"].shape[0] == 4

    assert batch["molecule"].num_graphs == 4

    assert batch["molecule"].x.shape[0] > 0