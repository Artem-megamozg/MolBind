from pathlib import Path

import pandas as pd

from molbind.data.dataset import (
    create_dataloader,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:

    train_path = (
        ROOT
        / "data"
        / "processed"
        / "train.parquet"
    )

    data = pd.read_parquet(
        train_path
    )

    sample = data.head(32)

    loader = create_dataloader(
        sample,
        batch_size=8,
        shuffle=False,
        max_protein_length=1024,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("=" * 70)
    print("MolBind dataset pipeline test")
    print("=" * 70)

    print(
        f"Molecule graphs: "
        f"{batch['molecule'].num_graphs}"
    )

    print(
        f"Molecule nodes: "
        f"{batch['molecule'].x.shape}"
    )

    print(
        f"Molecule edges: "
        f"{batch['molecule'].edge_index.shape}"
    )

    print(
        f"Protein tokens: "
        f"{batch['protein_tokens'].shape}"
    )

    print(
        f"Protein mask: "
        f"{batch['protein_mask'].shape}"
    )

    print(
        f"Targets: "
        f"{batch['target'].shape}"
    )

    print(
        f"Target values: "
        f"{batch['target'].tolist()}"
    )

    print("\nPipeline test passed.")


if __name__ == "__main__":
    main()