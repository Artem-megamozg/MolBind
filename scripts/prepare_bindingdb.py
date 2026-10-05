from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from molbind.data.preprocessing import (
    load_bindingdb,
    prepare_affinity_dataset,
    split_by_group,
)


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "base.yaml"
RAW_PATH = ROOT / "data" / "raw" / "BindingDB_All.tsv"
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    print("=" * 70)
    print("MolBind - BindingDB preprocessing")
    print("=" * 70)

    print("\nLoading BindingDB...")

    data = load_bindingdb(
        RAW_PATH,
    )

    print(
        f"Raw rows: {len(data):,}"
    )

    prepared = prepare_affinity_dataset(
        data,
        min_pkd=config["data"]["min_pkd"],
        max_pkd=config["data"]["max_pkd"],
        max_protein_length=config["data"]["max_protein_length"],
    )

    print(
        f"Prepared rows: {len(prepared):,}"
    )

    print(
        f"Unique ligands: "
        f"{prepared['ligand_id'].nunique():,}"
    )

    print(
        f"Unique targets: "
        f"{prepared['target_id'].nunique():,}"
    )

    print(
        f"pKd mean: "
        f"{prepared['pkd'].mean():.3f}"
    )

    print(
        f"pKd std: "
        f"{prepared['pkd'].std():.3f}"
    )

    print(
        f"pKd min: "
        f"{prepared['pkd'].min():.3f}"
    )

    print(
        f"pKd max: "
        f"{prepared['pkd'].max():.3f}"
    )

    train, validation, test = split_by_group(
        prepared,
        group_column="ligand_id",
        test_size=config["data"]["test_size"],
        validation_size=config["data"]["validation_size"],
        random_state=config["project"]["seed"],
    )

    print("\nSplit sizes:")

    print(
        f"Train:      {len(train):,}"
    )

    print(
        f"Validation: {len(validation):,}"
    )

    print(
        f"Test:       {len(test):,}"
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        PROCESSED_DIR
        / "train.parquet"
    )

    validation_path = (
        PROCESSED_DIR
        / "validation.parquet"
    )

    test_path = (
        PROCESSED_DIR
        / "test.parquet"
    )

    full_path = (
        PROCESSED_DIR
        / "bindingdb_pkd.parquet"
    )

    train.to_parquet(
        train_path,
        index=False,
    )

    validation.to_parquet(
        validation_path,
        index=False,
    )

    test.to_parquet(
        test_path,
        index=False,
    )

    prepared.to_parquet(
        full_path,
        index=False,
    )

    metadata = {
        "dataset": "BindingDB",
        "task": "pKd regression",
        "target_measurement": "Kd",
        "split_strategy": "ligand-cold",
        "rows": {
            "full": len(prepared),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "unique_ligands": int(
            prepared["ligand_id"].nunique()
        ),
        "unique_targets": int(
            prepared["target_id"].nunique()
        ),
    }

    with open(
        PROCESSED_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSaved:")
    print(train_path)
    print(validation_path)
    print(test_path)
    print(full_path)

    print("\nDone.")


if __name__ == "__main__":
    main()