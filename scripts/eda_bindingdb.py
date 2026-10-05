from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        FIGURES_DIR / filename,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_pkd_distribution(data: pd.DataFrame) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.hist(
        data["pkd"],
        bins=60,
        density=True,
    )

    axis.axvline(
        data["pkd"].mean(),
        linestyle="--",
        label=f"Mean = {data['pkd'].mean():.2f}",
    )

    axis.set_title(
        "pKd Distribution"
    )

    axis.set_xlabel(
        "pKd"
    )

    axis.set_ylabel(
        "Density"
    )

    axis.legend()

    save_figure(
        figure,
        "pkd_distribution.png",
    )


def plot_protein_length_distribution(
    data: pd.DataFrame,
) -> None:

    lengths = (
        data["protein_sequence"]
        .str.len()
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.hist(
        lengths,
        bins=60,
    )

    axis.axvline(
        lengths.mean(),
        linestyle="--",
        label=f"Mean = {lengths.mean():.0f} aa",
    )

    axis.axvline(
        lengths.median(),
        linestyle=":",
        label=f"Median = {lengths.median():.0f} aa",
    )

    axis.set_title(
        "Protein Length Distribution"
    )

    axis.set_xlabel(
        "Protein length, amino acids"
    )

    axis.set_ylabel(
        "Count"
    )

    axis.legend()

    save_figure(
        figure,
        "protein_length_distribution.png",
    )


def plot_samples_per_ligand(
    data: pd.DataFrame,
) -> None:

    counts = (
        data["ligand_id"]
        .value_counts()
    )

    clipped = counts.clip(
        upper=counts.quantile(0.99)
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.hist(
        clipped,
        bins=50,
    )

    axis.set_yscale(
        "log"
    )

    axis.set_title(
        "Samples per Ligand"
    )

    axis.set_xlabel(
        "Number of measurements"
    )

    axis.set_ylabel(
        "Ligands"
    )

    save_figure(
        figure,
        "samples_per_ligand.png",
    )


def plot_samples_per_target(
    data: pd.DataFrame,
) -> None:

    counts = (
        data["target_id"]
        .value_counts()
    )

    clipped = counts.clip(
        upper=counts.quantile(0.99)
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.hist(
        clipped,
        bins=50,
    )

    axis.set_yscale(
        "log"
    )

    axis.set_title(
        "Samples per Target"
    )

    axis.set_xlabel(
        "Number of measurements"
    )

    axis.set_ylabel(
        "Targets"
    )

    save_figure(
        figure,
        "samples_per_target.png",
    )


def plot_top_targets(
    data: pd.DataFrame,
) -> None:

    top_targets = (
        data["target_name"]
        .value_counts()
        .head(20)
        .sort_values()
    )

    figure, axis = plt.subplots(
        figsize=(12, 8)
    )

    axis.barh(
        top_targets.index.astype(str),
        top_targets.values,
    )

    axis.set_title(
        "Top 20 Targets by Number of Measurements"
    )

    axis.set_xlabel(
        "Measurements"
    )

    axis.set_ylabel(
        "Target"
    )

    save_figure(
        figure,
        "top_targets.png",
    )


def plot_pkd_vs_protein_length(
    data: pd.DataFrame,
) -> None:

    sample_size = min(
        100_000,
        len(data),
    )

    sample = data.sample(
        sample_size,
        random_state=42,
    )

    protein_lengths = (
        sample["protein_sequence"]
        .str.len()
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.scatter(
        protein_lengths,
        sample["pkd"],
        alpha=0.15,
        s=8,
    )

    axis.set_title(
        "pKd vs Protein Length"
    )

    axis.set_xlabel(
        "Protein length, amino acids"
    )

    axis.set_ylabel(
        "pKd"
    )

    save_figure(
        figure,
        "pkd_vs_protein_length.png",
    )


def check_split_leakage() -> dict:
    train = pd.read_parquet(
        PROCESSED_DIR / "train.parquet"
    )

    validation = pd.read_parquet(
        PROCESSED_DIR / "validation.parquet"
    )

    test = pd.read_parquet(
        PROCESSED_DIR / "test.parquet"
    )

    train_ligands = set(
        train["ligand_id"]
    )

    validation_ligands = set(
        validation["ligand_id"]
    )

    test_ligands = set(
        test["ligand_id"]
    )

    train_targets = set(
        train["target_id"]
    )

    validation_targets = set(
        validation["target_id"]
    )

    test_targets = set(
        test["target_id"]
    )

    return {
        "train_validation_ligand_overlap": len(
            train_ligands & validation_ligands
        ),
        "train_test_ligand_overlap": len(
            train_ligands & test_ligands
        ),
        "validation_test_ligand_overlap": len(
            validation_ligands & test_ligands
        ),
        "train_validation_target_overlap": len(
            train_targets & validation_targets
        ),
        "train_test_target_overlap": len(
            train_targets & test_targets
        ),
        "validation_test_target_overlap": len(
            validation_targets & test_targets
        ),
    }


def build_summary(
    data: pd.DataFrame,
) -> dict:

    protein_lengths = (
        data["protein_sequence"]
        .str.len()
    )

    summary = {
        "rows": int(len(data)),
        "unique_ligands": int(
            data["ligand_id"].nunique()
        ),
        "unique_targets": int(
            data["target_id"].nunique()
        ),
        "unique_smiles": int(
            data["smiles"].nunique()
        ),
        "pkd": {
            "mean": float(
                data["pkd"].mean()
            ),
            "median": float(
                data["pkd"].median()
            ),
            "std": float(
                data["pkd"].std()
            ),
            "min": float(
                data["pkd"].min()
            ),
            "max": float(
                data["pkd"].max()
            ),
        },
        "kd_nm": {
            "mean": float(
                data["kd_nm"].mean()
            ),
            "median": float(
                data["kd_nm"].median()
            ),
            "min": float(
                data["kd_nm"].min()
            ),
            "max": float(
                data["kd_nm"].max()
            ),
        },
        "protein_length": {
            "mean": float(
                protein_lengths.mean()
            ),
            "median": float(
                protein_lengths.median()
            ),
            "min": int(
                protein_lengths.min()
            ),
            "max": int(
                protein_lengths.max()
            ),
        },
        "missing_values": {
            column: int(
                data[column].isna().sum()
            )
            for column in data.columns
        },
    }

    summary["split_leakage"] = (
        check_split_leakage()
    )

    return summary


def main() -> None:
    path = (
        PROCESSED_DIR
        / "bindingdb_pkd.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}"
        )

    print("=" * 70)
    print("MolBind - BindingDB EDA")
    print("=" * 70)

    print("\nLoading dataset...")

    data = pd.read_parquet(
        path
    )

    print(
        f"Rows: {len(data):,}"
    )

    print(
        f"Ligands: {data['ligand_id'].nunique():,}"
    )

    print(
        f"Targets: {data['target_id'].nunique():,}"
    )

    summary = build_summary(
        data
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORTS_DIR / "eda_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nGenerating figures...")

    plot_pkd_distribution(
        data
    )

    plot_protein_length_distribution(
        data
    )

    plot_samples_per_ligand(
        data
    )

    plot_samples_per_target(
        data
    )

    plot_top_targets(
        data
    )

    plot_pkd_vs_protein_length(
        data
    )

    print("\nFigures saved to:")
    print(FIGURES_DIR)

    print("\nLeakage check:")

    leakage = summary[
        "split_leakage"
    ]

    for key, value in leakage.items():
        print(
            f"{key}: {value}"
        )

    print("\nEDA complete.")


if __name__ == "__main__":
    main()