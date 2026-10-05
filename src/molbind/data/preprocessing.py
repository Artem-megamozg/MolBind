from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit


REQUIRED_COLUMNS = [
    "BindingDB Reactant_set_id",
    "Ligand SMILES",
    "Ligand InChI Key",
    "Ki (nM)",
    "IC50 (nM)",
    "Kd (nM)",
    "EC50 (nM)",
    "Target Name",
    "Number of Protein Chains in Target (>1 implies a multichain complex)",
]


def normalize_column_name(column: str) -> str:
    return re.sub(r"\s+", " ", column.strip())


def find_column(
    columns: list[str],
    exact_names: list[str] | None = None,
    patterns: list[str] | None = None,
) -> str | None:
    normalized = {
        normalize_column_name(column): column
        for column in columns
    }

    if exact_names:
        for name in exact_names:
            name = normalize_column_name(name)
            if name in normalized:
                return normalized[name]

    if patterns:
        for column in columns:
            normalized_column = normalize_column_name(column)

            if all(
                re.search(pattern, normalized_column, re.IGNORECASE)
                for pattern in patterns
            ):
                return column

    return None


def find_target_sequence_column(columns: list[str]) -> str | None:
    candidates = []

    for column in columns:
        normalized = normalize_column_name(column)

        match = re.search(
            r"BindingDB Target Chain Sequence(?:\s+1)?$",
            normalized,
            re.IGNORECASE,
        )

        if match:
            candidates.append(column)

    if not candidates:
        return None

    return candidates[0]


def find_uniprot_column(columns: list[str]) -> str | None:
    exact_patterns = [
        r"^UniProt \(SwissProt\) Primary ID of Target Chain(?:\s+1)?$",
        r"^UniProt \(TrEMBL\) Primary ID of Target Chain(?:\s+1)?$",
    ]

    for pattern in exact_patterns:
        for column in columns:
            normalized = normalize_column_name(column)

            if re.search(pattern, normalized, re.IGNORECASE):
                return column

    return None


def read_bindingdb_header(path: str | Path) -> list[str]:
    path = Path(path)

    header = pd.read_csv(
        path,
        sep="\t",
        nrows=0,
        compression="infer",
    )

    return list(header.columns)


def resolve_columns(columns: list[str]) -> dict[str, str | None]:
    normalized_columns = [
        normalize_column_name(column)
        for column in columns
    ]

    mapping = {}

    for required in REQUIRED_COLUMNS:
        if required in normalized_columns:
            original = columns[
                normalized_columns.index(required)
            ]
            mapping[required] = original
        else:
            mapping[required] = None

    mapping["protein_sequence"] = find_target_sequence_column(
        columns
    )

    mapping["uniprot_id"] = find_uniprot_column(
        columns
    )

    return mapping


def load_bindingdb(
    path: str | Path,
    chunksize: int = 100_000,
) -> pd.DataFrame:

    path = Path(path)

    columns = read_bindingdb_header(path)
    mapping = resolve_columns(columns)

    missing = [
        column
        for column, value in mapping.items()
        if value is None
        and column
        not in {
            "uniprot_id",
            "protein_sequence",
        }
    ]

    if missing:
        raise ValueError(
            "Required BindingDB columns were not found: "
            + ", ".join(missing)
        )

    selected_columns = [
        value
        for value in mapping.values()
        if value is not None
    ]

    chunks = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=selected_columns,
        chunksize=chunksize,
        low_memory=False,
        compression="infer",
        dtype=str,
    ):
        chunks.append(chunk)

    data = pd.concat(
        chunks,
        ignore_index=True,
    )

    rename_map = {
        original: canonical
        for canonical, original in mapping.items()
        if original is not None
    }

    data = data.rename(
        columns=rename_map,
    )

    return data


def numeric_column(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        data[column],
        errors="coerce",
    )


def canonicalize_smiles(
    smiles: str,
) -> str | None:

    if not isinstance(smiles, str):
        return None

    smiles = smiles.strip()

    if not smiles:
        return None

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    return Chem.MolToSmiles(
        molecule,
        canonical=True,
    )


def sequence_is_valid(sequence: str) -> bool:
    if not isinstance(sequence, str):
        return False

    sequence = sequence.strip().upper()

    if not sequence:
        return False

    valid = set(
        "ACDEFGHIKLMNPQRSTVWYX"
    )

    return all(
        amino_acid in valid
        for amino_acid in sequence
    )


def sequence_hash(sequence: str) -> str:
    return hashlib.sha1(
        sequence.encode("utf-8")
    ).hexdigest()


def prepare_affinity_dataset(
    data: pd.DataFrame,
    min_pkd: float = 3.0,
    max_pkd: float = 14.0,
    max_protein_length: int = 1024,
) -> pd.DataFrame:

    data = data.copy()

    data["kd_nm"] = numeric_column(
        data,
        "Kd (nM)",
    )

    data["number_of_chains"] = numeric_column(
        data,
        "Number of Protein Chains in Target (>1 implies a multichain complex)",
    )

    data = data[
        data["kd_nm"].notna()
    ]

    data = data[
        data["kd_nm"] > 0
    ]

    data = data[
        data["number_of_chains"] == 1
    ]

    data["smiles"] = data[
        "Ligand SMILES"
    ].map(
        canonicalize_smiles
    )

    data = data[
        data["smiles"].notna()
    ]

    data["protein_sequence"] = data[
        "protein_sequence"
    ].astype(str).str.strip().str.upper()

    data = data[
        data["protein_sequence"].map(
            sequence_is_valid
        )
    ]

    data = data[
        data["protein_sequence"].str.len() >= 30
    ]

    data["protein_sequence"] = data[
        "protein_sequence"
    ].str.slice(
        0,
        max_protein_length,
    )

    data["pkd"] = (
        9.0
        - np.log10(
            data["kd_nm"].astype(float)
        )
    )

    data = data[
        data["pkd"].between(
            min_pkd,
            max_pkd,
        )
    ]

    data["ligand_id"] = data[
        "Ligand InChI Key"
    ].fillna(
        data["smiles"]
    )

    data["target_id"] = data[
        "uniprot_id"
    ].fillna("")

    missing_target = (
        data["target_id"].eq("")
    )

    data.loc[
        missing_target,
        "target_id",
    ] = data.loc[
        missing_target,
        "protein_sequence",
    ].map(sequence_hash)

    data["target_name"] = data[
        "Target Name"
    ].fillna(
        "Unknown target"
    )

    data = data[
        [
            "BindingDB Reactant_set_id",
            "smiles",
            "ligand_id",
            "protein_sequence",
            "target_id",
            "target_name",
            "kd_nm",
            "pkd",
        ]
    ]

    data = data.drop_duplicates(
        subset=[
            "ligand_id",
            "target_id",
            "pkd",
        ]
    )

    data = (
        data
        .groupby(
            [
                "ligand_id",
                "target_id",
                "smiles",
                "protein_sequence",
                "target_name",
            ],
            as_index=False,
        )
        .agg(
            kd_nm=("kd_nm", "median"),
            pkd=("pkd", "median"),
        )
    )

    data = data.reset_index(
        drop=True
    )

    return data


def split_by_group(
    data: pd.DataFrame,
    group_column: str,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_val_index, test_index = next(
        splitter.split(
            data,
            groups=data[group_column],
        )
    )

    train_val = data.iloc[
        train_val_index
    ].reset_index(
        drop=True
    )

    test = data.iloc[
        test_index
    ].reset_index(
        drop=True
    )

    validation_fraction = (
        validation_size
        / (1.0 - test_size)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_state,
    )

    train_index, validation_index = next(
        splitter.split(
            train_val,
            groups=train_val[group_column],
        )
    )

    train = train_val.iloc[
        train_index
    ].reset_index(
        drop=True
    )

    validation = train_val.iloc[
        validation_index
    ].reset_index(
        drop=True
    )

    return train, validation, test