from __future__ import annotations

from typing import List

import torch
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Data


ATOMIC_NUMBERS = [
    6,
    7,
    8,
    9,
    15,
    16,
    17,
    35,
    53,
    0,
]

HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]


def one_hot(value, choices: list) -> list[float]:
    return [float(value == choice) for choice in choices]


def atom_features(atom: Chem.Atom) -> list[float]:
    features = []

    atomic_number = atom.GetAtomicNum()
    if atomic_number not in ATOMIC_NUMBERS:
        atomic_number = 0

    features.extend(one_hot(atomic_number, ATOMIC_NUMBERS))
    features.extend(one_hot(min(atom.GetDegree(), 5), list(range(6))))

    features.append(float(atom.GetFormalCharge()))
    features.append(float(atom.GetIsAromatic()))

    features.extend(one_hot(atom.GetHybridization(), HYBRIDIZATIONS))
    features.extend(one_hot(min(atom.GetTotalNumHs(), 4), list(range(5))))

    return features


def bond_features(bond: Chem.Bond) -> list[float]:
    bond_type = bond.GetBondType()

    return [
        float(bond_type == Chem.BondType.SINGLE),
        float(bond_type == Chem.BondType.DOUBLE),
        float(bond_type == Chem.BondType.TRIPLE),
        float(bond_type == Chem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def smiles_to_graph(smiles: str) -> Data:
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.tensor(
        [atom_features(atom) for atom in molecule.GetAtoms()],
        dtype=torch.float32,
    )

    edge_indices: List[list[int]] = []
    edge_attributes: List[list[float]] = []

    for bond in molecule.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()

        features = bond_features(bond)

        edge_indices.append([start, end])
        edge_indices.append([end, start])

        edge_attributes.append(features)
        edge_attributes.append(features)

    if edge_indices:
        edge_index = torch.tensor(
            edge_indices,
            dtype=torch.long,
        ).t().contiguous()

        edge_attr = torch.tensor(
            edge_attributes,
            dtype=torch.float32,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float32)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )