from __future__ import annotations

import torch


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"

PAD_TOKEN = 0

AA_TO_ID = {
    amino_acid: index + 1
    for index, amino_acid in enumerate(AMINO_ACIDS)
}

UNK_TOKEN = AA_TO_ID["X"]


def encode_protein(
    sequence: str,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = sequence.upper()

    sequence = "".join(
        amino_acid if amino_acid in AA_TO_ID else "X"
        for amino_acid in sequence
    )

    sequence = sequence[:max_length]

    token_ids = [
        AA_TO_ID[amino_acid]
        for amino_acid in sequence
    ]

    tokens = torch.tensor(
        token_ids,
        dtype=torch.long,
    )

    attention_mask = torch.ones(
        len(tokens),
        dtype=torch.bool,
    )

    return tokens, attention_mask


def pad_proteins(
    sequences: list[str],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    encoded = [
        encode_protein(sequence, max_length)
        for sequence in sequences
    ]

    max_batch_length = max(
        tokens.shape[0]
        for tokens, _ in encoded
    )

    max_batch_length = min(
        max_batch_length,
        max_length,
    )

    batch_tokens = torch.full(
        (
            len(encoded),
            max_batch_length,
        ),
        PAD_TOKEN,
        dtype=torch.long,
    )

    batch_mask = torch.zeros(
        (
            len(encoded),
            max_batch_length,
        ),
        dtype=torch.bool,
    )

    for index, (tokens, mask) in enumerate(encoded):
        length = min(tokens.shape[0], max_batch_length)

        batch_tokens[index, :length] = tokens[:length]
        batch_mask[index, :length] = mask[:length]

    return batch_tokens, batch_mask