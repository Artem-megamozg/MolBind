from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from molbind.data.dataset import create_dataloader
from molbind.data.seed import seed_everything
from molbind.evaluation.metrics import calculate_metrics
from molbind.models.model import MolBindModel


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(
        ROOT / "configs" / "base.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def move_batch_to_device(
    batch: dict,
    device: torch.device,
) -> dict:

    batch["molecule"] = batch["molecule"].to(
        device
    )

    batch["protein_tokens"] = (
        batch["protein_tokens"].to(device)
    )

    batch["protein_mask"] = (
        batch["protein_mask"].to(device)
    )

    batch["target"] = batch["target"].to(
        device
    )

    return batch


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    gradient_clip: float,
) -> float:

    model.train()

    running_loss = 0.0
    samples = 0

    progress = tqdm(
        loader,
        desc="Train",
        leave=False,
    )

    for batch in progress:

        batch = move_batch_to_device(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            predictions = model(
                batch["molecule"],
                batch["protein_tokens"],
                batch["protein_mask"],
            )

            loss = torch.nn.functional.mse_loss(
                predictions,
                batch["target"],
            )

        if use_amp:
            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip,
            )

            optimizer.step()

        batch_size = batch[
            "target"
        ].shape[0]

        running_loss += (
            loss.item() * batch_size
        )

        samples += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return running_loss / samples


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, dict[str, float]]:

    model.eval()

    predictions = []
    targets = []

    running_loss = 0.0
    samples = 0

    progress = tqdm(
        loader,
        desc="Eval",
        leave=False,
    )

    for batch in progress:

        batch = move_batch_to_device(
            batch,
            device,
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            outputs = model(
                batch["molecule"],
                batch["protein_tokens"],
                batch["protein_mask"],
            )

            loss = torch.nn.functional.mse_loss(
                outputs,
                batch["target"],
            )

        batch_size = batch[
            "target"
        ].shape[0]

        running_loss += (
            loss.item() * batch_size
        )

        samples += batch_size

        predictions.append(
            outputs.detach()
            .float()
            .cpu()
            .numpy()
        )

        targets.append(
            batch["target"]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    predictions = np.concatenate(
        predictions
    )

    targets = np.concatenate(
        targets
    )

    metrics = calculate_metrics(
        targets,
        predictions,
    )

    average_loss = (
        running_loss / samples
    )

    return average_loss, metrics


def save_json(
    path: Path,
    data,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:

    config = load_config()

    seed_everything(
        config["project"]["seed"]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = device.type == "cuda"

    print("=" * 70)
    print("MolBind Training")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    train_data = pd.read_parquet(
        ROOT
        / "data"
        / "processed"
        / "train.parquet"
    )

    validation_data = pd.read_parquet(
        ROOT
        / "data"
        / "processed"
        / "validation.parquet"
    )

    test_data = pd.read_parquet(
        ROOT
        / "data"
        / "processed"
        / "test.parquet"
    )

    print(
        f"Train: {len(train_data):,}"
    )

    print(
        f"Validation: {len(validation_data):,}"
    )

    print(
        f"Test: {len(test_data):,}"
    )

    batch_size = config[
        "training"
    ][
        "batch_size"
    ]

    max_protein_length = config[
        "data"
    ][
        "max_protein_length"
    ]

    train_loader = create_dataloader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        max_protein_length=max_protein_length,
        num_workers=config[
            "training"
        ][
            "num_workers"
        ],
    )

    validation_loader = create_dataloader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        max_protein_length=max_protein_length,
        num_workers=config[
            "training"
        ][
            "num_workers"
        ],
    )

    test_loader = create_dataloader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        max_protein_length=max_protein_length,
        num_workers=config[
            "training"
        ][
            "num_workers"
        ],
    )

    model = MolBindModel(
        dropout=config[
            "model"
        ][
            "fusion"
        ][
            "dropout"
        ],
        max_protein_length=max_protein_length,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config[
            "training"
        ][
            "learning_rate"
        ],
        weight_decay=config[
            "training"
        ][
            "weight_decay"
        ],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=use_amp,
    )

    checkpoint_path = (
        ROOT
        / config["paths"]["checkpoint"]
    )

    history_path = (
        ROOT
        / config["paths"]["history"]
    )

    metrics_path = (
        ROOT
        / config["paths"]["metrics"]
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_validation_rmse = float(
        "inf"
    )

    epochs_without_improvement = 0

    epochs = config[
        "training"
    ][
        "epochs"
    ]

    patience = config[
        "training"
    ][
        "patience"
    ]

    gradient_clip = config[
        "training"
    ][
        "gradient_clip"
    ]

    for epoch in range(
        1,
        epochs + 1,
    ):

        print(
            f"\nEpoch {epoch}/{epochs}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            gradient_clip,
        )

        validation_loss, validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            use_amp,
        )

        scheduler.step(
            validation_loss
        )

        learning_rate = optimizer.param_groups[
            0
        ][
            "lr"
        ]

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": learning_rate,
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
        }

        history.append(
            epoch_result
        )

        print(
            f"Train loss: "
            f"{train_loss:.5f}"
        )

        print(
            f"Val loss: "
            f"{validation_loss:.5f}"
        )

        print(
            f"Val RMSE: "
            f"{validation_metrics['rmse']:.5f}"
        )

        print(
            f"Val MAE: "
            f"{validation_metrics['mae']:.5f}"
        )

        print(
            f"Val R²: "
            f"{validation_metrics['r2']:.5f}"
        )

        print(
            f"Val Pearson: "
            f"{validation_metrics['pearson']:.5f}"
        )

        print(
            f"Val Spearman: "
            f"{validation_metrics['spearman']:.5f}"
        )

        print(
            f"LR: "
            f"{learning_rate:.7f}"
        )

        if (
            validation_metrics["rmse"]
            < best_validation_rmse
        ):

            best_validation_rmse = (
                validation_metrics["rmse"]
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": validation_metrics,
                    "config": config,
                },
                checkpoint_path,
            )

            print(
                f"Best model saved: "
                f"{checkpoint_path}"
            )

        else:
            epochs_without_improvement += 1

        save_json(
            history_path,
            history,
        )

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                "\nEarly stopping."
            )
            break

    print("\nLoading best model...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        device,
        use_amp,
    )

    result = {
        "best_epoch": checkpoint[
            "epoch"
        ],
        "validation_metrics": checkpoint[
            "validation_metrics"
        ],
        "test_loss": test_loss,
        "test_metrics": test_metrics,
    }

    save_json(
        metrics_path,
        result,
    )

    print("\n" + "=" * 70)
    print("Final Test Metrics")
    print("=" * 70)

    for key, value in test_metrics.items():
        print(
            f"{key.upper():10s}: "
            f"{value:.6f}"
        )

    print(
        f"\nBest checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Metrics: "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()