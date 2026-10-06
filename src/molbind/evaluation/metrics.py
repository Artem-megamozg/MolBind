from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def calculate_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:

    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    rmse = np.sqrt(
        mean_squared_error(
            targets,
            predictions,
        )
    )

    mae = mean_absolute_error(
        targets,
        predictions,
    )

    r2 = r2_score(
        targets,
        predictions,
    )

    pearson = pearsonr(
        targets,
        predictions,
    ).statistic

    spearman = spearmanr(
        targets,
        predictions,
    ).statistic

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "pearson": float(pearson),
        "spearman": float(spearman),
    }