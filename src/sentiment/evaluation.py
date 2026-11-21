"""Cross-validation and error analysis utilities."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from .model import build_pipeline


def cross_validate(
    df: pd.DataFrame,
    folds: int = 5,
    seed: int = 42,
    C: float = 3.0,
) -> dict[str, float | np.ndarray]:
    counts = df["label"].value_counts()
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if counts.min() < folds:
        raise ValueError(
            f"Each label needs at least {folds} examples for stratified cross-validation"
        )

    labels = sorted(str(label) for label in df["label"].unique())
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_true: list[str] = []
    y_pred: list[str] = []
    X = df["text"]
    y = df["label"].map(str)

    for train_index, test_index in splitter.split(X, y):
        pipe = build_pipeline(C)
        pipe.fit(X.iloc[train_index], y.iloc[train_index])
        predictions = pipe.predict(X.iloc[test_index])
        y_true.extend(y.iloc[test_index].tolist())
        y_pred.extend(predictions.tolist())

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": matrix,
        "labels": labels,
    }


def save_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    output: str | Path,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def find_misclassified(pipe, df: pd.DataFrame) -> pd.DataFrame:
    texts = df["text"].tolist()
    predictions = pipe.predict(texts)
    result = df.copy()
    result["predicted_label"] = predictions
    return result[result["label"].map(str) != result["predicted_label"].map(str)]
