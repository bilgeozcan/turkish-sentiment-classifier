import pandas as pd

from src.sentiment.evaluation import cross_validate, find_misclassified, save_confusion_matrix
from src.sentiment.model import build_pipeline


def test_cross_validate_and_confusion_matrix(tmp_path):
    df = pd.DataFrame({
        "text": [
            "harika",
            "çok güzel",
            "güzel",
            "berbat",
            "kötü",
            "çok kötü",
        ],
        "label": ["positive", "positive", "positive", "negative", "negative", "negative"],
    })
    result = cross_validate(df, folds=3)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0
    output = save_confusion_matrix(
        result["confusion_matrix"], result["labels"], tmp_path / "confusion.png"
    )
    assert output.exists()


def test_find_misclassified():
    df = pd.DataFrame({
        "text": ["harika", "kötü", "çok güzel", "berbat"],
        "label": ["positive", "negative", "positive", "negative"],
    })
    pipe = build_pipeline().fit(df["text"], df["label"])
    result = find_misclassified(pipe, df)
    assert "predicted_label" in result.columns
