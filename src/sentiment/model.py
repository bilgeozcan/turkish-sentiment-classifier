"""Model construction, training, persistence and prediction."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from .preprocess import normalize


def build_pipeline(C: float = 3.0) -> Pipeline:
    """TF-IDF (word 1-2 grams + char 2-5 grams) -> Logistic Regression.
    Character n-grams help with Turkish agglutination (suffixes)."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb",
                                 ngram_range=(2, 5), sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns text,label (missing: {sorted(missing)})")
    return df.dropna(subset=["text", "label"])


def train(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42, C: float = 3.0):
    """Train on a stratified split; returns (pipeline, text report string)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["label"], test_size=test_size, random_state=seed, stratify=df["label"])
    pipe = build_pipeline(C).fit(X_tr, y_tr)
    report = classification_report(y_te, pipe.predict(X_te), zero_division=0)
    return pipe, report


def save(pipe: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load(path: str | Path) -> Pipeline:
    return joblib.load(path)


def predict(pipe: Pipeline, texts: list[str]) -> list[dict]:
    """Return label and confidence for every input text."""
    probs = pipe.predict_proba(texts)
    return [{"text": t, "label": str(pipe.classes_[p.argmax()]), "confidence": float(p.max())}
            for t, p in zip(texts, probs)]
