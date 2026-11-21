import pandas as pd
import pytest

from src.sentiment import model


@pytest.fixture(scope="module")
def trained():
    df = pd.DataFrame({
        "text": [
            "harika",
            "çok güzel",
            "mükemmel",
            "berbat",
            "çok kötü",
            "kötü",
        ],
        "label": [
            "positive",
            "positive",
            "positive",
            "negative",
            "negative",
            "negative",
        ],
    })
    pipe, report = model.train(df)
    return pipe, report


def test_report_is_text(trained):
    assert "precision" in trained[1]


def test_predicts_obvious_sentences(trained):
    pipe, _ = trained
    out = model.predict(pipe, ["çok güzel ve harika bir ürün, memnun kaldım",
                               "berbat ve kalitesiz, asla tavsiye etmem"])
    assert out[0]["label"] == "positive"
    assert out[1]["label"] == "negative"
    assert all(0.0 <= r["confidence"] <= 1.0 for r in out)


def test_save_and_load_roundtrip(trained, tmp_path):
    pipe, _ = trained
    path = tmp_path / "m.joblib"
    model.save(pipe, path)
    again = model.load(path)
    assert model.predict(again, ["harika"])[0]["label"] == model.predict(pipe, ["harika"])[0]["label"]


def test_dataset_validation(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        model.load_dataset(bad)


def test_predict_empty_and_missing_text(trained):
    pipe, _ = trained
    assert model.predict(pipe, ["", "   !!!", None])[0]["confidence"] >= 0.0
    with pytest.raises(ValueError):
        model.predict(pipe, [])


def test_train_requires_two_labels():
    df = pd.DataFrame({"text": ["harika", "kötü"], "label": ["positive", "positive"]})
    with pytest.raises(ValueError):
        model.train(df)


def test_load_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        model.load(tmp_path / "missing.joblib")
