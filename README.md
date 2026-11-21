# 🇹🇷 Turkish Sentiment Analysis (NLP)

A clean, testable baseline for **Turkish text sentiment classification** using scikit-learn:
Turkish-aware normalization → word + character TF-IDF n-grams → Logistic Regression.
Character n-grams help with Turkish agglutinative morphology without needing a stemmer.

## Features
- Turkish-correct lowercasing (`I → ı`, `İ → i`), URL/mention/punctuation cleaning, letter-repetition squeezing
- `FeatureUnion` of word (1-2) and char (2-5) TF-IDF features
- CLI for training and prediction, model persistence with `joblib`
- Unit tests with `pytest`
- Bring your own dataset (CSV with `text,label`)

> ⚠️ `data/sample.csv` is a tiny toy dataset (48 sentences) for demos and tests only.
> Train on a real dataset (e.g. product/movie reviews or tweets) for meaningful results.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.sentiment.cli train --data data/sample.csv
python -m src.sentiment.cli predict "ürün harika, çok memnun kaldım" "kargo berbattı, bir daha almam"
pytest
```

## Project structure
```
├── data/sample.csv           # toy dataset
├── src/sentiment/
│   ├── preprocess.py         # Turkish normalization
│   ├── model.py              # pipeline, train, save/load, predict
│   └── cli.py                # command line interface
└── tests/
```

## Results
| Dataset | Model | Accuracy | Macro-F1 |
|---------|-------|----------|----------|
| _your dataset_ | TF-IDF + LR | _TBD_ | _TBD_ |

## Ideas / roadmap
- Compare with a fine-tuned Turkish BERT (e.g. BERTurk) using Hugging Face Transformers
- Add neutral class, cross-validation and error analysis notebook
- Serve the model with FastAPI + Docker

## Author
Nur Bilge Özcan · [GitHub](https://github.com/BILGEOZCAN) · [LinkedIn](https://linkedin.com/in/nur-bilge-özcan-123975257)

Licensed under the MIT License.
