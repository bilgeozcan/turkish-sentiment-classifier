.PHONY: install lint test run serve docker train

install:
	python -m pip install -r requirements.txt

lint:
	python -m ruff check .

test:
	python -m pytest -q

run:
	python -m src.sentiment.cli predict --model models/sentiment.joblib

serve:
	python -m uvicorn src.sentiment.app:app --host 0.0.0.0 --port 8000

docker:
	docker build -t turkish-sentiment-classifier:local .

train:
	python -m src.sentiment.cli train --data data/sample.csv --out models/sentiment.joblib
