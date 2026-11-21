"""Command line interface: train / predict."""
import argparse
from pathlib import Path

from . import model
from .evaluation import cross_validate, find_misclassified, save_confusion_matrix


def main() -> None:
    ap = argparse.ArgumentParser(prog="sentiment", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train and save a model")
    t.add_argument("--data", default="data/sample.csv", help="CSV with columns text,label")
    t.add_argument("--out", default="models/sentiment.joblib")
    t.add_argument("--C", type=float, default=3.0)

    p = sub.add_parser("predict", help="predict sentiment of sentences")
    p.add_argument("--model", default="models/sentiment.joblib")
    p.add_argument("texts", nargs="+")

    e = sub.add_parser("evaluate", help="run stratified k-fold cross-validation")
    e.add_argument("--data", default="data/sample.csv")
    e.add_argument("--folds", type=int, default=5)
    e.add_argument("--C", type=float, default=3.0)
    e.add_argument("--output", default="reports/confusion_matrix.png")

    x = sub.add_parser("errors", help="list misclassified examples")
    x.add_argument("--data", default="data/sample.csv")
    x.add_argument("--model", default="models/sentiment.joblib")
    x.add_argument("--limit", type=int, default=20)
    x.add_argument("--output", default="reports/misclassified.csv")

    args = ap.parse_args()
    if args.cmd == "train":
        pipe, report = model.train(model.load_dataset(args.data), C=args.C)
        print(report)
        model.save(pipe, args.out)
        print(f"Saved model to {args.out}")
    elif args.cmd == "predict":
        try:
            pipe = model.load(args.model)
        except FileNotFoundError:
            ap.error(f"Model file not found: {args.model}")
        for r in model.predict(pipe, args.texts):
            print(f"{r['label']:>9}  {r['confidence']:.2f}  {r['text']}")
    elif args.cmd == "evaluate":
        df = model.load_dataset(args.data)
        result = cross_validate(df, folds=args.folds, C=args.C)
        matrix_path = save_confusion_matrix(
            result["confusion_matrix"], result["labels"], args.output
        )
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Macro-F1: {result['macro_f1']:.4f}")
        print(f"Saved confusion matrix to {matrix_path}")
    else:
        try:
            pipe = model.load(args.model)
        except FileNotFoundError:
            ap.error(f"Model file not found: {args.model}")
        df = model.load_dataset(args.data)
        errors = find_misclassified(pipe, df).head(args.limit)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        errors.to_csv(output, index=False)
        print(f"Misclassified examples: {len(errors)}")
        print(f"Saved misclassified examples to {output}")


if __name__ == "__main__":
    main()
