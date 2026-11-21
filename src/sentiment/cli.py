"""Command line interface: train / predict."""
import argparse

from . import model


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

    args = ap.parse_args()
    if args.cmd == "train":
        pipe, report = model.train(model.load_dataset(args.data), C=args.C)
        print(report)
        model.save(pipe, args.out)
        print(f"Saved model to {args.out}")
    else:
        for r in model.predict(model.load(args.model), args.texts):
            print(f"{r['label']:>9}  {r['confidence']:.2f}  {r['text']}")


if __name__ == "__main__":
    main()
