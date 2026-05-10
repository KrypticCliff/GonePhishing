"""Baseline phishing-email classifier: Logistic Regression vs. Multinomial Naive Bayes.

Reads a CSV with text and label columns, trains both models on TF-IDF features,
prints metrics on a held-out test split, and saves each fitted pipeline.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


URL_RE = re.compile(r"http\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" URL ", text)
    text = EMAIL_RE.sub(" EMAIL ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def load_dataset(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"expected columns {text_col!r} and {label_col!r}; found {list(df.columns)}"
        )
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    df["text"] = df["text"].astype(str).map(clean_text)
    return df


def evaluate(name: str, model: Pipeline, X_test, y_test) -> None:
    preds = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, digits=4))
    print("confusion matrix:")
    print(confusion_matrix(y_test, preds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="path to labeled CSV")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    df = load_dataset(args.data, args.text_col, args.label_col)
    print(f"loaded {len(df)} rows")
    print("class balance:")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"],
    )

    tfidf_kwargs = dict(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
    models = {
        "logreg": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "naive_bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
                ("clf", MultinomialNB()),
            ]
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        evaluate(name, pipe, X_test, y_test)
        path = args.out / f"{name}.joblib"
        joblib.dump(pipe, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
