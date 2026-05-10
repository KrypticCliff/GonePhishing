"""Phishing-email classifier: Logistic Regression vs. Multinomial Naive Bayes.

Features: TF-IDF bigrams + handcrafted numerical features (URL density, urgency
word count, exclamation marks, etc.) combined via FeatureUnion.

Hyperparameters are tuned with stratified k-fold cross-validation by default.
Use --no-tune to skip tuning and train with defaults (faster).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer


URL_RE = re.compile(r"http\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
WS_RE = re.compile(r"\s+")
URGENCY_RE = re.compile(
    r"\b(urgent|immediately|verify|suspend(?:ed)?|account|password|click|confirm|"
    r"update|expir(?:e|ed|ing)|login|security|alert|warning|limited|act\s+now|"
    r"bank|free|prize|winner|congratulations|offer|"
    r"dear\s+(?:user|customer|member|friend))\b",
    re.I,
)


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" URL ", text)
    text = EMAIL_RE.sub(" EMAIL ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def extract_handcrafted_features(texts) -> csr_matrix:
    """Numerical signals that complement bag-of-words for phishing detection.

    Works on already-cleaned text (URLs → 'URL', emails → 'EMAIL').
    Returns a sparse matrix so FeatureUnion can stack it with TF-IDF output.
    """
    rows = []
    for text in texts:
        n_url = text.count("URL")
        n_email_tok = text.count("EMAIL")
        n_urgency = len(URGENCY_RE.findall(text))
        n_exclaim = text.count("!")
        n_question = text.count("?")
        n_words = max(len(text.split()), 1)
        rows.append([
            n_url,
            n_email_tok,
            n_urgency,
            n_exclaim,
            n_question,
            n_words,
            n_url / n_words,      # URL density
            n_urgency / n_words,  # urgency density
        ])
    return csr_matrix(np.array(rows, dtype=float))


def load_dataset(
    path: Path,
    text_col: str,
    label_col: str,
    subject_col: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = [c for c in [text_col, label_col, subject_col] if c]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; found {list(df.columns)}")
    df = df.dropna(subset=[text_col, label_col])
    if subject_col:
        # Prepend subject so the model can learn from it
        df["text"] = (
            df[subject_col].fillna("").astype(str)
            + "\n\n"
            + df[text_col].astype(str)
        )
    else:
        df["text"] = df[text_col].astype(str)
    df["label"] = df[label_col]
    df["text"] = df["text"].map(clean_text)
    return df[["text", "label"]]


def evaluate(name: str, model, X_test, y_test) -> None:
    preds = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, digits=4))
    print("confusion matrix:")
    print(confusion_matrix(y_test, preds))


def build_models() -> dict:
    hand = FunctionTransformer(extract_handcrafted_features, validate=False)
    tfidf_kw = dict(min_df=2, max_df=0.95, sublinear_tf=True, max_features=50_000)
    return {
        "logreg": Pipeline([
            ("features", FeatureUnion([
                ("tfidf", TfidfVectorizer(**tfidf_kw)),
                ("hand", hand),
            ])),
            ("clf", LogisticRegression(solver="saga", max_iter=5000, tol=1e-3, class_weight="balanced")),
        ]),
        "naive_bayes": Pipeline([
            ("features", FeatureUnion([
                ("tfidf", TfidfVectorizer(**tfidf_kw)),
                ("hand", hand),
            ])),
            ("clf", MultinomialNB()),
        ]),
    }


PARAM_GRIDS = {
    "logreg": {
        "features__tfidf__ngram_range": [(1, 1), (1, 2)],
        "features__tfidf__max_features": [30_000, 50_000, None],
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    },
    "naive_bayes": {
        "features__tfidf__ngram_range": [(1, 1), (1, 2)],
        "features__tfidf__max_features": [30_000, 50_000, None],
        "clf__alpha": [0.01, 0.1, 0.5, 1.0],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="path to labeled CSV")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument(
        "--subject-col", default=None,
        help="column to prepend as subject (e.g. 'subject' for Champa CSVs)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument(
        "--cv-folds", type=int, default=3,
        help="stratified CV folds for hyperparameter tuning (default: 3)",
    )
    parser.add_argument(
        "--n-iter", type=int, default=10,
        help="random parameter combinations to try per model (default: 10)",
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="skip tuning and train with default hyperparameters (fastest)",
    )
    args = parser.parse_args()

    df = load_dataset(args.data, args.text_col, args.label_col, args.subject_col)
    print(f"loaded {len(df):,} rows")
    print(df["label"].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"],
    )

    args.out.mkdir(parents=True, exist_ok=True)
    models = build_models()
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    for name, pipe in models.items():
        if args.no_tune:
            pipe.fit(X_train, y_train)
            best = pipe
        else:
            print(f"\nTuning {name} ({args.n_iter} combinations, {args.cv_folds}-fold CV) ...")
            search = RandomizedSearchCV(
                pipe, PARAM_GRIDS[name],
                n_iter=args.n_iter,
                cv=cv, scoring="f1", n_jobs=-1,
                random_state=args.seed, verbose=1,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            print(f"Best params: {search.best_params_}")

        evaluate(name, best, X_test, y_test)
        out_path = args.out / f"{name}.joblib"
        joblib.dump(best, out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
