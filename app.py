"""Flask web app: upload an .eml file and get a phishing-classification verdict
from each trained pipeline.

Run:
    python app.py                    # default threshold 0.5
    python app.py --threshold 0.85   # only flag as Phishing when very confident
    python app.py --port 8000
"""

from __future__ import annotations

import argparse
import email
import email.policy
from pathlib import Path

import joblib
from flask import Flask, render_template, request

from eml_to_csv import extract_record
from train import clean_text, extract_handcrafted_features

MODELS_DIR = Path("models")
MODEL_FILES = {
    "Logistic Regression": MODELS_DIR / "logreg.joblib",
    "Multinomial Naive Bayes": MODELS_DIR / "naive_bayes.joblib",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def load_models() -> dict:
    models = {}
    for name, path in MODEL_FILES.items():
        if path.exists():
            models[name] = joblib.load(path)
    if not models:
        raise FileNotFoundError(
            f"No trained models found in {MODELS_DIR}/. Run train.py first."
        )
    return models


def parse_eml(raw: bytes) -> dict:
    msg = email.message_from_bytes(raw, policy=email.policy.compat32)
    return extract_record(msg, label="?", source="upload")


def classify(models: dict, text: str, threshold: float) -> list[dict]:
    """Run each model and apply `threshold` to the phishing probability.

    A higher threshold makes the "Phishing" verdict stricter — useful when the
    model over-flags legitimate marketing/transactional email.
    """
    cleaned = clean_text(text)
    results = []
    for name, pipe in models.items():
        if hasattr(pipe, "predict_proba"):
            phishing_score = float(pipe.predict_proba([cleaned])[0][1])
            is_phishing = phishing_score >= threshold
            confidence = phishing_score if is_phishing else 1.0 - phishing_score
        else:
            pred = int(pipe.predict([cleaned])[0])
            is_phishing = pred == 1
            phishing_score = float(pred)
            confidence = None
        results.append({
            "model": name,
            "verdict": "Phishing" if is_phishing else "Legitimate",
            "phishing_score": phishing_score,
            "confidence": confidence,
        })
    return results


def _parse_threshold(raw: str | None, default: float) -> float:
    if raw is None or raw == "":
        return default
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return default


def create_app(default_threshold: float = 0.5) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    models = load_models()

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            models=list(models),
            threshold=default_threshold,
        )

    @app.post("/predict")
    def predict():
        threshold = _parse_threshold(request.form.get("threshold"), default_threshold)

        upload = request.files.get("email")
        if not upload or upload.filename == "":
            return (
                render_template("index.html", models=list(models),
                                threshold=threshold,
                                error="No file uploaded."),
                400,
            )

        raw = upload.read()
        try:
            record = parse_eml(raw)
        except Exception as exc:
            return (
                render_template("index.html", models=list(models),
                                threshold=threshold,
                                error=f"Could not parse .eml: {exc}"),
                400,
            )

        if not record["text"]:
            return (
                render_template("index.html", models=list(models),
                                threshold=threshold,
                                error="Email body is empty after parsing."),
                400,
            )

        results = classify(models, record["text"], threshold=threshold)
        return render_template(
            "result.html",
            filename=upload.filename,
            record=record,
            results=results,
            threshold=threshold,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="GonePhishing web app.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="default phishing-probability threshold (0.0–1.0). "
             "Higher = stricter (fewer false positives). Default 0.5.",
    )
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0.0 and 1.0")

    create_app(default_threshold=args.threshold).run(
        host=args.host, port=args.port, debug=True
    )


if __name__ == "__main__":
    main()
