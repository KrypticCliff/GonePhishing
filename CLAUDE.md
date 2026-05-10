# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GonePhishing is a Python prototype for classifying phishing vs. legitimate email. Capstone-scale; emphasis on a defensible methodology and clear evaluation rather than production deployment.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the baseline

`train.py` trains a Logistic Regression and a Multinomial Naive Bayes classifier on TF-IDF features and reports precision/recall/F1 plus a confusion matrix on a held-out split.

```powershell
python train.py --data data\raw\emails.csv --text-col text --label-col label
```

The input CSV must have one column of email text (body, optionally concatenated with the subject) and one column of binary labels. Trained pipelines are written to `models/`.

## Downloading data

`scripts/download_data.py` fetches and preprocesses a public dataset into `data/raw/emails.csv` (stdlib only — no extra dependencies).

```powershell
python scripts/download_data.py                        # Enron-Spam (default, ~33k emails)
python scripts/download_data.py --dataset spamassassin # SpamAssassin (~6k emails)
python scripts/download_data.py --dataset csdmc2010    # CSDMC 2010 (~4.3k emails)
```

Archives are cached in `data/raw/.download_cache/`. Output CSV columns: `text`, `label` (1 = spam, 0 = ham).

## Web app

`app.py` is a Flask app that loads any trained pipelines from `models/` and classifies a single uploaded `.eml` file. It reuses `eml_to_csv.extract_record` for parsing and `train.clean_text` for preprocessing — those two imports are the contract; if you change preprocessing in `train.py`, the app inherits it automatically.

```powershell
python app.py   # http://127.0.0.1:5000
```

## Layout

- `train.py` — baseline training + evaluation script
- `eml_to_csv.py` — convert raw .eml/mbox corpora to a labeled CSV
- `app.py` — Flask web app for classifying uploaded .eml files
- `templates/` — Jinja templates for the Flask app
- `scripts/download_data.py` — dataset download + preprocessing
- `requirements.txt` — runtime dependencies
- `data/raw/` — labeled email datasets (gitignored)
- `models/` — fitted scikit-learn pipelines (gitignored)
