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

## Layout

- `train.py` — baseline training + evaluation script
- `requirements.txt` — runtime dependencies
- `data/raw/` — labeled email datasets (gitignored)
- `models/` — fitted scikit-learn pipelines (gitignored)
