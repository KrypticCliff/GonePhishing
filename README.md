# GonePhishing

A Python prototype for classifying phishing vs. legitimate email. Built as a capstone project with emphasis on defensible methodology and clear evaluation.

## What it does

GonePhishing trains and evaluates two classical text classifiers — Logistic Regression and Multinomial Naive Bayes — on TF-IDF features plus handcrafted signals (URL density, urgency-word count, exclamation marks, etc.). After training it reports precision, recall, F1, and a confusion matrix on a held-out test split, and saves the fitted pipelines for use in the web app.

---

## Quick Start

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Download a dataset (phishing vs. legitimate — recommended)
python scripts/download_phishing_data.py

# 3. Train
python train.py --data data\raw\emails.csv

# 4. Run the web app
python app.py
```

Then open <http://127.0.0.1:5000> and upload an `.eml` file to classify it.

---

## Setup

**Requirements:** Python 3.8+

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Data

Two download scripts are provided depending on what you want the model to learn.

### Phishing vs. legitimate — recommended

```powershell
python scripts/download_phishing_data.py
```

Combines the **Nazario phishing corpus** (~177 MB, 2005–2025) and **phishing_pot** (~1,100 community-verified phishing emails) with **SpamAssassin ham** (newsletters, marketing, transactional mail). Best choice for low false-positives on real-world email.

### Spam vs. ham — general-purpose

```powershell
python scripts/download_data.py                              # Enron-Spam (~33 000 emails) — default
python scripts/download_data.py --dataset spamassassin       # SpamAssassin (~6 000 emails)
python scripts/download_data.py --dataset csdmc2010          # CSDMC 2010 (~4 300 emails)
python scripts/download_data.py --dataset lingspam           # Ling-Spam AUEB (~2 800 emails)
python scripts/download_data.py --dataset pu_corpora         # PU1–PU4 AUEB (~5 000 emails)
python scripts/download_data.py --dataset email_dataset      # GitHub .eml corpus (~19 000 emails)
python scripts/download_data.py --dataset trec05             # TREC 2005 spam track (~92 000 emails)
python scripts/download_data.py --dataset trec06             # TREC 2006 spam track (~37 000 emails)
python scripts/download_data.py --dataset trec07             # TREC 2007 spam track (~75 000 emails)
python scripts/download_data.py --dataset ceas08             # CEAS 2008 challenge (~38 000 emails)
python scripts/download_data.py --dataset nigerian           # Nigerian fraud corpus (~5 000 emails)
python scripts/download_data.py --dataset nigerian5          # Nigerian fraud v2 (~10 000 emails)
python scripts/download_data.py --dataset all                # all of the above combined
```

> The TREC and CEAS datasets (Champa et al., Zenodo 8339691, CC BY 4.0) are the largest and most diverse. `--dataset all` downloads everything and may take a while. These datasets tend to over-flag modern marketing email because their "ham" is pre-2010 corporate inbox text.

### Clearing the cache

To switch datasets or re-download from scratch:

```powershell
python scripts/clear_cache.py              # remove downloaded archives only
python scripts/clear_cache.py --csv        # also remove data/raw/emails.csv
python scripts/clear_cache.py --models     # also remove trained models
python scripts/clear_cache.py --all        # remove everything and start fresh
```

### Bring your own dataset

Place it at `data/raw/emails.csv` with columns `text` (email body) and `label` (`1` = phishing/spam, `0` = legitimate/ham).

---

## Training

### Standard run

```powershell
python train.py --data data\raw\emails.csv
```

### With a subject column (e.g. Champa CSVs)

```powershell
python train.py --data data\raw\emails.csv --subject-col subject
```

### Speed vs. accuracy options

Training uses `RandomizedSearchCV` with 3-fold cross-validation by default. Use these commands to control the speed/accuracy trade-off:

```powershell
# Fastest — skip tuning entirely, use default hyperparameters
python train.py --data data\raw\emails.csv --no-tune

# Fast — fewer random combinations (default is 10)
python train.py --data data\raw\emails.csv --n-iter 5

# Balanced — default settings (recommended starting point)
python train.py --data data\raw\emails.csv

# Thorough — more combinations and folds for best accuracy
python train.py --data data\raw\emails.csv --n-iter 20 --cv-folds 5
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | *(required)* | Path to the labeled CSV |
| `--text-col` | `text` | Column containing email text |
| `--label-col` | `label` | Column containing binary labels |
| `--subject-col` | *(none)* | Column to prepend as subject line |
| `--no-tune` | off | Skip hyperparameter search (fastest) |
| `--n-iter` | `10` | Random parameter combinations to try |
| `--cv-folds` | `3` | Stratified CV folds during tuning |
| `--test-size` | `0.2` | Fraction of data held out for evaluation |
| `--seed` | `42` | Random seed for reproducibility |
| `--out` | `models/` | Directory to save fitted pipelines |

Trained pipelines are saved to `models/` (gitignored).

---

## Output

After training, the script prints evaluation metrics for each classifier:

```
=== logreg ===
Best params: {'clf__C': 1.0, 'features__tfidf__ngram_range': (1, 2), ...}
              precision    recall  f1-score   support
           0     0.9900    0.9950    0.9925      4000
           1     0.9950    0.9900    0.9925      4000
confusion matrix:
[[3980   20]
 [  40 3960]]

=== naive_bayes ===
...
```

---

## Web app

Once at least one model has been trained, launch the Flask classifier:

### Start the app

```powershell
python app.py
```

### Custom threshold (reduces false positives on legitimate email)

```powershell
python app.py --threshold 0.85
```

### Custom port

```powershell
python app.py --port 8000
```

Open <http://127.0.0.1:5000>, upload an `.eml` file, and the app returns a verdict (Phishing / Legitimate) with confidence score from each model.

**Threshold tuning.** The upload form includes a *Phishing threshold* slider (0.00–1.00). The verdict only shows "Phishing" when the model's phishing probability meets or exceeds this value. If the model is over-flagging legitimate email, raise the threshold to `0.80`–`0.90`. For a more durable fix, retrain on the phishing-vs-legitimate dataset.

---

## License

MIT
