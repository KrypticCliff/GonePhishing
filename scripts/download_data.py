#!/usr/bin/env python3
"""Download and preprocess a labeled email dataset for phishing/spam classification.

Output: data/raw/emails.csv  (columns: text, label — 1 = spam, 0 = ham)

Datasets
--------
enron        ~33 000 emails from 6 Enron mailboxes + spam (default)
             Source: Metsis et al., aueb.gr
spamassassin ~6 000 emails from the SpamAssassin public corpus
             Source: Apache SpamAssassin project
csdmc2010    ~4 300 emails from the CSDMC 2010 spam competition
             Source: hexgnu/spam_filter mirror on GitHub
"""

import argparse
import csv
import email
import io
import sys
import tarfile
import zipfile
import urllib.request
from pathlib import Path

# Champa CSVs include very long body fields.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DATA_DIR = Path("data/raw")
TMP_DIR = DATA_DIR / ".download_cache"

ENRON_URLS = [
    f"https://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron{i}.tar.gz"
    for i in range(1, 7)
]

# Tuples of (url, label) — label derived from archive name, set explicitly to avoid ambiguity
SPAMASSASSIN_URLS = [
    ("https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2", 0),
    ("https://spamassassin.apache.org/old/publiccorpus/20021010_hard_ham.tar.bz2", 0),
    ("https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2", 1),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2", 0),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2", 0),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2", 0),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2", 1),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2", 1),
    ("https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2", 1),
]

CSDMC2010_URL = "https://github.com/hexgnu/spam_filter/archive/refs/heads/master.zip"

# New datasets
LINGSPAM_URL = "http://pages.aueb.gr/users/ion/data/lingspam_public.tar.gz"
PU_CORPORA_URL = "http://pages.aueb.gr/users/ion/data/PU123ACorpora.tar.gz"
EMAIL_DATASET_URL = "https://github.com/realprogrammersusevim/email-dataset/archive/refs/heads/main.zip"

# Champa et al. curated datasets (Zenodo 8339691) — CSV with subject/body/label columns
_CHAMPA = "https://zenodo.org/api/records/8339691/files/{}/content"
CHAMPA_URLS = {
    "trec05":    (_CHAMPA.format("TREC_05.csv"),       "TREC_05.csv"),
    "trec06":    (_CHAMPA.format("TREC_06.csv"),       "TREC_06.csv"),
    "trec07":    (_CHAMPA.format("TREC_07.csv"),       "TREC_07.csv"),
    "ceas08":    (_CHAMPA.format("CEAS_08.csv"),       "CEAS_08.csv"),
    "nigerian":  (_CHAMPA.format("Nigerian_Fraud.csv"), "Nigerian_Fraud.csv"),
    "nigerian5": (_CHAMPA.format("Nigerian_5.csv"),    "Nigerian_5.csv"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"  Cached: {dest.name}")
        return dest
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    return dest


def decode(raw: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def email_body(raw: bytes) -> str:
    """Extract subject + plain-text body from a raw RFC 2822 email."""
    msg = email.message_from_bytes(raw)
    subject = msg.get("Subject", "") or ""
    parts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            payload = part.get_payload(decode=True)
            if payload:
                parts.append(decode(payload))
    body = " ".join(parts) if parts else decode(raw)
    return f"{subject}\n\n{body}".strip() if subject else body


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_enron(tmp: Path) -> list[tuple[str, int]]:
    """
    Preprocessed Enron-Spam dataset (Metsis et al.).
    Files are plain text (headers stripped); label is inferred from subfolder.
    """
    rows = []
    for url in ENRON_URLS:
        archive = fetch(url, tmp / Path(url).name)
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                parts = Path(member.name).parts  # (enronN, ham|spam, filename)
                if len(parts) < 3 or parts[-2] not in ("ham", "spam"):
                    continue
                label = 1 if parts[-2] == "spam" else 0
                f = tf.extractfile(member)
                if f:
                    rows.append((decode(f.read()), label))
    return rows


def build_spamassassin(tmp: Path) -> list[tuple[str, int]]:
    """
    SpamAssassin public corpus.
    Files are raw RFC 2822 emails; label is set per-archive.
    """
    rows = []
    for url, label in SPAMASSASSIN_URLS:
        archive = fetch(url, tmp / Path(url).name)
        with tarfile.open(archive, "r:bz2") as tf:
            for member in tf.getmembers():
                if not member.isfile() or member.name.endswith("cmds"):
                    continue
                f = tf.extractfile(member)
                if f:
                    rows.append((email_body(f.read()), label))
    return rows


def build_csdmc2010(tmp: Path) -> list[tuple[str, int]]:
    """
    CSDMC 2010 spam corpus (hexgnu/spam_filter GitHub mirror).
    SPAMTrain.label convention: 1 = ham, 0 = spam (inverted relative to ours).
    """
    archive = fetch(CSDMC2010_URL, tmp / "csdmc2010_master.zip")
    rows = []
    with zipfile.ZipFile(archive) as zf:
        label_file = next(n for n in zf.namelist() if n.endswith("SPAMTrain.label"))
        raw_labels = zf.read(label_file).decode().strip().splitlines()
        # Invert: file uses 1=ham/0=spam; we use 1=spam/0=ham
        # Each line may be "0" or "0 TRAIN_00000.eml" — take first token only
        labels = [1 - int(line.split()[0]) for line in raw_labels if line.strip()]

        eml_files = sorted(
            n for n in zf.namelist()
            if "/TRAINING/TRAIN_" in n and n.endswith(".eml")
        )
        for i, path in enumerate(eml_files):
            rows.append((email_body(zf.read(path)), labels[i]))
    return rows


def build_lingspam(tmp: Path) -> list[tuple[str, int]]:
    """Ling-Spam (Androutsopoulos et al.) — linguistics mailing list ham + spam."""
    archive = fetch(LINGSPAM_URL, tmp / "lingspam_public.tar.gz")
    rows = []
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            folder = Path(member.name).parts[-2] if len(Path(member.name).parts) >= 2 else ""
            if folder.startswith("spmsg"):
                label = 1
            elif folder.startswith("legit"):
                label = 0
            else:
                continue
            f = tf.extractfile(member)
            if f:
                rows.append((decode(f.read()), label))
    return rows


def build_pu_corpora(tmp: Path) -> list[tuple[str, int]]:
    """PU1–PU4 corpora (Androutsopoulos et al.) — personal inbox ham + spam."""
    archive = fetch(PU_CORPORA_URL, tmp / "PU123ACorpora.tar.gz")
    rows = []
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            folder = Path(member.name).parts[-2] if len(Path(member.name).parts) >= 2 else ""
            if folder == "spam":
                label = 1
            elif folder == "nonspam":
                label = 0
            else:
                continue
            f = tf.extractfile(member)
            if f:
                rows.append((decode(f.read()), label))
    return rows


def build_email_dataset(tmp: Path) -> list[tuple[str, int]]:
    """realprogrammersusevim/email-dataset — .eml files, folder 1=ham / 2=spam."""
    archive = fetch(EMAIL_DATASET_URL, tmp / "email_dataset_main.zip")
    rows = []
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not name.endswith(".eml"):
                continue
            parts = Path(name).parts
            folder = parts[-2] if len(parts) >= 2 else ""
            if folder == "1":
                label = 0
            elif folder == "2":
                label = 1
            else:
                continue
            rows.append((email_body(zf.read(name)), label))
    return rows


def _build_champa(key: str) -> "Callable[[Path], list[tuple[str, int]]]":
    """Factory that returns a builder for one Champa Zenodo CSV."""
    url, filename = CHAMPA_URLS[key]

    def builder(tmp: Path) -> list[tuple[str, int]]:
        dest = fetch(url, tmp / filename)
        rows = []
        with dest.open(encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    label = int((row.get("label") or "").strip())
                except ValueError:
                    continue
                subj = row.get("subject") or ""
                body = row.get("body") or ""
                text = f"{subj} {body}".strip()
                if text:
                    rows.append((text, label))
        return rows

    builder.__doc__ = f"Champa et al. Zenodo 8339691 — {filename}."
    return builder


DATASETS: "dict[str, Callable[[Path], list[tuple[str, int]]]]" = {
    "enron":        build_enron,
    "spamassassin": build_spamassassin,
    "csdmc2010":    build_csdmc2010,
    "lingspam":     build_lingspam,
    "pu_corpora":   build_pu_corpora,
    "email_dataset": build_email_dataset,
    "trec05":       _build_champa("trec05"),
    "trec06":       _build_champa("trec06"),
    "trec07":       _build_champa("trec07"),
    "ceas08":       _build_champa("ceas08"),
    "nigerian":     _build_champa("nigerian"),
    "nigerian5":    _build_champa("nigerian5"),
}


def build_all(tmp: Path) -> list[tuple[str, int]]:
    """All datasets combined."""
    rows = []
    for name, builder in DATASETS.items():
        if name == "all":
            continue
        print(f"\n-- {name} --")
        rows.extend(builder(tmp))
    return rows


DATASETS["all"] = build_all


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess an email dataset for spam classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k}" for k in DATASETS),
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS),
        default="enron",
        metavar="DATASET",
        help=(
            "Dataset to download (default: enron). Choices: "
            + ", ".join(k for k in DATASETS if k != "all")
            + ", all"
        ),
    )
    parser.add_argument(
        "--out",
        default=str(DATA_DIR / "emails.csv"),
        help="Output CSV path (default: data/raw/emails.csv)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset : {args.dataset}")
    rows = DATASETS[args.dataset](TMP_DIR)

    print(f"Writing {len(rows):,} rows to {out} ...")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    spam = sum(1 for _, label in rows if label == 1)
    ham = len(rows) - spam
    print(f"Done.    {len(rows):,} emails — {spam:,} spam / {ham:,} ham")
    print(f"Output:  {out}")


if __name__ == "__main__":
    main()
