#!/usr/bin/env python3
"""Download phishing + ham corpora, then build a labeled CSV via eml_to_csv.convert.

This produces a *phishing-vs-legitimate* dataset (as opposed to the *spam-vs-ham*
dataset built by scripts/download_data.py). Use this when you want the model to
distinguish phishing attempts from ordinary newsletters / marketing / transactional
email — not just spam.

Phishing sources:
  - Nazario corpus  (https://monkey.org/~jose/phishing/)       ~177 MB, 2005–2025
  - phishing_pot    (https://github.com/rf-peixoto/phishing_pot) ~134 MB, 1 100+ emails

Ham source:
  - SpamAssassin public corpus (easy_ham, easy_ham_2, hard_ham — 2003 batch)
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
import urllib.request
from pathlib import Path

# Make project root importable so we can call eml_to_csv.convert directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eml_to_csv import convert  # noqa: E402

DATA_DIR = ROOT / "data" / "raw"
NAZARIO_DIR = DATA_DIR / "nazario"
HAM_DIR = DATA_DIR / "spamassassin_ham"
CACHE_DIR = DATA_DIR / ".download_cache"

NAZARIO_BASE = "https://monkey.org/~jose/phishing/"
NAZARIO_FILES = [
    "20051114.mbox",
    "phishing0.mbox",
    "phishing1.mbox",
    "phishing2.mbox",
    "phishing3.mbox",
    "private-phishing4.mbox",
    "phishing-2015",
    "phishing-2016",
    "phishing-2017",
    "phishing-2018",
    "phishing-2019",
    "phishing-2020",
    "phishing-2021",
    "phishing-2022",
    "phishing-2023",
    "phishing-2024",
    "phishing-2025",
]

# 2003 batch only — it supersedes the 2002 release and avoids near-duplicate emails
SPAMASSASSIN_HAM_URLS = [
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
]

PHISHING_POT_URL = "https://github.com/rf-peixoto/phishing_pot/archive/refs/heads/main.zip"
PHISHING_POT_DIR = DATA_DIR / "phishing_pot"

USER_AGENT = "GonePhishing-dataset-download/1.0 (educational use)"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name}")
        return dest
    print(f"  downloading {dest.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as f:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return dest


def download_nazario() -> list[Path]:
    NAZARIO_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in NAZARIO_FILES:
        paths.append(fetch(NAZARIO_BASE + name, NAZARIO_DIR / name))
    return paths


def download_ham() -> list[Path]:
    HAM_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    top_dirs: set[Path] = set()
    for url in SPAMASSASSIN_HAM_URLS:
        archive = fetch(url, CACHE_DIR / Path(url).name)
        with tarfile.open(archive, "r:bz2") as tf:
            tops = {Path(m.name).parts[0] for m in tf.getmembers() if m.name}
            if all((HAM_DIR / t).exists() for t in tops):
                print(f"  already extracted: {archive.name}")
            else:
                print(f"  extracting {archive.name} ...")
                tf.extractall(HAM_DIR)
            top_dirs.update(HAM_DIR / t for t in tops)
    return sorted(top_dirs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def download_phishing_pot() -> list[Path]:
    """Download rf-peixoto/phishing_pot and extract .eml files to data/raw/phishing_pot/."""
    PHISHING_POT_DIR.mkdir(parents=True, exist_ok=True)
    archive = fetch(PHISHING_POT_URL, CACHE_DIR / "phishing_pot_main.zip")
    with zipfile.ZipFile(archive) as zf:
        eml_entries = [n for n in zf.namelist()
                       if n.startswith("phishing_pot-main/email/") and n.endswith(".eml")]
        already = list(PHISHING_POT_DIR.glob("*.eml"))
        if len(already) >= len(eml_entries):
            print(f"  already extracted: {len(already)} .eml files")
        else:
            print(f"  extracting {len(eml_entries)} .eml files ...")
            for entry in eml_entries:
                dest = PHISHING_POT_DIR / Path(entry).name
                dest.write_bytes(zf.read(entry))
    return [PHISHING_POT_DIR]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "emails.csv",
        help="output CSV path (default: data/raw/emails.csv)",
    )
    args = parser.parse_args()

    print("== Nazario phishing corpus ==")
    phishing_paths = download_nazario()

    print("\n== phishing_pot ==")
    phishing_paths += download_phishing_pot()

    print("\n== SpamAssassin ham ==")
    ham_paths = download_ham()

    print(f"\n== Building labeled CSV at {args.out} ==")
    counts = convert(phishing_paths, ham_paths, args.out)

    print("\nDone.")
    print(f"  phishing:    {counts['phishing']:,}")
    print(f"  legitimate:  {counts['legitimate']:,}")
    print(f"  skipped:     {counts['skipped_empty']:,}")
    print(f"\nNext: python train.py --data {args.out}")


if __name__ == "__main__":
    main()
