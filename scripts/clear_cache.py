#!/usr/bin/env python3
"""Remove downloaded dataset archives and extracted directories so you can
re-download from scratch or switch to a different dataset.

By default removes only the cached archives and extracted source directories,
leaving emails.csv and trained models untouched.

Examples
--------
python scripts/clear_cache.py              # clear downloads only
python scripts/clear_cache.py --csv        # also remove data/raw/emails.csv
python scripts/clear_cache.py --models     # also remove trained models
python scripts/clear_cache.py --all        # remove everything
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

# Directories created by the download scripts
CACHE_DIRS = {
    "download cache":   DATA_DIR / ".download_cache",
    "nazario corpus":   DATA_DIR / "nazario",
    "spamassassin ham": DATA_DIR / "spamassassin_ham",
    "phishing_pot":     DATA_DIR / "phishing_pot",
}


def remove(label: str, path: Path) -> None:
    if not path.exists():
        print(f"  skip  {label} (not found)")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"  removed {label} ({path.relative_to(ROOT)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", action="store_true",
                        help="also remove data/raw/emails.csv")
    parser.add_argument("--models", action="store_true",
                        help="also remove trained models in models/")
    parser.add_argument("--all", dest="everything", action="store_true",
                        help="remove everything (downloads, emails.csv, models)")
    args = parser.parse_args()

    print("Clearing cached downloads...")
    for label, path in CACHE_DIRS.items():
        remove(label, path)

    if args.csv or args.everything:
        remove("emails.csv", DATA_DIR / "emails.csv")

    if args.models or args.everything:
        remove("models", ROOT / "models")

    print("\nDone. Re-run a download script to fetch fresh data.")


if __name__ == "__main__":
    main()
