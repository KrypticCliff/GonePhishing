"""Convert Nazario phishing + SpamAssassin ham corpora into a single labeled CSV
suitable for train.py.

Each --phishing / --ham argument may be one or more of:
  - an mbox file (Nazario distributes messages this way)
  - a directory of one-message-per-file emails (SpamAssassin layout)
  - a single .eml file

Output CSV columns: text, label, subject, sender, source
where `text` is "<subject>\\n\\n<body>" so train.py works with `--text-col text`.
"""

from __future__ import annotations

import argparse
import csv
import email
import email.policy
import mailbox
import sys
from email.header import decode_header, make_header
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator


# CSV cells can grow large for long phishing bodies; raise the default limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML -> text extractor; drops <script> and <style> contents."""

    _SKIP_TAGS = {"script", "style"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        return " ".join(self._chunks)


def _strip_html(html: str) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return html
    return parser.text()


def _decode_part(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("utf-8", errors="replace")


def extract_body(msg: Message) -> str:
    """Best-effort plaintext body. Prefers text/plain; falls back to text/html stripped."""
    plain_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue
            disp = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disp:
                continue
            ctype = part.get_content_type()
            if ctype == "text/plain":
                plain_parts.append(_decode_part(part))
            elif ctype == "text/html":
                html_parts.append(_decode_part(part))
    else:
        text = _decode_part(msg)
        if msg.get_content_type() == "text/html":
            html_parts.append(text)
        else:
            plain_parts.append(text)

    if plain_parts:
        return "\n".join(p for p in plain_parts if p).strip()
    if html_parts:
        return _strip_html("\n".join(html_parts)).strip()
    return ""


def _decode_rfc2047(value: str | None) -> str:
    """Decode an RFC 2047 encoded-word header (e.g. =?UTF-8?B?...?=) to plain text."""
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:
        return value.strip()


def extract_record(msg: Message, label: str, source: str) -> dict:
    subject = _decode_rfc2047(msg.get("Subject"))
    sender = _decode_rfc2047(msg.get("From"))
    body = extract_body(msg)
    text = f"{subject}\n\n{body}".strip()
    return {
        "text": text,
        "label": label,
        "subject": subject,
        "sender": sender,
        "source": source,
    }


def _looks_like_mbox(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.readline(8).startswith(b"From ")
    except OSError:
        return False


def _parse_single_file(path: Path) -> Message | None:
    try:
        with path.open("rb") as f:
            return email.message_from_binary_file(f, policy=email.policy.compat32)
    except Exception as exc:
        print(f"  skipped {path}: {exc}", file=sys.stderr)
        return None


def iter_messages(path: Path) -> Iterator[Message]:
    """Yield every email Message found at `path` (directory, mbox, or single file)."""
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            # SpamAssassin ships an index named `cmds`; also skip dotfiles.
            if child.name.startswith(".") or child.name == "cmds":
                continue
            msg = _parse_single_file(child)
            if msg is not None:
                yield msg
    elif path.is_file():
        if _looks_like_mbox(path):
            box = mailbox.mbox(str(path))
            try:
                for msg in box:
                    yield msg
            finally:
                box.close()
        else:
            msg = _parse_single_file(path)
            if msg is not None:
                yield msg
    else:
        print(f"  path not found: {path}", file=sys.stderr)


def convert(
    phishing_paths: Iterable[Path],
    ham_paths: Iterable[Path],
    out: Path,
) -> dict[str, int]:
    counts = {"phishing": 0, "legitimate": 0, "skipped_empty": 0}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["text", "label", "subject", "sender", "source"]
        )
        writer.writeheader()
        for label, paths in (("phishing", phishing_paths), ("legitimate", ham_paths)):
            for path in paths:
                print(f"reading {path} as {label}")
                for msg in iter_messages(path):
                    rec = extract_record(msg, label=label, source=str(path))
                    if not rec["text"]:
                        counts["skipped_empty"] += 1
                        continue
                    writer.writerow(rec)
                    counts[label] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phishing",
        type=Path,
        nargs="+",
        required=True,
        help="phishing data: mbox file(s), directory(s), or .eml file(s)",
    )
    parser.add_argument(
        "--ham",
        type=Path,
        nargs="+",
        required=True,
        help="legitimate data: mbox file(s), directory(s), or .eml file(s)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/emails.csv")
    )
    args = parser.parse_args()
    counts = convert(args.phishing, args.ham, args.out)
    print(f"wrote {args.out}")
    print(f"  phishing:    {counts['phishing']}")
    print(f"  legitimate:  {counts['legitimate']}")
    print(f"  skipped (empty body): {counts['skipped_empty']}")


if __name__ == "__main__":
    main()
