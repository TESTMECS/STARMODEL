#!/usr/bin/env python3
"""
extract_star_sentences.py
~~~~~~~~~~~~~~~~~~~~~~~~
Fetch the transcript of a YouTube video that contains vocalised STAR markers
("Situation", "Task", "Action", "Result") and export a CSV with columns
    sentence,label
ready for fine‑tuning your STAR classifier.

Usage
-----
$ python extract_star_sentences.py --video_id DX92jIynHjw --out_csv star_labels.csv

Dependencies
------------
- youtube_transcript_api
- pandas
- nltk  (for punkt sentence splitter)
  > pip install youtube-transcript-api pandas nltk
  > python -m nltk.downloader punkt
"""

from __future__ import annotations
import argparse, re, sys
from typing import Iterator, Tuple, List

import pandas as pd
from youtube_transcript_api._api import YouTubeTranscriptApi
from nltk.tokenize import sent_tokenize

# ---------------------------------------------------------------------------
# Regex for robust STAR marker detection
# ---------------------------------------------------------------------------
STAR_PATTERN = re.compile(
    r"""
    \b                           # word boundary
    (Situation|Task|Action|Result) # one of the four labels
    [\s\-\–\—\u2014]*           # optional spaces / dash / em‑dash
    [:\.]?                        # optional colon or period
    \s*                           # trailing whitespace
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def fetch_transcript(video_id: str) -> str:
    """Return full transcript text for *video_id* or raise a helpful error."""
    try:
        t = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        sys.exit(f"ERROR: Failed to fetch transcript: {e}")
    return " ".join(item["text"] for item in t)


def iterate_labeled_chunks(text: str) -> Iterator[Tuple[str, str]]:
    """Yield (label, chunk_text) pairs based on STAR markers in *text*."""
    matches = list(STAR_PATTERN.finditer(text))
    if not matches:
        raise ValueError(
            "No STAR markers (Situation/Task/Action/Result) found in transcript.\n"
            "Inspect the raw transcript or adjust the regex."
        )

    for idx, m in enumerate(matches):
        label = m.group(1).title()  # canonical case
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            yield label, chunk


def explode_sentences(label: str, chunk: str) -> List[Tuple[str, str]]:
    """Sentence‑tokenise *chunk* and return list of (sentence, label)."""
    sents = [s.strip() for s in sent_tokenize(chunk) if s.strip()]
    return [(s, label) for s in sents]


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract STAR‑labeled sentences from a YouTube transcript."
    )
    p.add_argument(
        "--video_id",
        default="DX92jIynHjw",
        help="YouTube video ID (the part after v=)",
    )
    p.add_argument(
        "--out_csv",
        default="star_labels.csv",
        help="Output CSV path [default: %(default)s]",
    )
    p.add_argument(
        "--debug", action="store_true", help="Print found chunks for manual inspection"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("[INFO] Fetching transcript …")
    raw = fetch_transcript(args.video_id)

    print("[INFO] Parsing STAR chunks …")
    rows: List[Tuple[str, str]] = []
    for label, chunk in iterate_labeled_chunks(raw):
        if args.debug:
            print(f"\n[DEBUG] {label}: {chunk[:120]}…")
        rows.extend(explode_sentences(label, chunk))

    if not rows:
        sys.exit(
            "ERROR: No sentences were extracted. Enable --debug to inspect parsing."
        )

    df = pd.DataFrame(rows, columns=["sentence", "label"])
    df.to_csv(args.out_csv, index=False)
    print(f"[SUCCESS] Saved {len(df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
