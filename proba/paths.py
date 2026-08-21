#!/usr/bin/env python3
"""Where the harness reads the corpus and where it writes what it produces.

Structure follows quadrat-ipi-eval (Apache-2.0), retargeted at proba-ipi, whose
released data ships as SIX files, not two: an injected and a clean JSONL per
carrier. `positives`/`negatives` are the union of the three of each.

    PROBA_DATA      the corpus directory holding the six JSONL files.  Default: ./data
    PROBA_OUT       everything a run produces.                          Default: ./eval-out
    PROBA_REPO      Hugging Face dataset to fetch when the corpus is not on disk.
    PROBA_VERSION   which released tag to take — PINNED, never `main`.

`--data` / `--results` override per command. Paths resolve against the working
directory, never against this file.
"""
from __future__ import annotations

import os
import pathlib

#: The corpus. `fingerprint()` hashes these files, and every result records the hash.
DATA_ROOT = pathlib.Path(os.environ.get("PROBA_DATA", "data"))

#: Where the corpus comes from when not on disk. PINNED TO A TAG on purpose: a
#: harness that followed `main` would change what a saved score means between two
#: runs of the same code. Canonical dataset is privettoha/proba-ipi (gran-ipi is a
#: duplicate; see HARNESS_PLAN.md).
DATASET_REPO = os.environ.get("PROBA_REPO", "privettoha/proba-ipi")
#: BUMPED WITH EACH DATASET RELEASE, never `main`. Set PROBA_VERSION=main to follow
#: the newest state; the fingerprint in every result still says which bytes you read.
#: NOTE: the tag on the 92-cell content must be cut before this default resolves
#: (the new proba-ipi content on `main` is untagged as of the harness build).
DATASET_VERSION = os.environ.get("PROBA_VERSION", "v2.0.0")

#: Released layout: injected + clean per carrier.
CARRIERS = ("cards", "reviews", "web")
INJECTED_FILES = tuple(f"injected_{c}.jsonl" for c in CARRIERS)
CLEAN_FILES = tuple(f"{c}_clean.jsonl" for c in CARRIERS)
CORPUS_FILES = INJECTED_FILES + CLEAN_FILES


def corpus(root=None) -> pathlib.Path:
    """The directory holding the corpus, fetching it on first use if absent.

    AN EXPLICIT PATH IS NEVER SECOND-GUESSED. If PROBA_DATA or --data names a
    directory and the files are not in it, that is an error worth seeing rather
    than a silent download into a different place. The fetch happens only for the
    default location."""
    root = pathlib.Path(root) if root is not None else DATA_ROOT
    if all((root / name).is_file() for name in CORPUS_FILES):
        return root
    if os.environ.get("PROBA_DATA") or root != DATA_ROOT:
        missing = [n for n in CORPUS_FILES if not (root / n).is_file()]
        raise SystemExit(
            f"{root}: corpus incomplete, missing {missing}. "
            f"Point PROBA_DATA at a released version's data/ directory, or unset it "
            f"to fetch {DATASET_REPO} {DATASET_VERSION} automatically.")
    return fetch()


def fetch() -> pathlib.Path:
    """Download the pinned version's data/ from the Hub into its cache."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "the corpus is not on disk and huggingface_hub is not installed.\n"
            f"  Either `pip install huggingface_hub` to fetch {DATASET_REPO} "
            f"{DATASET_VERSION},\n"
            "  or download it yourself and point PROBA_DATA at its data/ directory.")
    print(f"corpus not found locally — fetching {DATASET_REPO} {DATASET_VERSION}", flush=True)
    try:
        path = snapshot_download(repo_id=DATASET_REPO, repo_type="dataset",
                                 revision=DATASET_VERSION, allow_patterns=["data/*.jsonl"])
    except Exception as e:
        raise SystemExit(
            f"could not fetch {DATASET_REPO} at {DATASET_VERSION}: {e}\n"
            f"  If that revision does not exist yet, the tag has not been cut — set "
            f"PROBA_VERSION to a tag listed at "
            f"https://huggingface.co/datasets/{DATASET_REPO}/tags, or PROBA_VERSION=main.")
    out = pathlib.Path(path) / "data"
    if not all((out / name).is_file() for name in CORPUS_FILES):
        raise SystemExit(f"{DATASET_REPO} {DATASET_VERSION}: data/ is missing corpus files")
    print(f"corpus ready: {out}", flush=True)
    return out


#: Everything a run produces, under one root so a measurement campaign moves as a unit.
OUT_ROOT = pathlib.Path(os.environ.get("PROBA_OUT", "eval-out"))
RESULTS = OUT_ROOT / "results"
REPORTS = OUT_ROOT / "reports"
