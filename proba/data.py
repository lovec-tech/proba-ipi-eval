#!/usr/bin/env python3
"""Loading the corpus, and the slices worth measuring separately.

Structure follows quadrat-ipi-eval (Apache-2.0). The differences are the released
layout — six files, an injected and a clean JSONL per carrier — and one field-shape
fix: the two sides name their carriers with different vocabularies
(`cards/reviews/web` on the injected rows, `product_card/ugc_review/web` on the
clean ones). They are the same three carriers, so `_doc` normalises both to one set;
without it the per-carrier false-positive rate and the per-carrier recall would sit
under different keys and never line up.

Slices are filters over the rows, each a caveat the reader can re-measure. A slice
that names a POSITIVE-only property (verified, obfuscated) keeps every clean row, so
the threshold is still chosen over the whole clean pool the recall is read against.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import typing as t

from .detector import Doc
from .paths import DATA_ROOT as DEFAULT_ROOT, INJECTED_FILES, CLEAN_FILES, corpus

#: both carrier vocabularies -> one set of three names
CARRIER_NORM = {
    "cards": "cards", "product_card": "cards",
    "reviews": "reviews", "ugc_review": "reviews",
    "web": "web",
}


def _kind(r: dict) -> str:
    """Obfuscation flattened to its kind, so it can be a marginal axis. Injected only."""
    if r.get("label") != "injected":
        return None
    return (r.get("obfuscation") or {}).get("kind") or "none"


#: name -> predicate on the raw row. Each is a caveat the reader can re-measure.
#: A positive-only predicate must pass every clean row through (guard on label),
#: or the clean pool the threshold is set on shrinks with the positive filter.
SLICES: dict[str, t.Callable[[dict], bool]] = {
    "all":        lambda r: True,
    # NOT the headline. The headline denominator is every injected row, 16,800, the
    # same as quadrat-ipi (`n_positives: 16800`). This slice drops the ~14% quota-fill
    # (`inj_verified: false`) — payloads a cell was topped up with but the blind judge
    # did not confirm — for a recall figure over confirmed injections only. Clean rows
    # are kept, so the threshold and FPR are unchanged.
    "verified":   lambda r: r.get("label") != "injected" or bool(r.get("inj_verified")),
    "obfuscated": lambda r: r.get("label") != "injected" or _kind(r) not in (None, "none"),
    "clean_typo": lambda r: r.get("label") != "injected" or not r.get("typography_folded"),
}


def _doc(r: dict) -> Doc:
    span = r.get("inj_span")
    injected = r.get("label") == "injected"
    return Doc(
        id=r["id"], text=r.get("text", ""), label=r["label"],
        host_type=CARRIER_NORM.get(r.get("host_type"), r.get("host_type")),
        host_source=r.get("host_source") or r.get("source") or "",
        family=r.get("family"), action=r.get("action"),
        spliced_at=r.get("spliced_at"),
        obfuscation=_kind(r),
        inj_span=tuple(span) if span else None,
        meta={k: r[k] for k in ("inj_verified", "gen_model", "typography_folded",
                                "obfuscation", "locality", "license")
              if k in r},
    )


def fingerprint(root=DEFAULT_ROOT) -> str:
    """Content hash of what a detector actually reads: id, label, text. Nothing else.

    Recorded with every result and checked when scores are reused, so a rebuilt corpus
    that reused an id for different text cannot be scored from another build's numbers.
    Hashing only the three read fields means dropping an unused metadata column does not
    invalidate a finished measurement."""
    root = corpus(root)
    h = hashlib.sha256()
    for name in INJECTED_FILES + CLEAN_FILES:
        h.update(name.encode())
        with (pathlib.Path(root) / name).open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                h.update(f"{r['id']}\0{r['label']}\0{r.get('text','')}\0".encode())
    return h.hexdigest()[:16]


def _iter(root, files, keep, limit):
    root = pathlib.Path(root)
    rows = []
    for name in files:
        with (root / name).open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if keep(r):
                    rows.append(_doc(r))
                    if limit and len(rows) >= limit:
                        return rows
    return rows


def load(root=DEFAULT_ROOT, slice_name="all", limit=None):
    """Return (positives, negatives) as lists of Doc, filtered by a named slice.

    The slice applies to both sides through the same predicate; the positive-only
    predicates keep every clean row by construction (see SLICES)."""
    root = corpus(root)
    keep = SLICES[slice_name]
    pos = _iter(root, INJECTED_FILES, keep, limit)
    neg = _iter(root, CLEAN_FILES, keep, limit)
    return pos, neg


def meta_docs(root=DEFAULT_ROOT) -> dict[str, Doc]:
    """id -> Doc carrying every axis but NO text — for re-deriving metrics from a saved
    scores file without reloading hundreds of MB of carrier text."""
    root = corpus(root)
    out = {}
    for name in INJECTED_FILES + CLEAN_FILES:
        with (pathlib.Path(root) / name).open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                r["text"] = ""
                out[r["id"]] = _doc(r)
    return out


def scored(scores_path, root=DEFAULT_ROOT, meta=None):
    """Re-read a saved `<run>.scores.jsonl` as (positives, negatives) of (doc, score).

    An id absent from the corpus is raised, not skipped: silently dropping foreign ids
    would compute a clean-looking result from a partly foreign run."""
    meta = meta if meta is not None else meta_docs(root)
    pos, neg = [], []
    with pathlib.Path(scores_path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            d = meta.get(r["id"])
            if d is None:
                raise KeyError(
                    f"{r['id']} is not in the current build — these scores belong to another corpus")
            (pos if d.label == "injected" else neg).append((d, r["score"]))
    return pos, neg
