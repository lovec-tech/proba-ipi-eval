#!/usr/bin/env python3
"""Import already-measured runs into harness-native results, without touching a model.

    python3 -m proba.ingest --data <release>/data --out eval-out/results
    python3 -m proba.ingest --only proventra,gbv --dry-run

WHY THIS EXISTS. Seven detectors were measured on this corpus before the harness was written,
and their per-document scores were kept. A score is the measurement; a threshold, an operating
point and a per-cell profile are arithmetic over those scores (`metrics.at_points`: "nothing here
touches a model"). So the runs do not need re-running to become harness results — they need
their scores read into the canonical shape and the metrics recomputed from them.

WHAT IT FIXES. Our scores were written per carrier, in three directories with three different
layouts, by two different runners. That is fine as a lab notebook and useless as a released
artifact: a third party cannot compare against a detector whose numbers live in a shape only our
analysis script understands. Quadrat-IPI ships `results/<slug>.json` + `<slug>.scores.jsonl` per
detector for exactly this reason — so its table can be re-derived, re-thresholded and re-sliced by
someone who never runs the models. This writes ours in that same shape.

WHAT IS CARRIED AND WHAT IS RECOMPUTED. Everything that DESCRIBES a measurement — model id,
revision, device, aperture, segmenter, aggregation — is carried verbatim from the original run's
`run.json`, because those are facts about a pass that already happened. Everything MEASURED —
thresholds, recall, FPR, cells, marginals, intervals — is recomputed here from the full score set.
`seconds` and `n_windows` are NOT carried when the source run did not record them: an imported run
must not report a timing that was never measured.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from .data import DEFAULT_ROOT, fingerprint, load
from .metrics import at_points, evaluate
from .paths import RESULTS as OUT

#: Where the pre-harness runs put their scores. Two runners, three layouts — the reason this
#: module exists. `inj` is a per-carrier template, `clean` a per-pool template.
REPO = pathlib.Path(__file__).resolve().parent.parent.parent
RUNS = REPO / "eval" / "runs"
CARRIERS = ("cards", "reviews", "web")
POOLS = {"cards": "cards_clean", "reviews": "reviews_clean", "web": "web_clean"}

#: Detector -> how its scores are laid out and what to call it in the table.
SOURCES = {
    "promptidote": {
        "display": "promptidote",
        "version": "intent-1e-04",
        "inj": lambda g: RUNS / f"ours-q92-{g}" / "results.jsonl",
        "meta": lambda: RUNS / "ours-q92-cards" / "run.json",
        "window": 5000,
    },
}
for _d in ("bastion", "deepset", "gbv", "piguard", "protectai", "proventra"):
    SOURCES[_d] = {
        "display": _d,
        "version": None,                       # taken from the run's own model revision
        "inj": (lambda d: lambda g: RUNS / "q92" / d / f"q92_{g}.scores.jsonl")(_d),
        "meta": (lambda d: lambda: RUNS / "q92" / d / "run.json")(_d),
        "window": 2000,
    }

#: Clean-pool scores moved between run directories over the project's life; the first that exists
#: wins, and which one was used is recorded in the result's provenance block.
CLEAN_CANDIDATES = (
    lambda d, p: RUNS / "v3-bank-final" / d / f"rel_{p}.scores.jsonl",
    lambda d, p: RUNS / "oss-release-0.1.1" / d / f"rel_{p}.scores.jsonl",
    lambda d, p: RUNS / "oss" / f"{d}__{p}.jsonl",
)


def _rows(path: pathlib.Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def max_per_id(path: pathlib.Path) -> dict[str, float]:
    """id -> max score over its windows.

    MAX, NOT LAST. A document longer than the detector's aperture is scored in several windows and
    written as several lines under one id; the verdict is `any(window)`, i.e. the maximum. Reading
    the file straight into a dict would keep whichever window happened to be written last, which is
    silently wrong exactly on the long documents where the aperture matters. Rows whose score is
    null (a failed API call the runner retried) are dropped rather than compared as None."""
    best: dict[str, float] = {}
    for r in path_rows(path):
        s = r.get("score")
        if s is None:
            continue
        i = r["id"]
        if i not in best or s > best[i]:
            best[i] = s
    return best


def path_rows(path: pathlib.Path):
    return _rows(path)


def clean_scores(det: str) -> tuple[dict[str, float], list[str]]:
    out, used = {}, []
    for pool in POOLS.values():
        for cand in CLEAN_CANDIDATES:
            p = cand(det, pool)
            if p.exists():
                out.update(max_per_id(p))
                used.append(str(p.relative_to(REPO)))
                break
    return out, used


def inj_scores(det: str, spec: dict) -> tuple[dict[str, float], list[str]]:
    out, used = {}, []
    for g in CARRIERS:
        p = spec["inj"](g)
        if p.exists():
            out.update(max_per_id(p))
            used.append(str(p.relative_to(REPO)))
    return out, used


def source_meta(spec: dict) -> dict:
    """Facts about the original pass, or {} when it kept none."""
    p = spec["meta"]()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def describe(det: str, spec: dict, meta: dict) -> dict:
    """The DESCRIPTIVE half of the record — carried, never recomputed."""
    # The two runners disagree on this field's SHAPE: the OSS runner writes `detector` as a block
    # (model_id, revision, positive_label), the API runner writes it as a bare name string. Both
    # are legitimate records of what ran, so normalise rather than demand one of them.
    d = meta.get("detector")
    d = d if isinstance(d, dict) else ({"model_id": d} if d else {})
    inf = meta.get("inference") or {}
    version = spec["version"] or (str(d.get("revision", ""))[:8] or "unknown")
    notes = []
    if d.get("model_id"):
        notes.append(f"{d['model_id']}@{str(d.get('revision',''))[:8]}")
    if inf.get("device"):
        notes.append(str(inf["device"]))
    if d.get("positive_label"):
        notes.append(f"positive={d['positive_label']}")
    if inf.get("sentence_segmenter"):
        notes.append(str(inf["sentence_segmenter"]))
    if meta.get("api_max_chars"):
        notes.append(f"API window {meta['api_max_chars']} chars, {meta.get('primary_metric','')}")
    return {
        "detector": det,
        "display": spec["display"],
        "version": version,
        "slice": "all",
        "policy": inf.get("window_policy", "chunk"),
        "window": inf.get("window_chars", spec["window"]),
        "overlap": inf.get("overlap_sentences", 4),
        "forced_aperture": False,
        "limit": None,
        "pseudonymised": False,
        # Same field `run.py` records, carried from the original pass so imported and fresh runs
        # are describable in one vocabulary. Our runs recorded it as
        # "quadrat regex fallback; blingfire disabled" — the Latin-only fallback.
        "segmenter": inf.get("sentence_segmenter"),
        "notes": " · ".join(n for n in notes if n),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", help="comma-separated detector names")
    ap.add_argument("--fpr", type=float, default=0.001, help="headline operating point")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    want = {s.strip() for s in a.only.split(",")} if a.only else set(SOURCES)
    out_dir = pathlib.Path(a.out)
    fp = fingerprint(a.data)
    pos, neg = load(a.data)
    print(f"build {fp} · {len(pos)} injected, {len(neg)} clean", flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    imported, skipped = [], []
    for det in sorted(want):
        spec = SOURCES.get(det)
        if spec is None:
            skipped.append((det, "no source layout registered"))
            continue
        isc, isrc = inj_scores(det, spec)
        csc, csrc = clean_scores(det)
        if not isc or not csc:
            skipped.append((det, f"scores missing (injected {len(isc)}, clean {len(csc)})"))
            continue

        # COMPLETENESS IS A GATE, not a warning. A detector missing scores for part of the corpus
        # still produces a plausible-looking recall over whatever it does cover, and nothing about
        # that number reads as wrong afterwards.
        miss_p = [d.id for d in pos if d.id not in isc]
        miss_n = [d.id for d in neg if d.id not in csc]
        if miss_p or miss_n:
            skipped.append((det, f"incomplete: {len(miss_p)} injected and {len(miss_n)} clean "
                                 f"documents have no score (e.g. {(miss_p or miss_n)[:2]})"))
            continue

        paired_pos = [(d, isc[d.id]) for d in pos]
        paired_neg = [(d, csc[d.id]) for d in neg]
        if a.dry_run:
            imported.append((det, None))
            print(f"  {det:12s} would import {len(paired_pos)}+{len(paired_neg)} scores")
            continue

        meta = source_meta(spec)
        res = evaluate(paired_pos, paired_neg, target_fpr=a.fpr)
        res["points"] = at_points(paired_pos, paired_neg)
        res.update(**describe(det, spec, meta))
        res.update(
            binary=False, dataset=fp, run_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            imported=True,
            provenance={
                "note": "scores measured before this harness existed; metrics recomputed here "
                        "from those scores, no model was run",
                "injected_scores": isrc,
                "clean_scores": csrc,
                "source_run": str(spec["meta"]().relative_to(REPO)) if spec["meta"]().exists() else None,
                "software": meta.get("software"),
                "inference": meta.get("inference"),
            },
        )
        # seconds / n_windows deliberately absent: the source runs did not record a comparable
        # figure, and an imported run must not report a timing that was never measured.

        tag = f"{res['policy']}{res['window']}o{res['overlap']}"
        slug = f"{det}-{res['version']}-{stamp}-{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{slug}.json"
        if target.exists():
            raise SystemExit(f"{target.name} already exists — refusing to overwrite a run")
        target.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        with (out_dir / f"{slug}.scores.jsonl").open("w", encoding="utf-8") as fh:
            for d, s in paired_pos + paired_neg:
                fh.write(json.dumps({"id": d.id, "score": s}) + "\n")
        imported.append((det, res))
        p01 = res["points"]["0.001"]["mean_recall"] * 100
        p1 = res["points"]["0.01"]["mean_recall"] * 100
        print(f"  {det:12s} recall @0.1% {p01:5.1f}%  @1% {p1:5.1f}%  "
              f"FPR {res['fpr_pooled']*100:.3f}%  -> {slug}.json", flush=True)

    for det, why in skipped:
        print(f"  skipped {det}: {why}")
    print(f"\nimported {len(imported)}, skipped {len(skipped)} -> {out_dir}")
    return 0 if imported else 1


if __name__ == "__main__":
    raise SystemExit(main())
