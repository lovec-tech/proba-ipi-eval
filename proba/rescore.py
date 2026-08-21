#!/usr/bin/env python3
"""Re-score only the documents that changed, splice them into a finished run, recompute.

    python3 -m proba.rescore --ids-from stale.json --dry-run
    python3 -m proba.rescore --ids ruq-cards-03987,ruq-web-00177 --only proventra

WHY THIS IS EXACT, and not an approximation of a re-run. These detectors are deterministic and
read one document at a time: windows are cut inside a document and never cross into the next, so a
document whose bytes did not change produces the same score it produced before, to the last bit.
Scoring only what changed and keeping the rest is therefore the SAME measurement as a full pass —
and a 32-row edit costs seconds instead of a day of somebody's card.

WHAT IS NOT INHERITED: the thresholds. They are re-derived from the whole clean pool, because a
changed negative can move the corpus-wide cut, and a moved cut changes verdicts on documents that
were never touched. Every metric is recomputed from the full spliced score set for that reason.

WHY IT IS NEEDED HERE. The seven detectors were measured on the corpus as composed, and the
released corpus was then edited: 32 documents had phone numbers masked digit-for-digit before
publication. The published metrics do not move (the largest effect measured was 0.006 points), but
the shipped `results/*.scores.jsonl` would otherwise hold, for those 32 rows, a score taken on text
that is not the text in `data/`. That is a provenance defect rather than a numbers defect, and it
is the kind that quietly makes a corpus non-reproducible: someone re-running the harness on the
published files gets different numbers for those rows and has nothing telling them why.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from .data import DEFAULT_ROOT, fingerprint, load
from .detector import REGISTRY, load_detectors
from .metrics import at_points, evaluate
from .paths import RESULTS as OUT
from .window import WINDOW_512


def spliceable(res, scores_path):
    """Why this run cannot be carried forward, or None if it can."""
    if not scores_path.exists():
        return "scores were not saved"
    # An unregistered adapter cannot be re-scored — there is nothing to score with. Skipped with a
    # reason rather than raised, so one absent adapter does not stop the others.
    if res.get("detector") not in REGISTRY:
        return "no adapter in the registry"
    if res.get("limit"):
        return "smoke run, not a measurement"
    return None


def build_detector(res):
    """The detector instance the parent run measured with, aperture and all.

    Restored by mutating the instance exactly as `run.py` does, so `det.aperture()` answers the
    same way it answered then. An aperture that is not restored would silently re-measure the
    detector's own opening and file the result under the parent's settings."""
    cls, version = REGISTRY[res["detector"]]
    det = cls()
    det.settings = None
    if res.get("forced_aperture"):
        det.max_chars = res.get("window") or det.max_chars or WINDOW_512
        det.policy = res.get("policy") or "chunk"
        if det.policy == "full":
            det.max_chars = None
    if res.get("overlap") is not None:
        det.overlap = res["overlap"]
    return det, version


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ids", help="comma-separated document ids")
    ap.add_argument("--ids-from", help="JSON: a list of ids, or {'rows': [{'id': ...}]}")
    ap.add_argument("--results", default=str(OUT))
    ap.add_argument("--data", default=str(DEFAULT_ROOT))
    ap.add_argument("--only", help="comma-separated detector names")
    ap.add_argument("--adapters", action="append", default=[])
    ap.add_argument("--in-place", action="store_true",
                    help="replace the parent run's files instead of writing a new pair; use when "
                         "the parent is a published artifact that should not gain a sibling")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ids = set()
    if a.ids:
        ids |= {s.strip() for s in a.ids.split(",") if s.strip()}
    if a.ids_from:
        raw = json.loads(pathlib.Path(a.ids_from).read_text(encoding="utf-8"))
        ids |= set(raw if isinstance(raw, list) else [r["id"] for r in raw["rows"]])
    if not ids:
        raise SystemExit("nothing to re-score: give --ids or --ids-from")

    load_detectors(a.adapters)
    results = pathlib.Path(a.results)
    fp = fingerprint(a.data)
    print(f"build {fp} · documents to re-score {len(ids)}", flush=True)

    pos, neg = load(a.data)
    by_id = {d.id: d for d in pos + neg}
    missing = ids - set(by_id)
    if missing:
        raise SystemExit(f"not in this build: {sorted(missing)[:5]}")
    targets = [by_id[i] for i in sorted(ids)]

    only = {s.strip() for s in a.only.split(",")} if a.only else None
    runs, skipped = [], []
    for f in sorted(results.glob("*.json")):
        try:
            res = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "mean_recall" not in res:
            continue
        sc = results / (f.stem + ".scores.jsonl")
        why = spliceable(res, sc)
        if why or (only and res.get("detector") not in only):
            skipped.append((f.name, why or "not in --only"))
            continue
        runs.append((f, res, sc))

    print(f"runs to recompute: {len(runs)}", flush=True)
    for n, why in skipped:
        print(f"  skipped {n}: {why}")
    if a.dry_run:
        for f, res, _ in runs:
            print(f"  {res['detector']:14s} {res.get('policy')}/{res.get('window')} <- {f.name}")
        return 0

    for i, (f, res, sc) in enumerate(runs, 1):
        name = res["detector"]
        print(f"\n[{i}/{len(runs)}] {name} · {res.get('policy')}/{res.get('window')}", flush=True)
        old = {}
        with sc.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                old[r["id"]] = r["score"]
        absent = set(by_id) - set(old)
        if absent:
            print(f"  ✗ skipped: the scores are missing {len(absent)} documents of this build "
                  f"(e.g. {sorted(absent)[:2]}) — this run is of another corpus")
            continue

        det, version = build_detector(res)
        try:
            det.setup()
        except SystemExit as e:            # e.g. a hosted detector with no key configured
            print(f"  ✗ skipped: {e}")
            continue
        t0 = time.time()
        fresh, n_win = det.score_documents(targets)
        det.teardown()
        moved = [(d.id, old[d.id], s) for d, s in zip(targets, fresh) if old[d.id] != s]
        for d, s in zip(targets, fresh):
            old[d.id] = s
        print(f"  re-scored {len(targets)} docs in {time.time() - t0:.1f} s · {n_win} windows · "
              f"score moved on {len(moved)}", flush=True)
        for did, o, s in moved[:5]:
            print(f"    {did:18s} {o:.6f} -> {s:.6f}  ({s - o:+.6f})")

        paired_pos = [(d, old[d.id]) for d in pos]
        paired_neg = [(d, old[d.id]) for d in neg]
        binary = bool(res.get("binary"))
        out = evaluate(paired_pos, paired_neg,
                       target_fpr=res.get("target_fpr") or 0.001, binary=binary)
        out["points"] = {} if binary else at_points(paired_pos, paired_neg)
        # Everything that DESCRIBES the measurement is carried; everything MEASURED is recomputed.
        for k in ("detector", "version", "slice", "policy", "window", "overlap", "segmenter",
                  "forced_aperture", "limit", "pseudonymised", "notes", "display",
                  "seconds", "n_windows", "provenance", "imported"):
            if k in res:
                out[k] = res[k]
        out.update(binary=binary, dataset=fp, run_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                   derived_from=f.name, rescored_ids=sorted(ids))

        if a.in_place:
            target_json, target_scores = f, sc
        else:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            tag = f"{res.get('policy')}{res.get('window') or ''}"
            if res.get("policy") == "chunk" and res.get("overlap"):
                tag += f"o{res['overlap']}"
            slug = f"{name}-{version}-{stamp}-{tag}"
            target_json = results / f"{slug}.json"
            target_scores = results / f"{slug}.scores.jsonl"
            if target_json.exists():
                raise SystemExit(f"{target_json.name} already exists — refusing to overwrite a run")
        target_json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        with target_scores.open("w", encoding="utf-8") as fh:
            for d in pos + neg:
                fh.write(json.dumps({"id": d.id, "score": old[d.id]}) + "\n")

        d_rec = (out["mean_recall"] - res["mean_recall"]) * 100
        d_fpr = (out["fpr_pooled"] - res["fpr_pooled"]) * 100
        print(f"  recall {res['mean_recall']*100:.3f}% -> {out['mean_recall']*100:.3f}% "
              f"({d_rec:+.3f}) · FPR {res['fpr_pooled']*100:.4f}% -> "
              f"{out['fpr_pooled']*100:.4f}% ({d_fpr:+.4f})")
        print(f"  -> {target_json.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
