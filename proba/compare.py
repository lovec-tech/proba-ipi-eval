#!/usr/bin/env python3
"""Put one detector beside the others, at one operating point, without re-running anything.

    python3 -m proba.compare                       # everyone at 0.1% and 1% FPR
    python3 -m proba.compare --against floor       # everyone at floor's OWN realised rate
    python3 -m proba.compare --fpr 0.005           # everyone at a budget nobody was run at

WHY IT EXISTS. The corpus ships every detector's per-document scores, which is what makes "compare
against the measured detectors without re-running them" true rather than a slogan — but only if
something can actually do the re-thresholding. A threshold is a number chosen over saved scores, so
moving every detector to a common budget is arithmetic, not inference.

THE BINARY CASE IS THE POINT. A detector that emits a verdict has one operating point and no
curve; ours is `floor`, which fires at 0.021% FPR. Reading it next to a column taken at 0.1% gives
the others five times its false-positive budget and calls the result a comparison. `--against
<binary detector>` re-cuts every scored detector at that detector's realised rate instead, which is
the only way the two are answering the same question.

Runs are matched on the corpus fingerprint: a result measured on another build is listed as skipped
with its reason rather than silently mixed in.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from .data import DEFAULT_ROOT, meta_docs, scored
from .metrics import evaluate, threshold_at_fpr
from .paths import RESULTS as OUT


def latest(results: pathlib.Path):
    """detector -> (result, scores path), newest run per detector."""
    out = {}
    for f in sorted(results.glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "mean_recall" not in r:
            continue
        sc = results / (f.stem + ".scores.jsonl")
        if not sc.exists():
            continue
        d = r.get("detector")
        if d not in out or r.get("run_at", "") >= out[d][0].get("run_at", ""):
            out[d] = (r, sc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(OUT))
    ap.add_argument("--data", default=str(DEFAULT_ROOT))
    ap.add_argument("--fpr", type=float, action="append",
                    help="operating point(s); repeatable. Default: 0.001 and 0.01")
    ap.add_argument("--against", help="re-cut everyone at THIS detector's realised rate "
                                      "(the only fair reading when it is binary)")
    ap.add_argument("--slice", default=None,
                    help="restrict the positives to a named slice before scoring the comparison")
    a = ap.parse_args()

    runs = latest(pathlib.Path(a.results))
    if not runs:
        raise SystemExit(f"no results with saved scores in {a.results}")

    meta = meta_docs(a.data)
    fps = {r.get("dataset") for r, _ in runs.values()}
    build = max(fps, key=lambda f: sum(1 for r, _ in runs.values() if r.get("dataset") == f))
    loaded, skipped = {}, []
    for det, (res, sc) in runs.items():
        if res.get("dataset") != build:
            skipped.append((det, f"measured on build {res.get('dataset')}, not {build}"))
            continue
        try:
            loaded[det] = (res, *scored(sc, a.data, meta))
        except KeyError as e:
            skipped.append((det, str(e)[:80]))

    targets = a.fpr or [0.001, 0.01]
    label = None
    if a.against:
        if a.against not in loaded:
            raise SystemExit(f"no run for {a.against} (have: {sorted(loaded)})")
        res, pos, neg = loaded[a.against]
        rate = res.get("fpr_pooled")
        if rate is None:
            raise SystemExit(f"{a.against} has no realised pooled FPR recorded")
        targets = [rate]
        label = (f"{a.against}'s own realised rate {rate*100:.4f}%"
                 + (" (binary: one point, no curve)" if res.get("binary") else ""))

    print(f"build {build} · {len(loaded)} detectors" + (f" · {label}" if label else ""))
    for det, why in skipped:
        print(f"  skipped {det}: {why}")

    head = "  ".join(f"@{t*100:g}%".rjust(9) for t in targets)
    print(f"\n{'detector':>14} {head}   {'point':>26}")
    print("─" * (16 + 11 * len(targets) + 28))

    rows = []
    for det, (res, pos, neg) in loaded.items():
        cells, first = [], None
        for t in targets:
            if res.get("binary"):
                # No threshold to move: it fires or it does not, whatever the budget says.
                r = evaluate(pos, neg, binary=True, with_curve=False)
                cells.append(f"{r['mean_recall']*100:8.1f}%")
                got = r["fpr_pooled"]
            else:
                tau = threshold_at_fpr([s for _, s in neg], t)
                hits = sum(1 for _, s in pos if s >= tau)
                got = sum(1 for _, s in neg if s >= tau) / len(neg)
                cells.append(f"{hits/len(pos)*100:8.1f}%")
            if first is None:
                first = (hits / len(pos) if not res.get("binary") else r["mean_recall"], got)
        note = ("its own point, FPR %.4f%%" % (first[1] * 100) if res.get("binary")
                else "threshold at target")
        rows.append((first[0], det, cells, note))

    for _, det, cells, note in sorted(rows, reverse=True):
        print(f"{det:>14} {'  '.join(cells)}   {note:>26}")

    if a.against and any(r.get("binary") for r, _, _ in loaded.values()):
        print("\nA binary detector keeps its own point in every column — it has no threshold to "
              "move.\nThe scored detectors are the ones re-cut, which is what makes the columns "
              "comparable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
