#!/usr/bin/env python3
"""Run a registered detector over the corpus and print the metrics.

    python3 -m proba.run --detector floor
    python3 -m proba.run --detector proventra --slice verified
    python3 -m proba.run --list

Raw scores are written next to the result, so a re-measure never needs another forward pass: the
threshold, the slice and every metric are recomputed from the saved scores in seconds
(`proba.table`, or `data.scored()` + `metrics.at_points`).

Guards and the resume mechanism follow quadrat-ipi-eval; they are what keeps a run from filing a
confident wrong number. See NOTICE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time

from .data import DEFAULT_ROOT, SLICES, fingerprint, load
from .detector import REGISTRY, load_detectors
from .metrics import at_points, evaluate, score_floor
from .paths import RESULTS as OUT
from .window import OVERLAP_SENTENCES, POLICIES, WINDOW_512


def partial_path(settings):
    """Where a half-finished pass keeps its scores.

    Keyed by the SETTINGS, not by the run's timestamp: a restart is a new run with a new name but
    the same measurement, and it has to find what the previous attempt already paid for."""
    key = json.dumps(settings, sort_keys=True, ensure_ascii=False)
    return OUT / f".partial-{hashlib.sha1(key.encode()).hexdigest()[:12]}.jsonl"


def load_partial(path):
    """({id: score}, windows) from a previous attempt. A truncated last line is dropped."""
    got, wins = {}, 0
    if not path.exists():
        return got, wins
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_windows" in r:
                wins += r["_windows"]
            else:
                got[r["id"]] = r["score"]
    return got, wins


#: documents per checkpoint — small enough that a crash loses seconds, large enough that the
#: per-batch overhead stays invisible.
BATCH = int(os.environ.get("PROBA_BATCH", "200"))


def score_with_checkpoint(det, docs, path, done, n_win=0):
    """Score what is not already in `done`, writing each batch as it lands.

    A pass over ~98,000 documents is hours of a card or dollars of somebody's API, and a crash at
    document 97,000 must not throw all of it away."""
    todo = [d for d in docs if d.id not in done]
    if done:
        print(f"  resuming: {len(done)} already scored, {len(todo)} to go", flush=True)
    if todo:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for i in range(0, len(todo), BATCH):
                batch = todo[i:i + BATCH]
                det.progress_offset = len(done)
                det.progress_total = len(done) + len(todo) - i
                scores, w = det.score_documents(batch)
                n_win += w
                for d, sc in zip(batch, scores):
                    done[d.id] = sc
                    fh.write(json.dumps({"id": d.id, "score": sc}) + "\n")
                fh.write(json.dumps({"_windows": w}) + "\n")
                fh.flush()
    return [done[d.id] for d in docs], n_win


def check_score_floor(neg, target, name, binary):
    """Refuse to publish an operating point the scores cannot express.

    A tie at the top of the clean distribution puts a floor under the reachable FPR: every
    threshold at or below it flags all of them, so the run reports a rate it never achieved — and
    nothing about the number looks wrong afterwards."""
    if binary:
        return
    tie, n, floor = score_floor(neg)
    if tie > 1 and floor > target:
        raise SystemExit(
            f"[{name}] target FPR {target*100:g}% is unreachable: {tie} of {n} clean documents "
            f"share the SAME maximal score ({floor*100:.4f}%).\n"
            f"  No threshold separates them, so any point below {floor*100:.4f}% is not a "
            f"measurement.\n"
            f"  This is score saturation: an fp32 softmax hits its ceiling and a max over many\n"
            f"  windows lands on it. Keep the logit margin instead of the probability, or use a\n"
            f"  coarser aperture; see metrics.score_floor.")


def check_score_type(det, scores, name):
    """Refuse to publish a run whose scores contradict how the detector declared itself.

    A scored detector that actually emits verdicts is the dangerous direction: threshold selection
    sees fewer firings than the budget allows, picks tau = 0, and `score >= tau` then flags every
    document — reported as 100% recall at 100% FPR, which reads like a triumph."""
    uniq = set(scores)
    verdicts = uniq <= {0.0, 1.0}
    if verdicts and not det.binary:
        raise SystemExit(
            f"[{name}] declares a score but returned only {sorted(uniq)} — a threshold at the "
            f"target\n  FPR degenerates on verdicts (tau=0 flags the whole corpus: 100% recall at "
            f"100% FPR).\n  If the detector really is binary, set `binary = True` on the adapter.\n"
            f"  If not, the adapter is binarising the score — return the continuous value.")
    if not verdicts and det.binary:
        print(f"  ⚠ [{name}] declares binary but returned {len(uniq)} distinct values — "
              f"measuring at its own point 0.5", flush=True)


def segmenter() -> str:
    """Which sentence splitter is actually active — recorded in every result.

    `window.sentence_spans` prefers blingfire and falls back to a regex when it cannot be
    imported. On Russian text those two are NOT interchangeable: the fallback requires a capital
    letter after the stop and its character class is Latin-only, so it finds almost no boundaries
    and degenerates into fixed-size slices. Which one ran is therefore part of how a number was
    obtained, and a result that does not say which is not reproducible. It goes into `settings`
    rather than beside them, so two runs that split differently cannot satisfy each other's
    `--skip-if-done` either."""
    try:
        import blingfire  # noqa: F401
        blingfire.text_to_sentences_and_offsets("Проверка. Ещё одна.")
        return "blingfire"
    except Exception:
        return "regex-fallback (Latin-only; near-no boundaries on Cyrillic)"


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--detector")
    ap.add_argument("--slice", default="all", choices=sorted(SLICES))
    ap.add_argument("--fpr", type=float, default=0.001,
                    help="target FPR for the single corpus-wide threshold")
    ap.add_argument("--data", default=str(DEFAULT_ROOT))
    ap.add_argument("--results", default=str(OUT))
    ap.add_argument("--limit", type=int, help="smoke test: first N of each class")
    ap.add_argument("--window", type=int, nargs="?", const=WINDOW_512,
                    help=f"OVERRIDE the detector's aperture, in CHARACTERS "
                         f"(bare flag = {WINDOW_512}). Omit to let the detector decide.")
    ap.add_argument("--policy", default=None, choices=POLICIES,
                    help="override how a document longer than the window is handled")
    ap.add_argument("--overlap", type=int, default=None,
                    help=f"override chunk overlap in SENTENCES (default {OVERLAP_SENTENCES})")
    ap.add_argument("--adapters", action="append", default=[],
                    help="directory of detector adapters to load (repeatable)")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="exit early if this exact pass was already measured on this dataset")
    ap.add_argument("--allow-offmachine", action="store_true",
                    help="required to run a detector that sends document text to a third party")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    load_detectors(args.adapters)
    if args.list or not args.detector:
        print("registered:")
        for n, (cls, v) in sorted(REGISTRY.items()):
            print(f"  {n:14s} {v:22s} {cls.notes}")
        return 0
    if args.detector not in REGISTRY:
        raise SystemExit(f"no such detector: {args.detector} (see --list)")

    out_dir = pathlib.Path(args.results)
    cls, version = REGISTRY[args.detector]
    det = cls()
    det.settings = None

    # The aperture is the DETECTOR's unless a flag overrides it. Applied by mutating the instance,
    # so `det.aperture()` stays the single answer to "how was this measured".
    forced = bool(args.window or args.policy)
    if forced:
        det.max_chars = args.window or det.max_chars or WINDOW_512
        det.policy = args.policy or "chunk"
        if args.policy in (None, "chunk") and args.overlap is not None:
            det.overlap = args.overlap
        if args.policy == "full":
            det.max_chars = None
    elif args.overlap is not None:
        det.overlap = args.overlap
    policy, window, overlap = det.aperture()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{policy}{window or ''}" + (f"o{overlap}" if policy == "chunk" and overlap else "")
    slug = f"{args.detector}-{version}-{stamp}-{tag}" + ("-f" if forced else "")
    settings = {"detector": args.detector, "version": version, "slice": args.slice,
                "policy": policy, "window": window, "overlap": overlap,
                "forced_aperture": forced, "limit": args.limit,
                # part of the aperture, not a footnote — see segmenter()
                "segmenter": segmenter() if policy != "full" else None,
                # recorded, so a row taken by shipping text out is never mistaken for a local one
                "offmachine": bool(getattr(det, "sends_text_offmachine", False)),
                "dataset": fingerprint(args.data)}
    if args.skip_if_done:
        for prior in out_dir.glob(f"{args.detector}-*.json"):
            try:
                got = json.loads(prior.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if all(got.get(k) == v for k, v in settings.items()):
                print(f"already measured ({prior.name}), skipping")
                return 0

    t0 = time.time()
    pos, neg = load(args.data, args.slice, args.limit)
    print(f"loaded: {len(pos)} injections, {len(neg)} clean "
          f"(slice {args.slice}, {time.time() - t0:.0f} s)", flush=True)

    # A forced aperture can exceed what a hosted detector accepts. That must stop the run, not be
    # discovered per-request: an API adapter returning 0.0 on refusal would file a complete,
    # plausible, empty result.
    limit = type(det).max_chars
    if forced and limit and (policy == "full" or (window or 0) > limit):
        raise SystemExit(
            f"[{args.detector}] accepts at most {limit} chars per document, but "
            f"{'the whole document' if policy == 'full' else window} was asked for.\n"
            f"  Drop --window (the detector cuts its own) or ask for at most {limit}.")

    # A detector that ships text to somebody else's service is a transfer of the corpus, not just
    # a measurement of it. The carriers here are scrubbed before publication, but our own licence
    # says that scrubbing is not guaranteed complete — so the decision is made explicitly, by the
    # person running the pass, rather than implied by their choice of adapter.
    if getattr(det, "sends_text_offmachine", False) and not args.allow_offmachine:
        raise SystemExit(
            f"[{args.detector}] sends every document to a third-party service "
            f"({getattr(det, 'endpoint', 'a remote API')}).\n"
            f"  That is a transfer of this corpus to someone else, separate from measuring on it.\n"
            f"  The carriers are scrubbed of contacts before publication and the scrubbing is\n"
            f"  checked by pattern search over the whole corpus, but it is not guaranteed complete.\n"
            f"  Pass --allow-offmachine to proceed.")

    det.settings = settings
    det.setup()
    docs = pos + neg
    t1 = time.time()
    part = partial_path({**settings, "binary": det.binary})
    done, seen_win = load_partial(part)
    scores, n_win = score_with_checkpoint(det, docs, part, done, seen_win)
    dt = time.time() - t1
    det.teardown()
    if len(scores) != len(docs):
        raise SystemExit(f"the detector returned {len(scores)} scores for {len(docs)} documents")
    check_score_type(det, scores, args.detector)
    paired = list(zip(docs, scores))
    check_score_floor(paired[len(pos):], args.fpr, args.detector, det.binary)
    win = ("whole document (the detector has no limit)" if policy == "full"
           else f"window {window} chars · {policy}"
                + (f" · overlap {overlap} sentences" if policy == "chunk" else "")
                + (" · SET BY FLAG" if forced else " · the detector's own limit"))
    print(f"run: {dt:.0f} s ({len(docs)/max(dt,1e-9):.0f} docs/s) · {win} · {n_win} windows",
          flush=True)

    res = evaluate(paired[:len(pos)], paired[len(pos):], target_fpr=args.fpr, binary=det.binary)
    res["points"] = {} if det.binary else at_points(paired[:len(pos)], paired[len(pos):])
    res.update(**settings, run_at=time.strftime("%Y-%m-%d %H:%M:%S"), binary=det.binary,
               notes=det.notes, display=getattr(det, "display", "") or args.detector,
               seconds=round(dt, 1), n_windows=n_win)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out_dir / f"{slug}.scores.jsonl").open("w", encoding="utf-8") as fh:
        for d, s in paired:
            fh.write(json.dumps({"id": d.id, "score": s}) + "\n")
    part.unlink(missing_ok=True)

    w, a = res["worst_family"], res["worst_action"]
    point = ("the detector's own point" if det.binary
             else f"one threshold over all clean documents @ FPR {pct(args.fpr)}")
    print(f"\n=== {args.detector} {version} · slice {args.slice} · {point} ===")
    if det.binary:
        print("  ⚠ binary: no threshold is chosen, the FPR is whatever it is; no AUC")
    print(f"  worst lever     {pct(w['recall'])}  {w['name']:20s} CI {pct(w['ci'][0])}-{pct(w['ci'][1])}")
    print(f"  worst objective {pct(a['recall'])}  {a['name']:20s} CI {pct(a['ci'][0])}-{pct(a['ci'][1])}")
    print(f"  range           {pct(res['attainable_range'][0],0)} - {pct(res['attainable_range'][1],0)}")
    print(f"  coverage r>=0.5 {pct(res['coverage_50'])}  ({res['n_cells']} cells)")
    print(f"  mean recall     {pct(res['mean_recall'])}  CI {pct(res['mean_ci'][0])}-{pct(res['mean_ci'][1])}")
    print("  FPR by carrier: " + "  ".join(
        f"{h} {pct(res['fpr'][h], 3)}" for h in sorted(res["fpr"])))
    print(f"\n-> {out_dir / (slug + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
