#!/usr/bin/env python3
"""One page per detector: what it caught, at what price, and where it is blind.

    python3 -m proba.report                       # a page for every run in results/
    python3 -m proba.report --only promptidote
    python3 -m proba.report --theme dark

Structure follows quadrat-ipi-eval's report; the prose, the vocabulary and the corpus facts are
this project's own, because they describe a different corpus. Drawing is `figures.py`, vendored
from that project unchanged.

WHY A PAGE AND NOT A ROW. A detector does not have a recall; it has a recall per price in false
alarms, per lever, per objective, per carrier — and the spread between those is usually wider than
the gap between two detectors' averages. A row hides that by construction. The page exists so the
number a reader quotes is one they have seen the shape of.

WHAT IT WILL NOT DO. It never computes a metric. Everything here is read from a result written by
`run.py` or `ingest.py`, so a page cannot disagree with the run it describes; if a figure looks
wrong, the run is wrong and the fix belongs upstream of here.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from . import figures as fg
from .metrics import wilson
from .paths import REPORTS, RESULTS as OUT

THEMES = ("auto", "light", "dark")

#: our three carriers, in the order the corpus lists them
CARRIERS = ("cards", "reviews", "web")
CARRIER_NOTE = {
    "cards": "marketplace product listings; a fifth run past one window",
    "reviews": "customer reviews — the shortest carrier, almost none reach one window",
    "web": "web pages — the longest, and the most often split into windows",
}

#: marginal axis -> (heading, what a difference along it does and does not mean)
MARGINALS = {
    "family": ("Recall by lever",
               "What makes the insertion get obeyed. This is the grid's rows, summarised."),
    "action": ("Recall by objective",
               "What the insertion asks for. This is the grid's columns, summarised."),
    "host_type": ("Recall by carrier",
                  "The three carriers do not admit the same cells — a third of the grid lives in "
                  "one carrier only — so a difference here mixes the carrier with which cells it "
                  "holds. Read it as 'what does this detector see on this kind of document', not "
                  "as a clean carrier effect."),
    "spliced_at": ("Recall by placement",
                   "Where the injection sits: appended at the end, at a paragraph break, or inside "
                   "a sentence. The mix here (56/24/20%) is this corpus's choice, not the world's; "
                   "a detector that only reads the tail shows up as a high `end` row rather than "
                   "as a good average."),
    "obfuscation": ("Recall by obfuscation",
                    "Homoglyph, zero-width and leetspeak substitutions against none. The classic "
                    "evasion question is which distortion a detector survives, so the field is "
                    "flattened to its kind."),
}

#: metric -> what it means to somebody who has never read a detection paper, saying what the
#: number is a share OF. "13.1%" alone is unreadable; "13.1% of the 16,800 injections" is not.
MEANING = {
    "recall": "share of the injections it caught",
    "false positives": "how often it flags a clean document — 0.1% is one false alarm per 1000 "
                       "clean documents, 1% is one per 100",
    "coverage": "in how many of the 92 attack types it catches at least half",
    "range over types": "worst attack type to best — how much an average depends on which types "
                        "you feed it",
    "weakest lever": "the construction it handles worst; what is left if the attacker picks it",
    "weakest objective": "the goal it handles worst, same reading",
}


def _pct(x, nd=1):
    return f"{x * 100:.{nd}f}%"


def _ci(c):
    return f"{_pct(c[0])}–{_pct(c[1])}" if c else "—"


def at_point(res, budget):
    """This run's numbers at one false-positive budget, or the run itself if that IS its point."""
    if budget is None:
        return res
    p = (res.get("points") or {}).get(f"{budget:g}")
    if p:
        return p
    return res if res.get("target_fpr") == budget else None


def curve_points(res):
    """This run's curve with a confidence interval on every point.

    The intervals are NOT stored in the result and do not need to be: a Wilson interval is a
    function of the rate and the count, both of which are already there, so they cost arithmetic
    rather than another pass over the corpus. Without them a line reads as exact, and two lines a
    point apart look like a difference when they are not one."""
    n_pos, n_neg = res.get("n_positives", 0), res.get("n_negatives", 0)
    out = []
    for p in res.get("curve") or []:
        q = dict(p)
        if n_pos:
            q["ci"] = wilson(round(p["recall"] * n_pos), n_pos)
        if n_neg:
            q["fpr_ci"] = wilson(round(p["fpr"] * n_neg), n_neg)
        out.append(q)
    return out


def binary_mark(res, colour=None):
    """A binary detector as one point with whiskers — it has no curve to interpolate through."""
    return (f'{res.get("display") or res.get("detector", "binary")} — own point',
            res.get("fpr_pooled", 0.0), res.get("mean_recall", 0.0),
            res.get("fpr_pooled_ci"), res.get("mean_ci"), None)


def cell_axes(cells):
    """(levers, objectives) in a fixed order, so every panel is drawn on the same axes.

    Derived from the cells present rather than from a constant: a slice can empty a row, and a
    panel drawn on axes the data does not fill reads as a detector's blind spot when it is the
    slice's."""
    fams, acts = [], []
    for c in cells:
        f, a = c.split("/", 1)
        if f not in fams:
            fams.append(f)
        if a not in acts:
            acts.append(a)
    return sorted(fams), sorted(acts)


def summary(res, points=(0.001, 0.01)):
    """The headline table: every claim paired with the operating point it was bought at.

    BOTH POINTS IN ONE TABLE, because a thresholded detector has no single recall. Printing only
    the budget a run happened to be launched at makes a choice look like a measurement."""
    def cols(fn):
        return {b: fn(at_point(res, b)) for b in points if at_point(res, b)}

    def rng(p):
        a, b = p["attainable_range"]
        return f"{_pct(a, 0)}–{_pct(b, 0)}"

    rows = [
        ("recall", cols(lambda p: f'{_pct(p["mean_recall"])} · CI {_ci(p["mean_ci"])}'),
         MEANING["recall"] + f' ({res["n_positives"]} of them)'),
        ("false positives", cols(lambda p: _pct(p.get("fpr_pooled", 0), 3)),
         MEANING["false positives"] + f' ({res["n_negatives"]} clean documents)'),
        ("coverage", cols(lambda p: f'{_pct(p["coverage_50"])} · '
                                    f'{round(p["coverage_50"] * p["n_cells"])} of {p["n_cells"]}'),
         MEANING["coverage"]),
        ("range over types", cols(rng), MEANING["range over types"]),
        ("weakest lever", cols(lambda p: f'{p["worst_family"]["name"]} '
                                         f'{_pct(p["worst_family"]["recall"])}'),
         MEANING["weakest lever"]),
        ("weakest objective", cols(lambda p: f'{p["worst_action"]["name"]} '
                                             f'{_pct(p["worst_action"]["recall"])}'),
         MEANING["weakest objective"]),
    ]
    return [r for r in rows if r[1]]


def build_figures(res, out_dir, theme, slug, floor=None):
    """Draw every diagram this page embeds, and nothing else.

    `out_dir` is the REPORTS root: `figures.save` appends `figures/` itself and returns the path
    already relative to the page, which is what the Markdown embeds."""
    a = at_point(res, 0.001) or res
    b = at_point(res, 0.01)
    cells = a.get("cells") or res["cells"]
    fams, acts = cell_axes(cells)
    figs = {}
    binary = bool(res.get("binary"))

    figs["cells"] = fg.save(fg.heat(
        cells, fams, acts, "all carriers",
        sub=("its own operating point" if binary else "left @0.1% FPR · right @1% FPR"),
        theme=theme, cells_b=(None if binary else (b or {}).get("cells")),
        modes=("", "") if binary else ("@0.1%", "@1%")), out_dir, f"{slug}-cells-all")

    by_host = a.get("cells_by_host") or res.get("cells_by_host") or {}
    for h in CARRIERS:
        if h in by_host:
            figs[f"cells:{h}"] = fg.save(fg.heat(
                by_host[h], fams, acts, h, sub=CARRIER_NOTE.get(h, ""), theme=theme,
                margins=False), out_dir, f"{slug}-cells-{h}")

    pts = curve_points(res)
    if pts:
        marks = [binary_mark(floor)] if floor and floor.get("binary") else []
        figs["curve"] = fg.save(
            fg.curve([(res.get("display") or res["detector"], pts, fg.LINE_COLORS[0])],
                     theme=theme, marks=marks,
                     unresolved=(1.0 / res["n_negatives"]) if res.get("n_negatives") else None),
            out_dir, f"{slug}-curve")

    for axis in MARGINALS:
        m = (a.get("marginals") or {}).get(axis) or {}
        if not m:
            continue
        rows = sorted(((k, v["recall"], tuple(v["ci"]), v["n"]) for k, v in m.items()),
                      key=lambda r: -r[1])
        base = None
        if floor and axis in (floor.get("marginals") or {}):
            fm = floor["marginals"][axis]
            n = sum(v["n"] for v in fm.values())
            base = sum(v["recall"] * v["n"] for v in fm.values()) / n if n else None
        figs[f"marg:{axis}"] = fg.save(
            fg.bars(rows, title=MARGINALS[axis][0], theme=theme,
                    baseline=base, baseline_label="regex floor" if base else ""),
            out_dir, f"{slug}-marg-{axis}")
    return figs


def render_md(res, figs, floor=None):
    """The page itself."""
    name = res.get("display") or res["detector"]
    binary = bool(res.get("binary"))
    pts = [0.001, 0.01] if not binary else [None]
    L = [f"# {name}", ""]

    ap = (f'{res.get("policy")} · {res.get("window")} chars'
          + (f' · overlap {res.get("overlap")} sentences' if res.get("policy") == "chunk" else "")
          if res.get("policy") != "full" else "whole document (no declared limit)")
    L += [f"**Measured on** proba-ipi build `{res.get('dataset')}`, slice `{res.get('slice')}` — "
          f"{res['n_positives']} injected and {res['n_negatives']} clean documents.  ",
          f"**Aperture** {ap}.  "]
    if res.get("segmenter"):
        L.append(f"**Sentence splitter** {res['segmenter']}.  ")
    notes = res.get("notes") or ""
    seg = res.get("segmenter")
    if seg and notes.endswith(seg):          # ingest carried it into both fields
        notes = notes[: -len(seg)].rstrip(" ·")
    if notes:
        L.append(f"**Detector** {notes}.  ")
    if res.get("offmachine"):
        L.append("**This pass sent every document to a third-party service.**  ")
    if res.get("imported"):
        L.append("*Scores predate this harness; the metrics below were recomputed from them, "
                 "no model was re-run.*  ")
    if res.get("dataset_note"):
        L.append(f"*{res['dataset_note']}*  ")
    L.append("")

    L += ["## The headline, with its price", ""]
    heads = ["metric"] + ([f"at {p*100:g}% false positives" for p in pts] if not binary
                          else ["at its own operating point"])
    L += ["| " + " | ".join(heads + ["what it means"]) + " |",
          "|" + "---|" * (len(heads) + 1)]
    for metric, cols, meaning in summary(res, pts if not binary else (None,)):
        vals = [cols.get(p, "—") for p in (pts if not binary else (None,))]
        L.append("| " + " | ".join([metric] + vals + [meaning]) + " |")
    L.append("")
    if binary:
        L += ["A binary detector has one operating point and no curve: it fires or it does not, "
              "and its false-positive rate is whatever that costs. It is not on the same footing "
              "as a thresholded detector and its tiles are drawn whole rather than split.", ""]
    else:
        L += ["Read the second column against the first. The same system at two prices is two "
              "different systems to whoever has to work the alert queue.", ""]

    L += ["## Where it is blind", "",
          f"![the lever × objective grid]({figs['cells']})", "",
          "Each tile is one of the 92 attack types"
          + ("" if binary else ", with its recall at 0.1% false positives on the left and at 1% "
                               "on the right")
          + ". The last row and column are the marginals — a row's mean **is** the lever's recall, "
            "over the same hits and the same n. Where the two halves of a tile differ sharply, "
            "that cell's recall is a fact about the budget rather than about the detector.", ""]

    per = [(h, figs[f"cells:{h}"]) for h in CARRIERS if f"cells:{h}" in figs]
    if per:
        L += ["### The same grid, one carrier at a time", ""]
        for h, path in per:
            L += [f"**{h}** — {CARRIER_NOTE.get(h,'')}", "", f"![{h}]({path})", ""]
        L += ["A cell missing from a carrier is structure, not a gap in the data: a third of the "
              "grid is built in one carrier only.", ""]

    if "curve" in figs:
        L += ["## The whole trade-off, not a point", "",
              f"![recall against the false-positive budget]({figs['curve']})", "",
              "Recall as a function of what a false alarm is allowed to cost, log-x because the "
              "question lives in the first decade: between 0.01% and 0.1% false alarms is the "
              "difference between a filter that can run on a firehose and one that cannot. The "
              "band is the 95% interval; the shaded strip on the left is below what "
              f"{res['n_negatives']} clean documents can express at all — a rate resting on a "
              "handful of false alarms is not a rate. A line can step sideways where ties at the "
              "cut move the realised rate without moving recall.", ""]

    for axis, (heading, note) in MARGINALS.items():
        key = f"marg:{axis}"
        if key in figs:
            L += [f"## {heading}", "", note, "", f"![{heading}]({figs[key]})", ""]

    if floor and floor.get("detector") != res.get("detector"):
        f_rec = floor["mean_recall"]
        rec = (at_point(res, 0.001) or res)["mean_recall"]
        verdict = ("**below the floor** — on this corpus it is not detecting, it is matching "
                   "boilerplate" if rec < f_rec else
                   f"**{rec/f_rec:.1f}× the floor**" if f_rec else "above the floor")
        L += ["## Against the triviality floor", "",
              f"A handful of quotable phrases catches {_pct(f_rec)} of these injections at "
              f"{_pct(floor.get('fpr_pooled',0),3)} false positives. This detector is {verdict}.",
              "",
              "The floor is not competing — nobody ships it. It exists so that every number above "
              "has something to be read against, and so that a detector which has learned nothing "
              "beyond the corpus's vocabulary is visible as such.", ""]

    L += ["---", "",
          f"<sub>Generated by `python3 -m proba.report` from `{res.get('_file','a result')}`. "
          f"No metric is computed here; every figure is drawn from that file.</sub>"]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(OUT))
    ap.add_argument("--out", default=str(REPORTS))
    ap.add_argument("--only", help="comma-separated detector names")
    ap.add_argument("--theme", default="auto", choices=THEMES)
    a = ap.parse_args()

    results = pathlib.Path(a.results)
    out = pathlib.Path(a.out)
    want = {s.strip() for s in a.only.split(",")} if a.only else None

    runs = {}
    for f in sorted(results.glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "mean_recall" not in r:
            continue
        r["_file"] = f.name
        d = r.get("detector")
        if d not in runs or r.get("run_at", "") >= runs[d].get("run_at", ""):
            runs[d] = r
    if not runs:
        raise SystemExit(f"no results in {results}")

    # The floor is a comparison column on every page, so it must have been measured on the SAME
    # build; one from another corpus would make the delta meaningless.
    floor = runs.get("floor")
    written = []
    for det, res in sorted(runs.items()):
        if want and det not in want:
            continue
        base = floor if (floor and floor.get("dataset") == res.get("dataset")) else None
        slug = pathlib.Path(res["_file"]).stem
        figs = build_figures(res, out, a.theme, slug, base)
        out.mkdir(parents=True, exist_ok=True)
        page = out / f"{slug}.md"
        page.write_text(render_md(res, figs, base), encoding="utf-8")
        written.append(page)
        print(f"  {det:14s} {len(figs):2d} figures -> {page.name}", flush=True)

    # An index, because a directory of slugs is not a table of contents: the reader wants to
    # know which page to open, and that decision is made on the headline numbers.
    if written:
        # ONE SHARED CURVE, and it is the figure the index exists for. A ranking at 0.1% cannot
        # say whether a detector is behind everywhere or only at the budget somebody chose, and
        # those are different findings. Scored detectors are lines; a binary one is a point with
        # whiskers, because interpolating through it would invent a budget it does not offer.
        scored = sorted((r for r in runs.values() if not r.get("binary") and r.get("curve")),
                        key=lambda r: -(at_point(r, 0.001) or r)["mean_recall"])
        series = [((r.get("display") or r["detector"]), curve_points(r),
                   fg.LINE_COLORS[i % len(fg.LINE_COLORS)])
                  for i, r in enumerate(scored)]
        marks = [binary_mark(r) for r in runs.values() if r.get("binary")]
        n_neg = next((r.get("n_negatives") for r in runs.values() if r.get("n_negatives")), None)
        shared = fg.save(fg.curve(series, theme=a.theme, marks=marks,
                                  unresolved=(1.0 / n_neg) if n_neg else None),
                         out, "comparison-curve")

        rows = sorted(((r.get("display") or d, d, r) for d, r in runs.items()
                       if not want or d in want),
                      key=lambda x: -((at_point(x[2], 0.001) or x[2])["mean_recall"]))
        L = ["# Detectors measured on proba-ipi", "",
             "One page each. Recall is over all "
             f"{rows[0][2]['n_positives']} injections, at a single threshold over the whole "
             f"clean pool — see any page for what that costs.", "",
             (f"![recall against the false-positive budget, every detector]({shared})\n"
              if shared else ""),
             ("Every detector on one axis, because a ranking at a single budget cannot say "
              "whether one is behind everywhere or only where somebody set the threshold. On this "
              "corpus the order of the three working detectors holds across the whole range — what "
              "changes is how much each one buys with the extra budget, and they differ by more "
              "than the gaps between them at either end. The only crossings are between "
              "English-only models at recalls too small for a swap to mean anything.\n"
              if shared else ""),
             "| detector | @0.1% FPR | @1% FPR | cells @0.1% | page |",
             "|---|---:|---:|---:|---|"]
        for label, det, r in rows:
            a = at_point(r, 0.001) or r
            b = at_point(r, 0.01)
            slug = pathlib.Path(r["_file"]).stem
            binary = " *(binary, own point)*" if r.get("binary") else ""
            L.append(f"| {label}{binary} | {_pct(a['mean_recall'])} | "
                     f"{_pct(b['mean_recall']) if b else '—'} | "
                     f"{round(a['coverage_50'] * a['n_cells'])} of {a['n_cells']} | "
                     f"[{slug}]({slug}.md) |")
        L += ["", "<sub>Generated by `python3 -m proba.report`.</sub>"]
        (out / "README.md").write_text("\n".join(L), encoding="utf-8")
        print(f"  {'index':14s}    -> README.md", flush=True)

    print(f"\n{len(written)} pages -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
