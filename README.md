# proba-ipi-eval — the evaluation harness

**Write one class. Get a report for your prompt-injection detector on Russian documents that is
comparable, line for line, with every other detector measured on the same corpus — and with the
English numbers from [Quadrat-IPI](https://huggingface.co/datasets/mihailgribov/quadrat-ipi).**

```bash
python3 -m proba.run --list                     # every registered adapter
python3 -m proba.run --detector floor           # measure one (this one needs no model)
python3 -m proba.report                         # a page per detector: the grid, the marginals
python3 -m proba.compare --against floor        # everyone at one budget, from saved scores
python3 -m proba.table                          # the dataset card's tables
```

The corpus is [privettoha/proba-ipi](https://huggingface.co/datasets/privettoha/proba-ipi): 16,800
injected and 81,000 clean Russian documents on a 92-cell grid of 10 levers × 10 objectives. It is
fetched on first run at a pinned tag.

## Why the measurement code is someone else's, unchanged

`metrics.py` and `window.py` are taken from
[quadrat-ipi-eval](https://github.com/mihail-gribov/quadrat-ipi-eval) (Apache-2.0) and kept
unchanged, and `detector.py` differs only in resolving its own package name. That is the point of
this harness rather than an economy: the corpus here mirrors Quadrat-IPI's grid so that a detector
can be measured in both languages and the difference read as language rather than as ruler. That
only holds while the code choosing the threshold, defining a cell and computing an interval is the
same code. See `NOTICE`.

## What a number here means

Recall at a target false-positive rate, with **one threshold over the whole pooled clean set** —
not one per carrier. A deployment has a single operating point and cannot know a document's
carrier before reading it, so a per-carrier threshold would hand the detector an oracle it will not
have. The per-carrier false-positive rate is therefore reported, never held fixed.

Two operating points are always reported: **0.1%**, where a document filter is actually deployed,
and **1%**, where most published prompt-injection numbers are quoted. They do not rank detectors
the same way, so quoting one alone is a choice about who wins.

**The denominator is all 16,800 injections**, including the ~14% quota fill (`inj_verified:
false` — payloads a cell was topped up with that the blind judge did not confirm). This matches
Quadrat-IPI, whose published rows are also over all 16,800; a recall over confirmed injections
only is available as `--slice verified` and runs 2–5 points higher on the strong detectors.

## Adding a detector

One class, one method:

```python
from proba.detector import Detector, register

@register("my-detector", version="1.2")
class MyDetector(Detector):
    max_chars = 2000                 # your context limit; omit for none

    def setup(self):                 # optional
        self.model = load_model()

    def score(self, docs):           # yield one float per document, in order
        for d in docs:
            yield self.model.predict(d.text)
```

Then `python3 -m proba.run --detector my-detector --adapters path/to/dir`.

**The aperture is yours.** The harness hands `score` whole documents. Declare `max_chars` and the
base class splits for you (sentence-aligned windows, four-sentence overlap, folded by max), or
override `score_documents` and split the way your product does. What it will not do is impose a
window on a detector that did not ask for one — how to read a long document is part of the
detector, and a harness that chunked for everyone would report its own splitter's recall.

**Return a continuous score**, higher = more injection-like. The whole protocol sets a threshold at
a target FPR, and a verdict cannot be thresholded. If your detector really is binary, say
`binary = True` and it is reported at its own operating point instead of pretending a curve exists.

## One caveat specific to Russian

`window.sentence_spans` prefers `blingfire` and falls back to a regex — and that regex requires a
capital letter after the stop, from a **Latin-only** character class. On Russian it therefore finds
almost no boundaries and degenerates into fixed 2000-character slices. The published proba-ipi
numbers were taken in exactly that state, so `blingfire` is an optional extra here rather than a
core dependency: installing it would make a fresh run disagree with the table it is meant to
reproduce.

Which splitter actually ran is recorded as `segmenter` in every result, and it is part of the run's
identity, so two runs that split differently cannot satisfy each other's `--skip-if-done`. The
effect is bounded and measured: 142 of 16,800 injections sit across a 2000-character boundary at
all (0.85%), and 80% of injected documents are shorter than one window.

## Working from scores instead of models

Every pass writes `results/<slug>.json` and `<slug>.scores.jsonl`. A threshold is a number chosen
over those scores afterwards, so moving an operating point, re-cutting a slice or redrawing the
table costs a second of arithmetic, not a re-run:

```bash
python3 -m proba.table --results eval-out/results     # the card's tables
python3 -m proba.ingest                               # import pre-harness runs into this shape
```

`report` draws the page: the headline paired with the price it was bought at, the 92-cell grid
with each tile split between the two operating points, the same grid per carrier, marginals by
lever / objective / carrier / placement / obfuscation, and the detector against the regex floor.
It computes nothing — every figure is read from a result file, so a page cannot disagree with the
run it describes. Drawing is `figures.py`, vendored unchanged (see NOTICE).

`compare` re-thresholds every saved detector to one budget. The case it exists for is a **binary**
detector: the regex floor fires at 0.021% false positives, and reading it beside a column taken at
0.1% hands the others five times its budget. At the floor's own rate, two of the English models
fall to 0.0% — they lose to a handful of regexes when the budget is matched.

`rescore` re-scores only documents whose text changed and splices them into a finished run,
recomputing every metric (a changed negative can move the corpus-wide threshold). It is what keeps
a released corpus reproducible from its own files after an edit.

`ingest` is how the seven detectors measured before this harness existed became harness results:
their scores were kept, so the metrics were recomputed from them without running a model. Runs
imported that way carry `imported: true` and a `provenance` block naming the score files they came
from, and they deliberately do **not** report `seconds` or `n_windows` — the original passes did
not record a comparable figure, and an imported run must not report a timing that never happened.

## Guards

A run refuses to file a number it cannot honestly produce:

* **score type** — a detector that declares a score but returns only 0.0/1.0 is stopped. Threshold
  selection on a verdict vector picks tau = 0, `score >= tau` flags the whole corpus, and the run
  reports 100% recall at 100% FPR, which reads like a triumph.
* **score floor** — if many clean documents share the same maximal score, no threshold separates
  them and any operating point below that tie is not a measurement. Asking for one is an error, not
  a rounding issue.
* **corpus fingerprint** — every result records a hash of the `id`/`label`/`text` it read, so saved
  scores cannot be silently reused across a rebuild.
* **checkpointing** — a long pass writes each batch as it lands and resumes from it, so a crash at
  document 97,000 does not throw the pass away.
