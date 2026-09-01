# docs/ — the page

The Russian article about the corpus and the run, served by GitHub Pages from this
directory: **https://lovec-tech.github.io/proba-ipi-eval/**

```
index.html       the page (generated -- edit build_page.py, not this)
build_page.py    the generator
results/         the eight runs, metrics only (the .scores.jsonl stay in the dataset)
figures/         the seven figures the article uses, copied from the release
.nojekyll        serve the files as they are
```

## Rebuilding

```bash
python3 docs/build_page.py
```

Every number on the page is read from `results/*.json` — the same files the dataset
ships — so the page cannot disagree with the run it describes. Two figures are not from
this corpus and are declared as constants at the top of `build_page.py` with their source:
Quadrat-IPI's English column, and the trivial baseline (whose run predates the harness and
has no score file at all — the page says so where it appears). The one external row a local
copy exists for is checked against it at build time.

To pull in a newer release of the corpus:

```bash
python3 docs/build_page.py --sync-from ../path/to/release_proba_ipi
```

That refreshes `results/` and the figures the article uses, then rebuilds. Nothing here
imports `proba/` — `metrics.py`, `window.py` and `figures.py` are vendored from
[quadrat-ipi-eval](https://github.com/mihail-gribov/quadrat-ipi-eval) unchanged and must
stay diffable against upstream (see `NOTICE`).

## Enabling Pages

Settings → Pages → Source: *Deploy from a branch*, branch `main`, folder `/docs`.
