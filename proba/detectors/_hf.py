#!/usr/bin/env python3
"""Shared base for the published transformer classifiers — not a detector itself.

Leading underscore on purpose: the runner auto-imports every module in this directory, and this
one registers nothing. Each detector gets its own file so its adapter can be replaced, version
-bumped or contributed by its author without touching a file its competitors live in.

Structure follows quadrat-ipi-eval's `_hf.py` (Apache-2.0). The models here are the checkpoints
these projects publish, at their default configuration, pinned to the exact revisions the
published proba-ipi table was measured with.

    pip install torch transformers

THE WINDOW IS DECLARED, NOT IMPOSED. These models stop at 512 tokens (~2000 characters) and a
fifth of the card carrier and nearly half the web carrier is longer, so handed a whole document
they would silently read its head — which measures truncation and reports it as detection. The
limit is declared as `max_chars`; the base class splits on sentence boundaries with a
four-sentence overlap and folds the windows back by max.

ONE CAVEAT SPECIFIC TO THIS CORPUS. `window.sentence_spans` prefers blingfire and falls back to a
regex that requires a capital letter after the stop — and that regex is **Latin-only**, so on
Russian text it finds almost no boundaries and the fallback degenerates into fixed 2000-character
slices. The published proba-ipi numbers were taken in exactly that state (blingfire could not load
on the measuring machine), so a machine where blingfire *does* import will cut differently. The
effect is bounded and small — 142 of 16,800 injections sit across a 2000-character boundary at all
(0.85%), and 80% of injected documents are shorter than one window — but it is real, so the
runner records which segmenter was active in every result rather than leaving it to be inferred.
"""
from __future__ import annotations

import os
import time

from ..detector import Detector
from ..window import WINDOW_512

#: Defaults for a 512-token DeBERTa. Both are per-adapter: a long-context model truncated at 512
#: would be handed the aperture it declared and shown a fraction of it.
BATCH = int(os.environ.get("PROBA_HF_BATCH", "32"))
MAXLEN = 512
PROGRESS_EVERY = int(os.environ.get("PROBA_HF_PROGRESS", "2000"))


class _HFClassifier(Detector):
    """One transformer, scored by the probability of its injection class."""

    _docs_in_batch: int = 0

    #: 512 tokens, in the characters the harness measures in. The base class does the splitting.
    max_chars = WINDOW_512
    model_id: str = ""
    #: hub revision. Pinned: `main` moves, and a measured row has to name what it measured.
    revision: str | None = None
    #: label whose probability is the score, resolved against the checkpoint's own id2label so a
    #: model that orders its classes differently cannot silently invert the metric.
    positive_label: str = "INJECTION"
    fallback_index: int = 1
    max_tokens: int = MAXLEN
    batch: int = BATCH
    #: Off everywhere by default. An adapter that genuinely needs vendor code sets it and says why
    #: in its own file, where the choice can be read and pinned.
    trust_remote_code: bool = False

    def _load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=self.trust_remote_code)
        tok = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=self.trust_remote_code)
        return model, tok

    def setup(self):
        import torch

        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model, self.tok = self._load()
        self.model = self.model.to(self.device).eval()
        id2label = getattr(self.model.config, "id2label", {}) or {}
        want = self.positive_label.upper()
        self.index = next((i for i, lab in id2label.items()
                           if str(lab).upper().startswith(want[:4])), self.fallback_index)
        got = str(id2label.get(self.index, "?"))
        if want[:4] not in got.upper():
            raise SystemExit(
                f"[{self.model_id}] expected a class starting {want[:4]!r}, but index "
                f"{self.index} is {got!r}; the checkpoint has "
                f"{sorted(str(v) for v in id2label.values())}.\n"
                f"  Scoring the wrong class inverts the metric silently — fix the adapter.")
        rev = f"@{self.revision[:8]}" if self.revision else ""
        self.notes = f"{self.model_id}{rev} · {self.device} · positive={got}"

    def score_documents(self, docs):
        docs = list(docs)
        self._docs_in_batch = len(docs)
        return super().score_documents(docs)

    def score(self, docs):
        # `docs` here are WINDOWS — the base class splits before scoring — so what is counted and
        # reported is windows, and the document figure comes from the runner's own offsets.
        base = getattr(self, "progress_offset", 0)
        total = getattr(self, "progress_total", 0)
        if getattr(self, "_t0", None) is None:
            self._t0 = time.time()
            self._win_base = 0
        n, batch = 0, []
        for d in docs:
            batch.append(d.text)
            if len(batch) == self.batch:
                yield from self._run(batch)
                n += len(batch)
                batch = []
                self._tick(base, total, n)
        if batch:
            yield from self._run(batch)
            n += len(batch)
            self._tick(base, total, n, last=True)
        self._win_base += n

    def _tick(self, base, total, n, last=False):
        if not total:
            return
        if not last and n % PROGRESS_EVERY >= self.batch:
            return
        el = time.time() - self._t0
        wins = self._win_base + n
        rate = wins / el if el else 0
        per_doc = (self._win_base / base) if base else 0
        if per_doc and rate:
            seen = base + min(self._docs_in_batch, n / per_doc)
            left = (total - seen) * per_doc / rate / 60
            tail = f"{rate:.0f} windows/s · ~{max(0, left):.0f} min left"
        else:
            tail = f"{rate:.0f} windows/s · -- left (first batch)"
        print(f"  {self.model_id.split('/')[-1]} · documents {base}/{total} · "
              f"{n} windows in batch, {tail}", flush=True)

    def _run(self, texts):
        torch = self.torch
        enc = self.tok(texts, return_tensors="pt", truncation=True,
                       max_length=self.max_tokens, padding=True).to(self.device)
        with torch.no_grad():
            probs = self.model(**enc).logits.softmax(-1)[:, self.index]
        out = probs.float().cpu().tolist()
        # MPS keeps every intermediate of a padded batch alive; DeBERTa's disentangled attention is
        # O(seq^2) in memory, and a full-length batch will exhaust the shared pool without this.
        if self.device == "mps":
            torch.mps.empty_cache()
        return out

    def teardown(self):
        self.model = None
        if getattr(self, "device", "") == "cuda":
            self.torch.cuda.empty_cache()
