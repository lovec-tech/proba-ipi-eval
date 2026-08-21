#!/usr/bin/env python3
"""Proventra, mDeBERTa-v3-base prompt-injection classifier.

    python3 -m proba.run --detector proventra

Multilingual by construction (mDeBERTa) — and on this corpus that is the point rather than a
footnote: it is the one detector measured on both quadrat-ipi (English) and proba-ipi (Russian),
so the difference between its two rows is the closest thing available to a read on how much of a
score is language and how much is the set.
"""
from __future__ import annotations

from ..detector import register
from ._hf import _HFClassifier


@register("proventra", version="mdeberta-v3-base")
class Proventra(_HFClassifier):
    display = "Proventra mDeBERTa v3 base"
    model_id = "proventra/mdeberta-v3-base-prompt-injection"
    #: pinned: the revision the published proba-ipi table was measured with
    revision = "b8a89d3096cf11a71d57b283c854f5ae2ed3df83"
    positive_label = "INJECTION"
