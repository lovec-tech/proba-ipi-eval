#!/usr/bin/env python3
"""deepset DeBERTa-v3-base injection classifier.

    python3 -m proba.run --detector deepset

English DeBERTa; see the note in `protectai.py` on what an English-only tokenizer means for a row
measured on Russian carriers.
"""
from __future__ import annotations

from ..detector import register
from ._hf import _HFClassifier


@register("deepset", version="deberta-v3-base")
class Deepset(_HFClassifier):
    display = "deepset DeBERTa v3 base injection"
    model_id = "deepset/deberta-v3-base-injection"
    revision = "80dda00d0b0d9a03917a7685e2ddbcd28e04dbb1"
    positive_label = "INJECTION"
