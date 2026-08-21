#!/usr/bin/env python3
"""gbv/mdeberta-ru-prompt-injection — a Russian-language prompt-injection classifier.

    python3 -m proba.run --detector gbv

The only detector in the roster trained for Russian, which is why it sits second on this corpus
and would be meaningless on an English one. Its class is named `prompt_injection`, not
`INJECTION`, so the label is declared here rather than left to the base class's default.
"""
from __future__ import annotations

from ..detector import register
from ._hf import _HFClassifier


@register("gbv", version="mdeberta-ru")
class GBV(_HFClassifier):
    display = "gbv mDeBERTa-ru prompt-injection"
    model_id = "gbv/mdeberta-ru-prompt-injection"
    revision = "546644285dd6c47d40cfcb6a80176dac7ce482c0"
    positive_label = "PROMPT_INJECTION"
