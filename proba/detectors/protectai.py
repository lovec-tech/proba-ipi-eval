#!/usr/bin/env python3
"""ProtectAI DeBERTa-v3-base prompt-injection v2.

    python3 -m proba.run --detector protectai

English DeBERTa: no Russian in its tokenizer's vocabulary, so its row here is largely a statement
about that rather than about the model's quality. Reported anyway, and labelled, because leaving
it out would let a reader assume the roster was Russian-capable throughout.
"""
from __future__ import annotations

from ..detector import register
from ._hf import _HFClassifier


@register("protectai", version="deberta-v3-base-v2")
class ProtectAI(_HFClassifier):
    display = "ProtectAI DeBERTa v3 base v2"
    model_id = "protectai/deberta-v3-base-prompt-injection-v2"
    revision = "90c9989b1a342275dd0d1a95aad283c04e075671"
    positive_label = "INJECTION"
